"""Directional support sampling: detector."""
import torch
from torch import nn
from torch.nn import functional as F
from .transport import ConditionalCDF
from .decoder_bridge import GeometryTransformer
from .density_router import SemanticStudent, DemandDensityRouter
from .dual_reader import DualPeriodicReader


class CoupledCanvasDetector(nn.Module):
    def __init__(self, native_model, frozen_class_text, residual_bins, levels,
                 canvas, nodes, reader_groups, semantic_temperature,
                 allocation_mode='thickness', density_floor=.15,
                 demand_source='corrected', ridge=1e-8):
        super().__init__()
        if allocation_mode not in ('uniform', 'area', 'thickness', 'svd'):
            raise ValueError('Unknown allocation comparison arm')
        self.backbone, self.encoder = native_model.backbone, native_model.encoder
        self.transformer = GeometryTransformer(native_model.decoder, residual_bins)
        for head in self.transformer.native.decoder.geometry_heads:
            head.ridge = ridge
        self.demand_source = demand_source
        self.gradient_checkpointing = False
        # All comparison arms preserve one common set of query identities.
        self.transformer.native.num_denoising = 0
        hidden = self.transformer.native.hidden_dim
        self.semantic = SemanticStudent(hidden, frozen_class_text, semantic_temperature)
        self.semantic_class_projection = nn.Linear(len(frozen_class_text), hidden, bias=False)
        nn.init.zeros_(self.semantic_class_projection.weight)
        self.router = DemandDensityRouter(hidden, len(frozen_class_text), nodes, floor=density_floor)
        self.reader = DualPeriodicReader(hidden, levels, reader_groups)
        self.canvas, self.nodes, self.allocation_mode = canvas, nodes, allocation_mode

    def encode(self, image):
        if self.training and self.gradient_checkpointing and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint
            features = checkpoint(self.backbone,image,use_reentrant=False)
        else:
            features = self.backbone(image)
        encoded = self.encoder(features)
        return encoded if isinstance(encoded, tuple) else (encoded, None)

    def score_query(self, state, semantic, layer_index):
        center = state['center']
        valid = ((center>=0)&(center<=1)).all(-1)
        grid = center[...,None,:].to(semantic['response'].dtype)*2-1
        response = F.grid_sample(semantic['response'],grid,align_corners=True,
                                 mode='bilinear',padding_mode='zeros')[:,:,:,0].transpose(1,2)
        response = response*valid[...,None].to(response.dtype)
        classification = state['query_features']+self.semantic_class_projection(response)
        return self.transformer.native.dec_score_head[layer_index](classification)

    def detail_decode(self, initial_state, context_features, detail_features, transports, semantic):
        decoder = self.transformer.native.decoder
        native = self.transformer.native
        queries, previous = initial_state['query_features'], initial_state
        reference_center = initial_state['reference_center']
        states = []
        for index, layer in enumerate(decoder.layers):
            references = previous['boxes'].detach() if self.training else previous['boxes']
            position = native.query_pos_head(references)
            key = layer.with_pos_embed(queries, position)
            attended = layer.self_attn(key, key, value=queries, attn_mask=None)[0]
            queries = layer.norm1(queries+layer.dropout1(attended))
            # Nonlinear detail coordinates are read through U at each original point.
            evidence = self.reader(queries, previous, context_features, detail_features, transports)
            queries = layer.norm2(queries+layer.dropout2(evidence))
            queries = layer.norm3(queries+layer.dropout3(layer.forward_ffn(queries)))
            state = decoder.geometry_heads[index](queries, reference_center, previous)
            state['reference_center'] = reference_center
            state['query_features'] = queries
            state['query_logits'] = self.score_query(state,semantic,index)
            states.append(state)
            previous = state
            if not self.training and index == decoder.eval_idx:
                break
        return states

    def forward(self, original, targets=None):
        if original.shape[-2] != original.shape[-1]:
            raise ValueError('Inputs and GT must share the padded original square frame')
        size = (self.canvas,self.canvas)
        context = F.interpolate(original,size=size,mode='bilinear',align_corners=True)
        context_features, initialization_aux = self.encode(context)
        coarse = self.transformer(context_features,targets)
        semantic = self.semantic(context_features[0])
        for index,state in enumerate(coarse['geometry_layers']):
            state['query_logits'] = self.score_query(state,semantic,index)
        initial_state = coarse['geometry_layers'][-1]
        self.set_demand(initial_state)
        if self.allocation_mode == 'uniform':
            density = original.new_ones(original.shape[0],self.nodes,self.nodes,dtype=torch.float32)
            transports = [ConditionalCDF(rho) for rho in density]
        else:
            density,transports = self.router(context_features[0],semantic['response'],initial_state)
        detail = torch.cat([transport.resample(original[image:image+1],size)
                            for image,transport in enumerate(transports)],0)
        detail_features,_ = self.encode(detail)
        fine = self.detail_decode(initial_state,context_features,detail_features,transports,semantic)
        final = fine[-1]
        output = dict(pred_boxes=final['boxes'],pred_logits=final['query_logits'],
                      context_geometry_layers=coarse['geometry_layers'],geometry_layers=fine,
                      semantic=semantic,density=density,transports=transports,
                      allocation_mode=self.allocation_mode,canvas=self.canvas)
        if self.training and initialization_aux is not None:
            output['encoder_aux_pred'] = initialization_aux
        if self.training and coarse.get('aux_outputs'):
            output['initial_prediction'] = coarse['aux_outputs'][-1]
        return output

    def set_demand(self, state):
        from .geometry import demand
        if self.demand_source == 'fixed':
            state['demand'] = torch.zeros_like(state['demand'])
        elif self.demand_source == 'raw':
            p = state['center_logits'].softmax(-1)
            bins = self.transformer.native.decoder.geometry_heads[0].bins
            mean = (p*bins).sum(-1)
            variance = (p*(bins-mean[...,None]).square()).sum(-1)
            ridge = self.transformer.native.decoder.geometry_heads[0].ridge
            state['demand'] = demand(torch.diag_embed(variance),state['shape'],ridge)
        elif self.demand_source == 'scale':
            inverse = state['sizes'].amin(-1).clamp_min(1e-8).reciprocal()
            order = inverse.argsort(-1).argsort(-1).to(inverse.dtype)
            bounded = order/(order.shape[-1]+1)
            state['demand'] = bounded/(1-bounded)
        elif self.demand_source != 'corrected':
            raise ValueError('Unknown demand source')
