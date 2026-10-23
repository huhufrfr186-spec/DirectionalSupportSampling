"""Directional support sampling: decoder bridge."""
import torch
from torch import nn
from .query_geometry import QueryGeometryHead


class GeometryDecoderBridge(nn.Module):
    def __init__(self, native_decoder, bins):
        super().__init__()
        self.layers = native_decoder.layers
        self.eval_idx = native_decoder.eval_idx
        self.geometry_heads = nn.ModuleList([
            QueryGeometryHead(native_decoder.hidden_dim, bins) for _ in self.layers
        ])
        self.last_states = []

    def forward(self, tgt, ref_points_unact, memory, memory_spatial_shapes,
                memory_level_start_index, bbox_head, score_head, query_pos_head,
                attn_mask=None, memory_mask=None):
        # Keep the native call signature; the directional support geometry heads replace bbox_head.
        initial = torch.cat((ref_points_unact[..., :4].sigmoid(),
                             ref_points_unact[..., 4:5]), -1)
        reference_center = initial[..., :2]
        phase = 4 * initial[..., 4]
        previous = dict(phase=torch.stack((phase.cos(), phase.sin()), -1),
                        center_logits=tgt.new_zeros(*tgt.shape[:-1], 2,
                                                    self.geometry_heads[0].bins.numel()))
        references, queries = initial, tgt
        boxes, logits, states = [], [], []
        for i, layer in enumerate(self.layers):
            queries = layer(queries, references.unsqueeze(2), memory,
                            memory_spatial_shapes, memory_level_start_index,
                            attn_mask, memory_mask, query_pos_head(references))
            state = self.geometry_heads[i](queries, reference_center, previous)
            state['reference_center'] = reference_center
            state['query_features'] = queries
            state['query_logits'] = score_head[i](queries)
            states.append(state)
            if self.training or i == self.eval_idx:
                boxes.append(state['boxes'])
                logits.append(state['query_logits'])
            if not self.training and i == self.eval_idx:
                break
            references = state['boxes'].detach() if self.training else state['boxes']
            previous = state
        self.last_states = states
        return torch.stack(boxes), torch.stack(logits)


class GeometryTransformer(nn.Module):
    def __init__(self, native_transformer, bins):
        super().__init__()
        self.native = native_transformer
        self.native.decoder = GeometryDecoderBridge(self.native.decoder, bins)

    def forward(self, features, targets=None):
        output = self.native(features, targets)
        states = self.native.decoder.last_states
        if 'dn_meta' in output:
            start = output['dn_meta']['dn_num_split'][0]
            states = [{name: value[:, start:] for name, value in state.items()}
                      for state in states]
        output['geometry_layers'] = states
        return output
