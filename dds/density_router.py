"""Directional support sampling: density router."""
import torch
from torch import nn
from torch.nn import functional as F
from .transport import ConditionalCDF
from .geometry import bounded_density


class SemanticStudent(nn.Module):
    def __init__(self, channels, frozen_class_text, temperature):
        super().__init__()
        text = F.normalize(torch.as_tensor(frozen_class_text).detach().float(), dim=-1)
        self.register_buffer('class_text', text)
        self.temperature = temperature
        self.local = nn.Conv2d(channels, text.shape[-1], 1)
        self.global_projection = nn.Linear(channels, text.shape[-1])

    def forward(self, features):
        scene = F.normalize(self.global_projection(features.mean((-2,-1))), dim=-1)
        local = F.normalize(self.local(features)+scene[...,None,None], dim=1)
        scores = torch.einsum('bdhw,cd->bchw', local, self.class_text)/self.temperature
        return dict(response=scores.softmax(1), local=local, scene=scene)


def stopped_query_maps(state, nodes):
    # Narrow Gaussian precision matrices can overflow half-precision einsum.
    # Keep the quadratic form and normalized regional pooling in FP32.
    with torch.autocast(device_type=state['center'].device.type, enabled=False):
        return _stopped_query_maps_fp32(state, nodes)


def _stopped_query_maps_fp32(state, nodes):
    center, shape = state['center'].detach().float(), state['shape'].detach().float()
    sizes, phase = state['sizes'].detach().float(), state['phase'].detach().float()
    need = state['demand'].detach().float()
    foreground = state['query_logits'].detach().float().sigmoid().amax(-1)
    axis = torch.linspace(0,1,nodes,device=center.device,dtype=center.dtype)
    yy,xx = torch.meshgrid(axis,axis,indexing='ij')
    points = torch.stack((xx,yy),-1).reshape(-1,2)
    delta = points[None,None]-center[:,:,None]
    ridge = torch.eye(2,device=center.device,dtype=center.dtype)*1e-8
    distance = torch.einsum('bqni,bqij,bqnj->bqn',delta,torch.linalg.inv(shape+ridge),delta)
    region = (-.5*distance).exp()*foreground[...,None]
    mass = region.sum(1)
    attributes = torch.cat((sizes.clamp_min(1e-8).log(),phase,(need/(1+need))[...,None]),-1)
    pooled = torch.einsum('bqn,bqa->ban',region,attributes)/mass[:,None].clamp_min(1e-8)
    return torch.cat((mass[:,None],pooled),1).reshape(center.shape[0],6,nodes,nodes)


class DemandDensityRouter(nn.Module):
    def __init__(self, feature_channels, classes, nodes, hidden=64, floor=.15):
        super().__init__()
        self.nodes, self.floor = nodes, floor
        self.predictor = nn.Sequential(nn.Conv2d(feature_channels+classes+6,hidden,3,padding=1),
                                       nn.SiLU(),nn.Conv2d(hidden,hidden,3,padding=1),
                                       nn.SiLU(),nn.Conv2d(hidden,1,1))

    def forward(self, features, semantic_response, coarse_state):
        size = (self.nodes,self.nodes)
        visual = F.interpolate(features,size=size,mode='bilinear',align_corners=True)
        semantic = F.interpolate(semantic_response,size=size,mode='bilinear',align_corners=True)
        query = stopped_query_maps(coarse_state,self.nodes).to(visual.dtype)
        logits = self.predictor(torch.cat((visual,semantic.to(visual.dtype),query),1))[:,0].float()
        densities = torch.stack([bounded_density(g,self.floor) for g in logits])
        return densities,[ConditionalCDF(rho) for rho in densities]
