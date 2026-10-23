"""Directional support sampling: dual reader."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .geometry import rotation


def positive_shape_root(shape):
    # Closed 2x2 SPD square root avoids eigenvector derivatives at equal axes.
    determinant_root = torch.linalg.det(shape).clamp_min(1e-20).sqrt()
    trace = shape.diagonal(dim1=-2,dim2=-1).sum(-1)
    identity = torch.eye(2,device=shape.device,dtype=shape.dtype)
    return (shape+determinant_root[...,None,None]*identity)/(trace+2*determinant_root).sqrt()[...,None,None]


class DualPeriodicReader(nn.Module):
    def __init__(self, hidden_dim, levels, groups):
        super().__init__()
        self.levels,self.groups = levels,groups
        self.offsets = nn.Linear(hidden_dim,levels*groups*2)
        self.weights = nn.Linear(hidden_dim,levels*groups)
        self.context_projection = nn.Linear(hidden_dim,hidden_dim)
        self.detail_projection = nn.Linear(hidden_dim,hidden_dim)

    def forward(self, queries, state, context_features, detail_features, transports):
        batch,count = queries.shape[:2]
        offsets = self.offsets(queries).reshape(batch,count,self.levels,self.groups,2).tanh()
        weights = self.weights(queries).softmax(-1).reshape(batch,count,self.levels,self.groups)
        shape_root = positive_shape_root(state['shape'].float())
        angles = state['angle'].float()[...,None]+torch.arange(4,device=queries.device)*math.pi/2
        frames = .5*shape_root[:,:,None] @ rotation(angles)
        points = state['center'].float()[:,:,None,None,None]+torch.einsum('bqkij,bqlgj->bqlgki',frames,offsets.float())
        context_sum,detail_sum = torch.zeros_like(queries),torch.zeros_like(queries)
        for image,transport in enumerate(transports):
            for level,(context,detail) in enumerate(zip(context_features,detail_features)):
                original = points[image,:,level]
                valid = ((original>=0)&(original<=1)).all(-1)
                safe = original.clamp(0,1)
                zoom = transport.forward(safe)
                def read(feature,locations):
                    grid = locations.reshape(1,count,self.groups*4,2).to(feature.dtype)*2-1
                    sampled = F.grid_sample(feature[image:image+1],grid,mode='bilinear',padding_mode='zeros',align_corners=True)
                    sampled = sampled[0].permute(1,2,0).reshape(count,self.groups,4,-1)
                    shared = weights[image,:,level,:,None,None]/4
                    return (sampled*valid[...,None].to(sampled.dtype)*shared).sum((1,2))
                context_sum[image] = context_sum[image]+read(context,safe)
                detail_sum[image] = detail_sum[image]+read(detail,zoom)
        return self.context_projection(context_sum)+self.detail_projection(detail_sum)
