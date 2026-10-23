"""Directional support sampling: semantic alignment."""
import torch
from torch.nn import functional as F
from .geometry import rotation


def semantic_alignment(student, targets, teacher, roi_nodes=5):
    scene_target = F.normalize(teacher['scene'].detach().to(student['scene']),dim=-1)
    scene_loss = (1-(student['scene']*scene_target).sum(-1)).mean()
    region_losses = []
    local = student['local']
    axis = torch.linspace(-.5,.5,roi_nodes,device=local.device,dtype=local.dtype)
    yy,xx = torch.meshgrid(axis,axis,indexing='ij')
    offsets = torch.stack((xx,yy),-1).reshape(-1,2)
    for image,target in enumerate(targets):
        boxes = target['boxes'].to(local)
        cached = teacher['regions'][image].detach().to(local)
        if len(cached) != len(boxes):
            raise ValueError('Teacher regions must follow current original GT identity/order')
        if not len(boxes):
            continue
        basis = rotation(boxes[:,4]) @ torch.diag_embed(boxes[:,2:4])
        points = boxes[:,None,:2]+torch.einsum('nij,kj->nki',basis,offsets)
        valid = ((points>=0)&(points<=1)).all(-1)
        if not valid.any(-1).all():
            raise ValueError('GT region has no visible semantic sample')
        samples = F.grid_sample(local[image:image+1],points[None]*2-1,
                                mode='bilinear',padding_mode='zeros',align_corners=True)[0].permute(1,2,0)
        pooled = (samples*valid[...,None]).sum(1)/valid.sum(1)[:,None]
        pooled = F.normalize(pooled,dim=-1)
        region_losses.append(1-(pooled*F.normalize(cached,dim=-1)).sum(-1))
    roi_loss = torch.cat(region_losses).mean() if region_losses else local.sum()*0
    return dict(scene=scene_loss,regions=roi_loss)
