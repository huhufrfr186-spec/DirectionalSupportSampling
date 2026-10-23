import torch


def box_cxcywh_to_xyxy(x):
    center,size = x[...,:2],x[...,2:4]
    return torch.cat((center-size/2,center+size/2),-1)


def box_xyxy_to_cxcywh(x):
    lo,hi = x[...,:2],x[...,2:4]
    return torch.cat(((lo+hi)/2,hi-lo),-1)
