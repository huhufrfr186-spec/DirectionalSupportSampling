"""Construction of the directional support detector and loss."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import yaml
from .networks.presnet import PResNet
from .networks.hybrid_encoder import HybridEncoder
from .networks.decoder import RTDETRTransformer
from .detector import CoupledCanvasDetector


def load_config(path):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
    base = cfg.pop('base', None)
    if base:
        parent = load_config(path.parent/base)
        for key,value in cfg.items():
            if isinstance(value,dict) and isinstance(parent.get(key),dict):
                parent[key].update(value)
            else:
                parent[key] = value
        cfg = parent
    return cfg


def build_model(cfg, text_vectors=None, initialize_backbone=False):
    m = cfg['model']
    classes = cfg['classes']
    if text_vectors is None:
        asset = np.load(cfg['semantic']['text_vectors'],allow_pickle=False)
        if asset['classes'].tolist() != classes:
            raise ValueError('Text class order does not match dataset classes')
        text_vectors = torch.from_numpy(asset['embeddings'].copy()).float()
    if text_vectors.shape[0] != len(classes):
        raise ValueError('Text embedding class count differs from model')
    hidden = m['hidden_dim']
    backbone = PResNet(depth=50,variant='d',return_idx=[1,2,3],freeze_at=0,
                      freeze_norm=True,pretrained=False)
    if initialize_backbone:
        path = m.get('backbone_weights')
        if path:
            saved = torch.load(path,map_location='cpu',weights_only=True)
            backbone.load_state_dict(saved.get('model',saved),strict=True)
        elif m.get('pretrained',True):
            from .networks.presnet import donwload_url
            weights = torch.hub.load_state_dict_from_url(donwload_url[50],map_location='cpu')
            backbone.load_state_dict(weights,strict=True)
    encoder = HybridEncoder(in_channels=[512,1024,2048],hidden_dim=hidden,
                            nhead=m['heads'],dim_feedforward=m['ffn_dim'],
                            num_classes=len(classes),eval_spatial_size=None)
    decoder = RTDETRTransformer(num_classes=len(classes),hidden_dim=hidden,
        feat_channels=[hidden]*3,feat_strides=[8,16,32],num_queries=m['queries'],
        num_decoder_layers=m['layers'],dim_feedforward=m['ffn_dim'],nhead=m['heads'],
        num_denoising=0,eval_spatial_size=None)
    bins = np.sign(np.linspace(-1,1,m['bins'])) * np.abs(np.linspace(-1,1,m['bins']))**2
    model = CoupledCanvasDetector(SimpleNamespace(backbone=backbone,encoder=encoder,decoder=decoder),
        text_vectors,residual_bins=bins.tolist(),levels=3,canvas=m['canvas'],nodes=m['density_nodes'],
        reader_groups=m['reader_groups'],semantic_temperature=cfg['semantic']['temperature'],
        allocation_mode=m['allocation'],density_floor=m['density_floor'],
        demand_source=m['demand_source'],ridge=m['shape_ridge'])
    # Geometry heads provide all decoder box updates.
    model.transformer.native.dec_bbox_head = torch.nn.ModuleList([
        torch.nn.Identity() for _ in range(m['layers'])])
    model.gradient_checkpointing = m.get('gradient_checkpointing',True)
    return model


def build_criterion(cfg, model):
    from .losses import DetectionCriterion
    bins = model.transformer.native.decoder.geometry_heads[0].bins
    return DetectionCriterion(bins,cfg['loss'],cfg['allocation'])


def load_checkpoint(path, device='cpu'):
    saved = torch.load(path,map_location='cpu',weights_only=False)
    cfg = saved['config']
    vectors = saved['model']['semantic.class_text']
    model = build_model(cfg,vectors)
    model.load_state_dict(saved['model'],strict=True)
    return model.to(device),cfg,saved
