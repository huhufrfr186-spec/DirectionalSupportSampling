"""Frozen CLIP scene, oriented-region, and class text features."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .geometry import rotation


class FrozenTeacher(nn.Module):
    def __init__(self,name,device='cpu'):
        super().__init__()
        from transformers import CLIPModel, CLIPProcessor
        self.model = CLIPModel.from_pretrained(name).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(name)
        self.model.requires_grad_(False)
        self.device = torch.device(device)

    @torch.no_grad()
    def class_vectors(self,classes):
        prompts = [f'an aerial photograph of a {c.replace("-"," ")}' for c in classes]
        tokens = self.processor(text=prompts,padding=True,return_tensors='pt').to(self.device)
        return F.normalize(self.model.get_text_features(**tokens),dim=-1)

    @torch.no_grad()
    def encode_images(self,images,batch=32):
        if not len(images):
            return torch.empty(0,self.model.config.projection_dim,device=self.device)
        result = []
        for start in range(0,len(images),batch):
            values = self.processor(images=[im.detach().float().cpu() for im in images[start:start+batch]],
                                    do_rescale=False,return_tensors='pt')['pixel_values'].to(self.device)
            result.append(F.normalize(self.model.get_image_features(pixel_values=values),dim=-1))
        return torch.cat(result)

    @torch.no_grad()
    def forward(self,images,targets):
        self.model.eval()
        scene = self.encode_images(images)
        regions = []
        axis = torch.linspace(-.5,.5,224,device=images.device)
        yy,xx = torch.meshgrid(axis,axis,indexing='ij')
        offsets = torch.stack((xx,yy),-1)
        for image,target in zip(images,targets):
            boxes = target['boxes']
            crops = []
            # Chunk the crop grid to keep the teacher's memory bounded.
            for chunk in boxes.split(16):
                if not len(chunk):
                    continue
                B = rotation(chunk[:,4]) @ torch.diag_embed(chunk[:,2:4])
                points = chunk[:,None,None,:2]+torch.einsum('nij,hwj->nhwi',B,offsets)
                crops.extend(F.grid_sample(image[None].expand(len(chunk),-1,-1,-1),points*2-1,
                                          align_corners=True,padding_mode='zeros').unbind())
            regions.append(self.encode_images(crops))
        return dict(scene=scene,regions=regions)


def save_text_vectors(name,classes,output,device):
    teacher = FrozenTeacher(name,device)
    vectors = teacher.class_vectors(classes).cpu().numpy()
    output = Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez(output,embeddings=vectors,classes=np.asarray(classes),teacher=np.asarray(name))
