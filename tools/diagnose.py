import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from dds.factory import load_checkpoint
from dds.data import PatchDataset,collate
from dds.geometry import metrics


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description='Evaluate directional support on a 17 by 17 object grid')
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--index',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda')
    a = p.parse_args()
    torch.set_num_threads(4)
    model,_,_ = load_checkpoint(a.checkpoint,a.device)
    model.eval()
    loader = DataLoader(PatchDataset(a.index),batch_size=1,collate_fn=collate)
    path = Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8') as stream:
        for images,targets,identities in loader:
            prediction = model(images.to(a.device))
            boxes = targets[0]['boxes'].to(a.device)
            values = metrics(prediction['transports'][0],boxes,canvas=model.canvas,grid=17)
            for index,box in enumerate(boxes):
                row = dict(patch_id=identities[0],object_index=index,box=box.cpu().tolist(),
                           **{k:v[index].item() for k,v in values.items()})
                stream.write(json.dumps(row)+'\n')


if __name__=='__main__':
    main()
