"""Export all patch predictions in patch pixels and input box coordinates."""
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .data import PatchDataset,collate


@torch.no_grad()
def export_predictions(model,index,output,device,batch_size=1,workers=0,threshold=.001,topk=300):
    previous = model.training
    model.eval()
    dataset = PatchDataset(index)
    loader = DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=workers,collate_fn=collate)
    path = Path(output)
    path.parent.mkdir(parents=True,exist_ok=True)
    partial = path.with_suffix(path.suffix+'.partial')
    with partial.open('w',encoding='utf-8') as stream:
        for images,_,identities in loader:
            prediction = model(images.to(device))
            probabilities = prediction['pred_logits'].sigmoid()
            for batch,identity in enumerate(identities):
                scores,indices = probabilities[batch].flatten().topk(min(topk,probabilities[batch].numel()))
                valid = scores>=threshold
                scores,indices = scores[valid],indices[valid]
                labels = indices % probabilities.shape[-1]
                queries = indices // probabilities.shape[-1]
                boxes = prediction['pred_boxes'][batch,queries].clone()
                # The data frame, not the detail canvas, determines this scale.
                boxes[:,:4]*=images.shape[-1]
                row = dict(patch_id=identity,boxes=boxes.cpu().tolist(),labels=labels.cpu().tolist(),scores=scores.cpu().tolist())
                stream.write(json.dumps(row)+'\n')
    partial.replace(path)
    model.train(previous)
    return path
