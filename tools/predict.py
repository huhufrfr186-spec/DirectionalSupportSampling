import argparse
import torch
from dds.factory import load_checkpoint
from dds.inference import export_predictions
from dds.evaluation import merge_predictions,save_merged


def main():
    p = argparse.ArgumentParser(description='Patch inference and original-image rotated NMS')
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--index',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--workers',type=int,default=0)
    p.add_argument('--threshold',type=float,default=.001)
    p.add_argument('--merged')
    a = p.parse_args()
    torch.set_num_threads(4)
    model,cfg,_ = load_checkpoint(a.checkpoint,a.device)
    export_predictions(model,a.index,a.output,a.device,a.batch_size,a.workers,a.threshold)
    if a.merged:
        scenes = merge_predictions(a.index,a.output,cfg['classes'],**cfg['evaluation']['merge'])
        save_merged(scenes,a.merged)


if __name__=='__main__':
    main()
