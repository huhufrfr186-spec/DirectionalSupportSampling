import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from dds.factory import load_checkpoint


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description='Measure synchronized full detector latency')
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--size',type=int,default=1024)
    p.add_argument('--warmup',type=int,default=10)
    p.add_argument('--iterations',type=int,default=30)
    p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--device',default='cuda')
    a = p.parse_args()
    if a.iterations<=0 or a.warmup<0:
        raise ValueError('Invalid timing iteration count')
    torch.set_num_threads(4)
    model,_,_ = load_checkpoint(a.checkpoint,a.device)
    model.eval()
    image = torch.rand(a.batch_size,3,a.size,a.size,device=a.device)
    cuda = torch.device(a.device).type=='cuda'
    def synchronize():
        if cuda:torch.cuda.synchronize(a.device)
    for _ in range(a.warmup):model(image)
    synchronize()
    if cuda:torch.cuda.reset_peak_memory_stats(a.device)
    durations = []
    for _ in range(a.iterations):
        synchronize();start = time.perf_counter()
        model(image)
        synchronize();durations.append(time.perf_counter()-start)
    mean = float(np.mean(durations))
    result = dict(mean_latency_ms=1000*mean,median_latency_ms=1000*float(np.median(durations)),
        images_per_second=a.batch_size/mean,batch_size=a.batch_size,input_size=a.size,
        peak_memory_bytes=torch.cuda.max_memory_allocated(a.device) if cuda else None,
        warmup=a.warmup,iterations=a.iterations,device=str(a.device))
    path = Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
