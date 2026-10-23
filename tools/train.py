import argparse
import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from dds.factory import load_config,build_model,build_criterion
from dds.data import PatchDataset,collate
from dds.teacher import FrozenTeacher
from dds.inference import export_predictions
from dds.evaluation import merge_predictions,evaluate_scenes


def atomic_save(value,path):
    partial = path.with_suffix('.partial.pth')
    torch.save(value,partial)
    partial.replace(path)


def seed_worker(worker_id):
    seed = torch.initial_seed()%2**32
    random.seed(seed)
    np.random.seed(seed)


def train(cfg,output,device,resume=None):
    t = cfg['train']
    seed = t['seed']
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(t.get('threads',4))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    saved = torch.load(resume,map_location='cpu',weights_only=False) if resume else None
    if saved and saved['config']!=cfg:
        raise ValueError('Resume configuration must match the checkpoint configuration')
    vectors = saved['model']['semantic.class_text'] if saved else None
    model = build_model(cfg,vectors,initialize_backbone=not bool(saved)).to(device)
    criterion = build_criterion(cfg,model).to(device)
    dataset = PatchDataset(cfg['data']['train_index'],augment=True)
    microbatch,effective = t['batch_size'],t['effective_batch_size']
    if effective % microbatch:
        raise ValueError('effective_batch_size must be divisible by batch_size')
    accumulation = effective//microbatch
    backbone,remaining = [],[]
    for name,param in model.named_parameters():
        if param.requires_grad:
            (backbone if name.startswith('backbone.') else remaining).append(param)
    optimizer = torch.optim.AdamW([dict(params=backbone,lr=t['backbone_lr']),
                                  dict(params=remaining,lr=t['lr'])],weight_decay=t['weight_decay'])
    base_lrs = [t['backbone_lr'],t['lr']]
    amp = t['amp'] and str(device).startswith('cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    teacher = FrozenTeacher(cfg['semantic']['teacher'],device) if cfg['loss']['semantic'] else None
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    if not saved and (output/'last.pth').exists():
        raise ValueError('Output already has a checkpoint; use --resume or a new output directory')
    best,start,updates = -1.,0,0
    if saved:
        model.load_state_dict(saved['model'],strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        scaler.load_state_dict(saved['scaler'])
        best,start,updates = saved['best_ap'],saved['epoch']+1,saved['updates']
        random.setstate(saved['rng']['python']);np.random.set_state(saved['rng']['numpy'])
        torch.set_rng_state(saved['rng']['torch'])
        if torch.cuda.is_available() and saved['rng']['cuda'] is not None:
            torch.cuda.set_rng_state_all(saved['rng']['cuda'])
        del saved
    (output/'config.yaml').write_text(__import__('yaml').safe_dump(cfg,sort_keys=False,allow_unicode=True),encoding='utf-8')
    for epoch in range(start,t['epochs']):
        generator = torch.Generator().manual_seed(seed+epoch)
        loader = DataLoader(dataset,batch_size=microbatch,shuffle=True,num_workers=t['workers'],
            collate_fn=collate,worker_init_fn=seed_worker,generator=generator,pin_memory=str(device).startswith('cuda'))
        model.train();optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        running = 0.
        for step,(images,targets,_) in enumerate(loader):
            images = images.to(device)
            targets = [{k:v.to(device) for k,v in target.items()} for target in targets]
            group_start = (step//accumulation)*accumulation
            group_size = min(accumulation,len(loader)-group_start)
            group_samples = min(effective,len(dataset)-group_start*microbatch)
            progress = epoch+step/len(loader)
            if progress<t['warmup_epochs']:
                factor = max(t['warmup_start'],progress/t['warmup_epochs'])
            else:
                fraction = (progress-t['warmup_epochs'])/max(t['epochs']-t['warmup_epochs'],1)
                factor = t['minimum_lr_ratio']+(1-t['minimum_lr_ratio'])*.5*(1+math.cos(math.pi*fraction))
            for group,base in zip(optimizer.param_groups,base_lrs):
                group['lr'] = base*factor
            teacher_features = teacher(images,targets) if teacher else None
            offload = t.get('activation_offload',False) and str(device).startswith('cuda')
            with torch.autograd.graph.save_on_cpu(pin_memory=True) if offload else nullcontext():
                with torch.autocast(device_type=torch.device(device).type,enabled=amp):
                    prediction = model(images,targets)
                with torch.autocast(device_type=torch.device(device).type,enabled=False):
                    losses = criterion(prediction,targets,teacher_features)
                    loss = sum(losses.values())
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss at epoch {epoch+1}, step {step+1}')
            scaler.scale(loss*len(images)/group_samples).backward()
            if step+1==group_start+group_size:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),t['clip_norm'],error_if_nonfinite=True)
                scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
                updates+=1
            running+=loss.detach().item()
            if (step+1)%t['log_interval']==0 or step+1==len(loader):
                print(json.dumps(dict(epoch=epoch+1,step=step+1,steps=len(loader),loss=running/(step+1),
                                      lr=optimizer.param_groups[1]['lr'],updates=updates)),flush=True)
        result = dict(epoch=epoch+1,loss=running/len(loader),seconds=time.perf_counter()-started)
        is_best = False
        validation = cfg['data'].get('val_index')
        if validation and (epoch+1)%cfg['evaluation']['interval']==0:
            predictions = export_predictions(model,validation,output/'validation.jsonl',device,
                threshold=cfg['evaluation']['score_threshold'])
            scenes = merge_predictions(validation,predictions,cfg['classes'],**cfg['evaluation']['merge'])
            metrics = evaluate_scenes(scenes,cfg['classes'])
            result.update({k:metrics[k] for k in ['AP','AP50','AP75']})
            is_best = metrics['AP']>best
            best = max(best,metrics['AP'])
        checkpoint = dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
            config=cfg,epoch=epoch,best_ap=best,updates=updates,
            rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                     cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None))
        atomic_save(checkpoint,output/'last.pth')
        if is_best:
            atomic_save(checkpoint,output/'best.pth')
        if (epoch+1)%t['save_interval']==0:
            atomic_save(checkpoint,output/f'epoch_{epoch+1:02d}.pth')
        with (output/'history.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(result)+'\n')
        print(json.dumps(result),flush=True)


def main():
    p = argparse.ArgumentParser(description='Train directional support sampling')
    p.add_argument('--config',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--resume')
    p.add_argument('--seed',type=int)
    a = p.parse_args()
    cfg = load_config(a.config)
    if a.seed is not None:
        cfg['train']['seed'] = a.seed
    train(cfg,a.output,a.device,a.resume)


if __name__=='__main__':
    main()
