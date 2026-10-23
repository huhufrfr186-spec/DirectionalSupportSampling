"""Original-image rotated NMS and continuous precision-envelope AP."""
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
from shapely.geometry import Polygon
from .data import read_annotations


def box_polygon(box):
    x,y,w,h,a = box
    c,s = math.cos(a),math.sin(a)
    points = np.array([[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5]])*[w,h]
    return Polygon(points @ np.array([[c,s],[-s,c]])+[x,y])


def overlap(a,b):
    ax,ay,ar,ab = a.bounds
    bx,by,br,bb = b.bounds
    if ar<=bx or br<=ax or ab<=by or bb<=ay:
        return 0.
    intersection = a.intersection(b).area
    return intersection/max(a.area+b.area-intersection,1e-30)


def rotated_nms(predictions,threshold=.5):
    kept = []
    for item in sorted(predictions,key=lambda p:-p['score']):
        if not any(overlap(item['polygon'],other['polygon'])>threshold for other in kept):
            kept.append(item)
    return kept


def merge_predictions(index,prediction_file,classes,nms=.5,max_detections=3000):
    rows = [json.loads(line) for line in Path(index).read_text(encoding='utf-8').splitlines() if line.strip()]
    records = {r['id']:r for r in rows}
    if len(records)!=len(rows):
        raise ValueError('Duplicate patch identity')
    scenes = {}
    for r in rows:
        scenes.setdefault(r['original_id'],dict(size=r['original_size'],annotation=r['annotation'],predictions=[]))
    seen = set()
    for line in Path(prediction_file).read_text(encoding='utf-8').splitlines():
        p = json.loads(line)
        if p['patch_id'] in seen or p['patch_id'] not in records:
            raise ValueError('Duplicate or unknown patch prediction')
        seen.add(p['patch_id'])
        r = records[p['patch_id']]
        if not len(p['boxes'])==len(p['labels'])==len(p['scores']):
            raise ValueError('Prediction fields have inconsistent lengths')
        for box,label,score in zip(p['boxes'],p['labels'],p['scores']):
            b = np.asarray(box,dtype=float)
            if b.shape!=(5,) or not np.isfinite(b).all() or not np.isfinite(score) or (b[2:4]<=0).any():
                raise ValueError('Invalid oriented prediction')
            if label not in range(len(classes)):
                raise ValueError('Prediction class outside class list')
            b[:2]+=r['origin']
            scenes[r['original_id']]['predictions'].append(dict(box=b.tolist(),polygon=box_polygon(b),label=label,score=score))
    if seen!=set(records):
        raise ValueError(f'{len(set(records)-seen)} patch predictions missing; export empty patches too')
    for scene in scenes.values():
        scene['predictions'] = sorted([p for label in range(len(classes)) for p in
            rotated_nms([p for p in scene['predictions'] if p['label']==label],nms)],
            key=lambda p:-p['score'])[:max_detections]
    return scenes


def continuous_ap(tp,fp,total):
    if not total:
        return None
    tp,fp = np.cumsum(tp),np.cumsum(fp)
    recall = np.r_[0,tp/total,1]
    precision = np.r_[0,tp/np.maximum(tp+fp,1),0]
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    change = np.flatnonzero(recall[1:]!=recall[:-1])
    return float(((recall[change+1]-recall[change])*precision[change+1]).sum())


def evaluate_scenes(scenes,classes,include_difficult=False,small=False,score_threshold=0.):
    truth,regions = {},{}
    for identity,scene in scenes.items():
        if not scene['annotation']:
            raise ValueError('Original annotations are required for evaluation')
        objects = read_annotations(scene['annotation'],classes)
        truth[identity],regions[identity] = [],[]
        scale = 1024/max(scene['size'])
        for item in objects:
            polygon = Polygon(item['polygon'])
            if item['ignored']:
                regions[identity].append(polygon)
                continue
            ignored = (item['difficult'] and not include_difficult) or (small and polygon.area*scale**2>=32**2)
            truth[identity].append(dict(polygon=polygon,label=item['label'],ignore=ignored))
    thresholds = np.linspace(.5,.95,10)
    class_ap = {}
    totals75 = [0,0,0]
    for label,name in enumerate(classes):
        total = sum(o['label']==label and not o['ignore'] for items in truth.values() for o in items)
        predictions = sorted([(identity,p) for identity,s in scenes.items() for p in s['predictions']
                              if p['label']==label and p['score']>=score_threshold],key=lambda v:-v[1]['score'])
        values = []
        for threshold in thresholds:
            matched = set()
            tp,fp = [],[]
            for identity,p in predictions:
                candidates = [(i,o,overlap(p['polygon'],o['polygon'])) for i,o in enumerate(truth[identity]) if o['label']==label]
                # Prefer regular objects to ignored objects when either can match.
                regular = [(i,o,v) for i,o,v in candidates if not o['ignore'] and v>=threshold]
                if regular:
                    best = max(regular,key=lambda v:v[2])[0]
                    key = identity,best
                    success = key not in matched
                    matched.add(key)
                    tp.append(int(success));fp.append(int(not success))
                elif any(o['ignore'] and v>=threshold for _,o,v in candidates):
                    continue
                elif any(p['polygon'].intersection(r).area/max(p['polygon'].area,1e-30)>.5 for r in regions[identity]):
                    continue
                else:
                    tp.append(0);fp.append(1)
            values.append(continuous_ap(tp,fp,total))
            if abs(threshold-.75)<1e-8:
                totals75 = [totals75[0]+sum(tp),totals75[1]+sum(fp),totals75[2]+total]
        class_ap[name] = values
    array = np.asarray([[np.nan if v is None else v for v in row] for row in class_ap.values()])
    if np.isnan(array).all():
        raise ValueError('No evaluable ground truth in this group')
    means = np.nanmean(array,axis=0)
    tp,fp,total = totals75
    return dict(AP=float(np.nanmean(means)*100),AP50=float(means[0]*100),AP75=float(means[5]*100),
        R75=100*tp/max(total,1),FPI75=fp/max(len(scenes),1),objects=total,images=len(scenes),
        score_threshold=score_threshold,include_difficult=include_difficult,small=small,
        thresholds=thresholds.tolist(),AP_by_threshold=(means*100).tolist(),
        per_class={k:[None if v is None else 100*v for v in values] for k,values in class_ap.items()})


def save_merged(scenes,path):
    serial = {identity:dict(size=s['size'],predictions=[{k:v for k,v in p.items() if k!='polygon'}
              for p in s['predictions']]) for identity,s in scenes.items()}
    Path(path).write_text(json.dumps(serial,ensure_ascii=False,indent=2),encoding='utf-8')
