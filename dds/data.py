"""DOTA quadrilateral annotations and original-image patch manifests."""
import json
import math
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from shapely.geometry import Polygon, box as polygon_box
import torch
from torch.utils.data import Dataset


def read_annotations(path,classes):
    result = []
    for number,line in enumerate(Path(path).read_text(encoding='utf-8-sig').splitlines(),1):
        f = line.split()
        if not f or f[0].startswith(('imagesource:','gsd:','#')):
            continue
        if len(f)<9:
            raise ValueError(f'{path}:{number}: expected 8 coordinates and a class name')
        points = np.asarray([float(v) for v in f[:8]],dtype=np.float64).reshape(4,2)
        polygon = Polygon(points)
        if not polygon.is_valid or polygon.area<=0:
            raise ValueError(f'{path}:{number}: invalid quadrilateral')
        ignored = f[8].lower() in ('ignore','ignored','###')
        if not ignored and f[8] not in classes:
            raise ValueError(f'{path}:{number}: unknown class {f[8]}')
        result.append(dict(polygon=points.tolist(),label=-1 if ignored else classes.index(f[8]),
                           difficult=bool(len(f)>9 and int(f[9])!=0),ignored=ignored))
    return result


def polygon_to_box(points):
    points = np.asarray(points,dtype=np.float32).reshape(-1,2)
    (x,y),(w,h),degrees = cv2.minAreaRect(points)
    return [x,y,max(w,1e-5),max(h,1e-5),math.radians(degrees)]


def tile_origins(length,size,stride):
    if length<=size:
        return [0]
    values = list(range(0,length-size+1,stride))
    if values[-1]!=length-size:
        values.append(length-size)
    return values


def prepare_split(source,output,split,classes,size=1024,stride=512,visible=.7,require_labels=True):
    source,output = Path(source).resolve(),Path(output).resolve()
    if not 0<stride<=size or not 0<visible<=1:
        raise ValueError('Invalid patch preparation settings')
    images = sorted(p for p in (source/split/'images').iterdir()
                    if p.suffix.lower() in ('.jpg','.jpeg','.png','.tif','.tiff','.bmp'))
    if not images:
        raise ValueError(f'No images under {source/split/"images"}')
    if len({p.stem for p in images})!=len(images):
        raise ValueError('Image stems must be unique within each split')
    target = output/split
    (target/'images').mkdir(parents=True,exist_ok=True)
    manifest = []
    annotation_dir = source/split/'labelTxt'
    if not annotation_dir.exists():
        annotation_dir = source/split/'annfile'
    for image_path in images:
        ann_path = annotation_dir/(image_path.stem+'.txt')
        if require_labels and not ann_path.is_file():
            raise FileNotFoundError(ann_path)
        objects = read_annotations(ann_path,classes) if ann_path.is_file() else []
        with Image.open(image_path) as im:
            image = im.convert('RGB')
        width,height = image.size
        for y in tile_origins(height,size,stride):
            for x in tile_origins(width,size,stride):
                patch_id = f'{image_path.stem}__{x}___{y}'
                patch = image.crop((x,y,x+size,y+size))
                path = target/'images'/(patch_id+'.png')
                patch.save(path)
                bounds = polygon_box(x,y,min(x+size,width),min(y+size,height))
                kept,ignored = [],[]
                for obj in objects:
                    points = np.asarray(obj['polygon'])
                    polygon = Polygon(points)
                    fraction = polygon.intersection(bounds).area/polygon.area
                    if fraction<=0:
                        continue
                    item = dict(obj,polygon=(points-[x,y]).tolist())
                    if fraction>=visible and not obj['difficult'] and not obj['ignored']:
                        kept.append(item)
                    else:
                        item['ignored'] = True
                        ignored.append(item)
                manifest.append(dict(id=patch_id,image=str(path.relative_to(output)),
                    original_id=image_path.stem,original_image=str(image_path),
                    annotation=str(ann_path) if ann_path.is_file() else None,
                    original_size=[width,height],origin=[x,y],size=size,
                    objects=kept,ignored=ignored))
    index = output/(split+'.jsonl')
    index.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in manifest),encoding='utf-8')
    return index,len(manifest)


class PatchDataset(Dataset):
    def __init__(self,index,augment=False):
        self.index = Path(index).resolve()
        self.records = [json.loads(line) for line in self.index.read_text(encoding='utf-8').splitlines() if line.strip()]
        if not self.records:
            raise ValueError('Empty patch manifest')
        self.augment = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self,index):
        row = self.records[index]
        with Image.open(self.index.parent/row['image']) as source:
            image = np.asarray(source.convert('RGB'),dtype=np.float32)/255
        if image.shape[:2]!=(row['size'],row['size']):
            raise ValueError('Patch dimensions differ from manifest')
        boxes = np.asarray([polygon_to_box(o['polygon']) for o in row['objects']],dtype=np.float32).reshape(-1,5)
        ignored = np.asarray([polygon_to_box(o['polygon']) for o in row['ignored']],dtype=np.float32).reshape(-1,5)
        for values in [boxes,ignored]:
            values[:,:4]/=row['size']
        if self.augment and torch.rand(()).item()<.5:
            image = image[:,::-1].copy()
            for values in [boxes,ignored]:
                values[:,0] = 1-values[:,0]
                values[:,4] = math.pi-values[:,4]
        target = dict(boxes=torch.from_numpy(boxes),
            labels=torch.tensor([o['label'] for o in row['objects']],dtype=torch.long),
            ignored_boxes=torch.from_numpy(ignored))
        return torch.from_numpy(image.copy()).permute(2,0,1),target,row['id']


def collate(batch):
    images,targets,identities = zip(*batch)
    return torch.stack(images),list(targets),list(identities)
