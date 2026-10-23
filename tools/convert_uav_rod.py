"""Convert UAV-ROD rotated XML annotations into quadrilateral text annotations."""
import argparse
import math
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET


def convert(xml_path):
    result = []
    for obj in ET.parse(xml_path).getroot().findall('object'):
        box = obj.find('robndbox')
        if box is None:
            raise ValueError(f'{xml_path}: missing robndbox')
        cx,cy,w,h,angle = [float(box.findtext(k)) for k in ['cx','cy','w','h','angle']]
        c,s = math.cos(angle),math.sin(angle)
        points = [(cx+c*u-s*v,cy+s*u+c*v) for u,v in [(-w/2,-h/2),(w/2,-h/2),(w/2,h/2),(-w/2,h/2)]]
        difficult = int(obj.findtext('difficult','0'))
        result.append(' '.join(f'{v:.8f}' for point in points for v in point)+f' car {difficult}')
    return '\n'.join(result)+'\n' if result else ''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--images',required=True)
    p.add_argument('--annotations',required=True)
    p.add_argument('--split-file',required=True,help='One original image identity per line')
    p.add_argument('--output',required=True,help='Destination split directory, e.g. datasets/UAV_ROD/train')
    a = p.parse_args()
    output = Path(a.output)
    (output/'images').mkdir(parents=True,exist_ok=True)
    (output/'labelTxt').mkdir(exist_ok=True)
    identities = [line.split()[0] for line in Path(a.split_file).read_text().splitlines() if line.strip()]
    if len(identities)!=len(set(identities)):
        raise ValueError('Duplicate source image identity')
    for identity in identities:
        name = Path(identity).stem
        candidates = [p for p in Path(a.images).glob(name+'.*') if p.suffix.lower() in ['.jpg','.jpeg','.png','.tif','.tiff','.bmp']]
        if len(candidates)!=1:
            raise ValueError(f'{name}: expected one source image')
        shutil.copy2(candidates[0],output/'images'/candidates[0].name)
        (output/'labelTxt'/(name+'.txt')).write_text(convert(Path(a.annotations)/(name+'.xml')),encoding='utf-8')
    print(f'{len(identities)} original images converted')


if __name__=='__main__':
    main()
