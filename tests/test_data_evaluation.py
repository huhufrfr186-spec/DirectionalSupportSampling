import json
import numpy as np
from PIL import Image
from dds.data import prepare_split,PatchDataset,tile_origins
from dds.evaluation import merge_predictions,evaluate_scenes


def test_original_image_merge_ignores_and_metrics(tmp_path):
    source = tmp_path/'source'/'test'
    (source/'images').mkdir(parents=True)
    (source/'labelTxt').mkdir()
    Image.fromarray(np.zeros((64,96,3),dtype=np.uint8)).save(source/'images'/'one.png')
    (source/'labelTxt'/'one.txt').write_text('20 20 40 20 40 30 20 30 car 0\n60 20 80 20 80 30 60 30 car 1\n')
    index,count = prepare_split(tmp_path/'source',tmp_path/'patches','test',['car'],size=64,stride=32)
    assert count==2 and tile_origins(100,64,32)==[0,32,36]
    records = [json.loads(line) for line in index.read_text().splitlines()]
    prediction_file = tmp_path/'pred.jsonl'
    rows = []
    for r in records:
        x,y = r['origin']
        rows.append(dict(patch_id=r['id'],boxes=[[30-x,25-y,20,10,0],[70-x,25-y,20,10,0]],labels=[0,0],scores=[.9,.8]))
    prediction_file.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    scenes = merge_predictions(index,prediction_file,['car'])
    assert len(scenes['one']['predictions'])==2
    result = evaluate_scenes(scenes,['car'])
    assert result['AP']==100 and result['objects']==1 and result['FPI75']==0
    extended = evaluate_scenes(scenes,['car'],include_difficult=True)
    assert extended['AP']==100 and extended['objects']==2
    dataset = PatchDataset(index)
    image,target,identity = dataset[0]
    assert image.shape==(3,64,64) and target['boxes'].shape==(1,5)


def test_missing_patch_export_is_rejected(tmp_path):
    import pytest
    index = tmp_path/'test.jsonl'
    index.write_text(json.dumps(dict(id='a',original_id='one',original_size=[64,64],annotation=None,origin=[0,0])))
    prediction = tmp_path/'pred.jsonl';prediction.write_text('')
    with pytest.raises(ValueError,match='missing'):
        merge_predictions(index,prediction,['car'])
