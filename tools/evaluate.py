import argparse
import json
from pathlib import Path
from dds.factory import load_config
from dds.evaluation import merge_predictions,evaluate_scenes


def main():
    p = argparse.ArgumentParser(description='Evaluate original-image oriented AP')
    p.add_argument('--config',required=True)
    p.add_argument('--index',required=True)
    p.add_argument('--predictions',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--include-difficult',action='store_true')
    p.add_argument('--small',action='store_true')
    p.add_argument('--score-threshold',type=float,default=0.)
    a = p.parse_args()
    cfg = load_config(a.config)
    scenes = merge_predictions(a.index,a.predictions,cfg['classes'],**cfg['evaluation']['merge'])
    result = evaluate_scenes(scenes,cfg['classes'],a.include_difficult,a.small,a.score_threshold)
    output = Path(a.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['AP','AP50','AP75','R75','FPI75']},indent=2))


if __name__=='__main__':
    main()
