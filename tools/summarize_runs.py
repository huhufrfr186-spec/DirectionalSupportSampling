import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser(description='Summarize original-image metrics across random seeds')
    p.add_argument('--metrics',nargs='+',required=True)
    p.add_argument('--output',required=True)
    a = p.parse_args()
    runs = [json.loads(Path(path).read_text(encoding='utf-8')) for path in a.metrics]
    for key in ['images','objects','include_difficult','small','score_threshold']:
        if len({str(r[key]) for r in runs})!=1:
            raise ValueError(f'Runs use different evaluation settings: {key}')
    result = dict(runs=len(runs),metrics={})
    for key in ['AP','AP50','AP75','R75','FPI75']:
        values = np.asarray([r[key] for r in runs])
        result['metrics'][key] = dict(mean=float(values.mean()),
            std=float(values.std(ddof=1)) if len(values)>1 else 0.,values=values.tolist())
    output = Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
