import argparse
from pathlib import Path
from dds.data import prepare_split
from dds.factory import load_config


def main():
    p = argparse.ArgumentParser(description='Prepare overlapping oriented detection patches')
    p.add_argument('--config',required=True)
    p.add_argument('--source',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--splits',nargs='+',default=['train','val','test'])
    p.add_argument('--unlabeled-test',action='store_true')
    a = p.parse_args()
    cfg = load_config(a.config)
    seen = set()
    for split in a.splits:
        names = {f.stem for f in (Path(a.source)/split/'images').iterdir() if f.is_file()}
        if seen & names:
            raise ValueError('Source image identities overlap between splits')
        seen.update(names)
        index,count = prepare_split(a.source,a.output,split,cfg['classes'],
                                   **cfg['data']['patch'],require_labels=not (a.unlabeled_test and split=='test'))
        print(f'{split}: {count} patches -> {index}',flush=True)


if __name__=='__main__':
    main()
