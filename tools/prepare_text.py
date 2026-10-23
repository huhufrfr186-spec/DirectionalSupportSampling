import argparse
from dds.factory import load_config
from dds.teacher import save_text_vectors


def main():
    p = argparse.ArgumentParser(description='Encode frozen foreground class text vectors')
    p.add_argument('--config',required=True)
    p.add_argument('--device',default='cuda')
    a = p.parse_args()
    cfg = load_config(a.config)
    save_text_vectors(cfg['semantic']['teacher'],cfg['classes'],cfg['semantic']['text_vectors'],a.device)
    print(cfg['semantic']['text_vectors'])


if __name__=='__main__':
    main()
