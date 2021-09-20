import argparse
from configs.defaults import get_cfg

def parse_args():
    parser = argparse.ArgumentParser(
        description="Configurations for the membership inference attack."
    )

    parser.add_argument(
        "opts",
        help="See configs/defaults.py for all the options that can be specified.",
        default=None,
        nargs=argparse.REMAINDER,
    )
    
    return parser.parse_args()

def load_config(args):
    cfg = get_cfg()
    
    if args.opts is not None:
         cfg.merge_from_list(args.opts)

    return cfg