import torch

from configs.parser import parse_args, load_config
from configs.defaults import assert_and_infer_cfg
from utils.train_shadow import run_shadow_training
from utils.train_attacker import run_attack_training
from utils.test_attacker import run_attack_test

def main():
    args = parse_args()
    cfg = load_config(args)
    cfg = assert_and_infer_cfg(cfg)

    if cfg.USE_GPU == 1 and torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Using GPU.")
    else:
        device = torch.device("cpu")
        print("GPU not specified or unavailable. Using CPU.")

    if cfg.SHADOW.TRAIN.ENABLE == True:
        run_shadow_training(cfg, device)
    
    if cfg.ATTACKER.TRAIN.ENABLE == True:
        run_attack_training(cfg, device)
    
    run_attack_test(cfg, device)

if __name__ == "__main__":
    main()