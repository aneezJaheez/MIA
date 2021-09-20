import os
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime

from attack_models.attack_backbones import get_attack_model
from datasets.make_dataset import Cifar10Attack
from utils.trainer import test_step

def run_attack_test(cfg, device):
    log_flag = False

    for victim_class in cfg.ATTACKER.VICTIM_CLASSES:
        if cfg.ATTACKER.TEST.DATASET == "cifar10":
            testset = Cifar10Attack(
                root=cfg.ATTACKER.TEST.DATA_PATH,
                split="test",
                transforms=None,
                cifar_class_label=victim_class,
            )
        
        testloader = DataLoader(
            testset, 
            batch_size=cfg.ATTACKER.TEST.BATCH_SIZE, 
            shuffle=True,
            num_workers=cfg.NUM_WORKERS,
        )

        attack_checkpoint_path = os.path.join(
            cfg.ATTACKER.TEST.CHECKPOINT_DIR, 
            "attack_model_" + str(victim_class),
            "checkpoint.pth"
        )
        if not os.path.exists(attack_checkpoint_path):
            print("The checkpoint for this class does not exist in directory " + attack_checkpoint_path + ". Skipping this class...")
            continue

        model = get_attack_model(
            model=cfg.ATTACKER.MODEL.ARCH,
            device=device,
            layer_dims=cfg.ATTACKER.MODEL.LAYER_DIMS,
            in_features=cfg.VICTIM.NUM_CLASSES,
            checkpoint_path=attack_checkpoint_path,
        )

        if not log_flag:
            with open(cfg.LOG_FILE, "a") as af:
                to_write = "==========ATTACK MODEL ARCHITECTURE==========\n"
                af.write(str(model) + "\n\n")
                columns = ['victim_class', 'split', 'loss', 'accuracy']
                af.write('\t'.join(columns) + '\n')

            log_flag = True


        if cfg.ATTACKER.CRITERION == "bce":
            attacker_criterion = nn.BCELoss()
        elif cfg.ATTACKER.CRITERION == "ce":
            attacker_criterion = nn.CrossEntropyLoss()
        else:
            print("Specified loss criterion for attacker models", cfg.ATTACKER.CRITERION, "is not supported.")

        print("Testing attack on class " + str(victim_class))
        test_loss, test_acc = test_step(
            model=model,
            testloader=testloader,
            criterion=attacker_criterion,
            epoch=1,
            device=device,
            pred_threshold=cfg.ATTACKER.PRED_THRESHOLD,
        )

        with open(cfg.LOG_FILE, 'a') as af:
            test_cols = [victim_class, 'test', test_loss, test_acc]
            af.write('\t'.join([str(c) for c in test_cols]) + '\n')


