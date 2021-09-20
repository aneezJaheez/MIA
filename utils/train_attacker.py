import os
import torch.optim as optim
import torch.nn as nn

from attack_models.attack_backbones import get_attack_model
from datasets.make_dataset import Cifar10Attack
from utils.trainer import train_model

def run_attack_training(cfg, device):
    log_flag = False
    
    for victim_class in cfg.ATTACKER.VICTIM_CLASSES:
        if cfg.ATTACKER.TEST.DATASET == "cifar10":
            attack_trainset = Cifar10Attack(
                root=cfg.ATTACKER.TRAIN.DATA_PATH,
                transforms=None,
                cifar_class_label=victim_class,
                split="train",
            )

            attack_testset = Cifar10Attack(
                root=cfg.ATTACKER.TEST.DATA_PATH,
                transforms=None,
                cifar_class_label=victim_class,
                split="test",
            )
        else:
            print("You have selected a test dataset that is not implemented. Please select a test dataset for the attacker from " + str(["cifar10"]))
        
        model = get_attack_model(
            model=cfg.ATTACKER.MODEL.ARCH, 
            device=device, 
            layer_dims=cfg.ATTACKER.MODEL.LAYER_DIMS,
            in_features=cfg.VICTIM.NUM_CLASSES,
            checkpoint_path=None,
        )

        if not log_flag:
            with open(cfg.LOG_FILE, "a") as af:
                to_write = "==========ATTACK MODEL ARCHITECTURE==========\n"
                af.write(str(model) + "\n\n")
            log_flag = True

        if cfg.ATTACKER.OPTIMIZER.NAME == "adam":
            attacker_optimizer = optim.Adam(model.parameters(), lr=cfg.ATTACKER.OPTIMIZER.ALPHA)

        if cfg.ATTACKER.CRITERION == "bce":
            attacker_criterion = nn.BCELoss()
        elif cfg.ATTACKER.CRITERION == "ce":
            attacker_criterion = nn.CrossEntropyLoss()
        else:
            print("Specified loss criterion for attacker models", cfg.ATTACKER.CRITERION, "is not supported.")
        
        temp_str = "Starting attack sequence for class " + str(victim_class)
        print(temp_str)
        with open(cfg.LOG_FILE, "a") as af:
            af.write(temp_str + "\n")
        
        checkpoint_dir = os.path.join(
            cfg.ATTACKER.TRAIN.CHECKPOINT_DIR, 
            "attack_model_" + str(victim_class)
        )

        train_model(
            model=model,
            num_epochs=cfg.ATTACKER.TRAIN.NUM_EPOCHS,
            trainset=attack_trainset,
            testset=attack_testset,
            optimizer=attacker_optimizer,
            criterion=attacker_criterion,
            device=device,
            batch_size=cfg.ATTACKER.TRAIN.BATCH_SIZE,
            checkpoint_dir=checkpoint_dir,
            data_out_dir=None,
            num_workers=cfg.NUM_WORKERS,
            log_file=cfg.LOG_FILE,
            bce_pred_threshold=cfg.ATTACKER.PRED_THRESHOLD,
        )