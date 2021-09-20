import random
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import random_split, Subset

from datasets.make_dataset import Cifar10
from datasets import dataset_to_transforms
from shadow_models.shadow_backbones import get_shadow_model
from utils.trainer import train_model

def run_shadow_training(cfg, device):
    if cfg.SHADOW.TRAIN.DATASET == "cifar10":
        transform_func = dataset_to_transforms["cifar"]["test"]
        shadow_dataset = Cifar10(
            root=cfg.SHADOW.TRAIN.DATA_PATH, 
            data_split="train", 
            model_split="shadow", 
            transforms=transform_func
        )
        
    train_test_size = len(shadow_dataset) // 2
    train_test_split = [train_test_size] * 2
    shadow_trainset_all, shadow_testset_all = random_split(shadow_dataset, train_test_split)

    log_flag = False

    for i in range(cfg.SHADOW.NUM_MODELS):
        subset_train_indices = random.sample(range(train_test_size), cfg.SHADOW.TRAIN.SET_SIZE)
        shadow_trainset = Subset(shadow_trainset_all, subset_train_indices)
        subset_test_indices = random.sample(range(train_test_size), cfg.SHADOW.TRAIN.SET_SIZE)
        shadow_testset = Subset(shadow_testset_all, subset_test_indices)
        print("Created train and test dataset for shadow model #" + str(i+1))

        shadow_model_backbone = get_shadow_model(
            cfg.SHADOW.MODEL.ARCH, 
            num_classes=cfg.VICTIM.NUM_CLASSES, 
            input_features=cfg.VICTIM.IN_FEATURES, 
            device=device
        )

        if not log_flag:
            with open(cfg.LOG_FILE, "a") as af:
                to_write = "==========SHADOW MODEL ARCHITECTURE==========\n"
                af.write(str(shadow_model_backbone) + "\n\n")
            log_flag = True
        
        if cfg.SHADOW.OPTIMIZER.NAME == "adam":
            shadow_optimizer = optim.Adam(
                shadow_model_backbone.parameters(), 
                lr=cfg.SHADOW.OPTIMIZER.ALPHA
            )
        elif cfg.SHADOW.OPTIMIZER.NAME == "sgd":
            shadow_optimizer = optim.SGD(
                shadow_model_backbone.parameters(), 
                lr=cfg.SHADOW.OPTIMIZER.ALPHA
            )
        
        if cfg.SHADOW.CRITERION == "ce":
            criterion = nn.CrossEntropyLoss()
        else:
            print("Specified loss function for shadow model", cfg.SHADOW.CRITERION, "is not supported.")

        temp_str = "Training shadow model #" + str(i+1)
        print(temp_str)
        with open(cfg.LOG_FILE, "a") as af:
            af.write(temp_str + "\n")
        
        train_model(
            model=shadow_model_backbone, 
            num_epochs=cfg.SHADOW.TRAIN.NUM_EPOCHS, 
            trainset=shadow_trainset, 
            testset=shadow_testset,
            optimizer=shadow_optimizer, 
            criterion=criterion, 
            device=device, 
            batch_size=cfg.SHADOW.TRAIN.BATCH_SIZE,
            checkpoint_dir=cfg.SHADOW.TRAIN.CHECKPOINT_DIR,
            data_out_dir=cfg.ATTACKER.TRAIN.DATA_PATH,
            num_workers=cfg.NUM_WORKERS,
            log_file=cfg.LOG_FILE,
        )