import sys
sys.path.append("/export/home/aneezahm001/libraries/paper-to-code/MI/")

import os
import argparse
import torchvision
import torch
import torch.optim as optim
import torch.nn as nn

from victims.victim_backbones import get_victim_model
from datasets import dataset_to_transforms
from utils.trainer import train_model
from datasets.make_dataset import Cifar10

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Arguments to train a victim model for MI attack"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="simplecnn",
        help="name of the victim model which will be used to load the model function"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        help="name of the dataset which will be used to select the pytorch dataset function"
    )

    parser.add_argument(
        "--data_prefix",
        type=str,
        default="/export/home/aneezahm001/libraries/papers-to-code/MI/datasets/cifar-10-batches-py/victim_data/",
        help="absolute path to the directory containing the dataset you want to load using the specified dataset function"
    )

    parser.add_argument(
        "--num_classes",
        type=int,
        default=10,
        help="number of classes in the dataset to specify for the classifier"
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        help="optimizer to use to train the victim"
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=50,
        help="number of epochs to train the victim model for."
    )

    parser.add_argument(
        "--criterion",
        type=str,
        default="ce",
        help="loss function to train the victim"
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=1e-4,
        help="Learning rate for the optimizer"
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.,
        help="weight decay for the optimizer. Set to 0 to encourage overfitting."
    )

    parser.add_argument(
        "--gpu",
        type=int,
        default=1,
        help="set to value > 0 to use GPU, set to value <= 0 to use CPU"
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--log_checkpoint_dir",
        type=str,
        default="./checkpoints/simplecnn-cifar10",
        help="./directory to store the log file and checkpoints"
    )

    args = parser.parse_args()
    print("Running victim training with args", str(args))

    if args.gpu > 0 and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        print("GPU not specified or unavailable. Using CPU...")
        device = torch.device("cpu")

    model = get_victim_model(args.model, args.num_classes, device=device, checkpoint_path=None)

    if args.optimizer == "adam":
        optimizer = optim.Adam(lr=args.alpha, wd=args.gamma)
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(lr=args.alpha, wd=args.gamma)

    assert args.criterion in ["ce"], "You have specified a loss function that is not supported."
    if args.criterion == "ce":
        criterion = nn.CrossEntropyLoss()

    if args.dataset == "cifar10":
        transform_func = dataset_to_transforms["cifar"]["test"]
        trainset = Cifar10(
            root=args.data_prefix, 
            transforms=transform_func, 
            model_split="victim", 
            data_split="train"
        )
        
        testset = Cifar10(
            root=args.data_prefix, 
            transforms=transform_func, 
            model_split="victim", 
            data_split="test"
        )

    if not os.path.exists(args.log_checkpoint_dir):
        os.makedirs(args.log_checkpoint_dir)
    log_file = os.path.join(args.log_checkpoint_dir, "train.log.tsv")
    
    train_model(
        model=model, 
        num_epochs=100, 
        trainset=trainset, 
        testset=testset, 
        optimizer=optimizer, 
        criterion=criterion, 
        device=device, 
        batch_size=64, 
        checkpoint_dir=args.log_checkpoint_dir,
        num_workers=args.num_workers,
        log_file=log_file,
        num_epochs=args.num_epochs,
        data_out_dir=None,
    )
