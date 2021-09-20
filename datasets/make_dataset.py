import os
import random
import pickle
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from PIL import Image

class Cifar10Attack(Dataset):
    def __init__(self, root, cifar_class_label, split, transforms):
        super(Cifar10Attack, self).__init__()

        self.data = []
        self.membership_labels = []
        self.transforms = transforms

        assert split in ["train", "test"], "Specified dataset split value " + split + " but expected values in " + str(["train", "test"])
        assert cifar_class_label <= 10 and cifar_class_label >= 0, "Received class label " + str(cifar_class_label) + " but this is invalid."

        if split == "train":
            data_filepath = os.path.join(root, "attack_data_train")
        elif split == "test":
            data_filepath = os.path.join(root, "attack_data_test")

        with open(data_filepath, "rb") as infile:
            alldata = pickle.load(infile, encoding="latin1")

        attack_data = alldata[cifar_class_label]
        self.data = attack_data["data"]
        self.labels = attack_data["labels"]

    def __getitem__(self, idx):
        data, target = self.data[idx], self.labels[idx]

        if self.transforms is not None:
            data = self.transforms(data)

        return data, target

    def __len__(self):
        return len(self.labels)

        
class Cifar10(Dataset):
    def __init__(self, root, model_split, data_split, transforms):
        super(Cifar10, self).__init__()
        
        self.data = []
        self.labels = []
        self.root = root
        self.transforms = transforms

        assert model_split in ["victim", "shadow"], "Received model split value " + model_split + " but expected value in " + str(["victim", "shadow"])
        assert data_split in ["train", "test"], "Received data split value " + data_split + " but expected value in " + str(["train", "test"])

        if data_split == "train":
            if model_split == "victim":
                data_path = os.path.join(self.root, "victim_dataset")
                with open(data_path, "rb") as infile:
                    data_loaded = pickle.load(infile, encoding="latin1")
                    self.data = data_loaded["data"]
                    self.labels = data_loaded["labels"]
            elif model_split == "shadow":
                data_path = os.path.join(self.root, "shadow_dataset")
                with open(data_path, "rb") as infile:
                    data_loaded = pickle.load(infile, encoding="latin1")
                    self.data = data_loaded["data"]
                    self.labels = data_loaded["labels"]
        elif data_split == "test":
            data_path = os.path.join(self.root, "test_batch")
            with open(data_path, "rb") as infile:
                data_loaded = pickle.load(infile, encoding="latin1")
                self.data = data_loaded["data"]
                self.labels = data_loaded["labels"]

        self.data = np.array(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))
        self.labels = np.array(self.labels)
        assert self.data.shape[0] == self.labels.shape[0]

    def __getitem__(self, idx):
        img, target = self.data[idx], self.labels[idx]

        img = Image.fromarray(img)

        if self.transforms is not None:
            img =  self.transforms(img)

        return img, target

    def __len__(self):
        return self.data.shape[0]