# MIA
A torch-based implementation of the Membership Inference Attack described in the paper [Membership Inference Attacks against Machine Learning Models](https://arxiv.org/pdf/1610.05820.pdf)

## Index
* [Overview](#Overview)
* [Dependencies and Environment](#Dependencies-and-Environment)
* [Datasets and Preparation](#Datasets-and-Preparation)
* [Running the Attack](#Running-the-Attack)
* [Using Custom Datasets and Models](#Using-Custom-Datasets-and-Models)
* [Attack Architecture](#Attack-Architecture)
  * [Victim Model](#Victim-Model)
  * [Shadow Model](#Shadow-Model)
  * [Attack Model](#Attack-Model)
* [Functionality to be added]

## Overview
A membership inference attack involves involves an adversarial ML classifier, which, given a data record and a blackbox access to a victim classifier, tries to determine whether or not the data record was part of the victim's training dataset. The attack succeeds if the adversary correctly determines if said data record is part of the victim model's training dataset. The objective of the attacker is to recognise such differences in the victim model's behaviour and use it to distinguish members from non-members of the victim's training dataset based solely on the victim's posterior probability outputs.

## Dependencies and Environment

The architecture is implemented using PyTorch 1.9.1. A full list of dependencies are listed in the [environment](https://github.com/aneezJaheez/MIA/blob/main/environment.yml) file. Run the command below to install the required packages in a conda environment. 

```
conda env create -f environment.yml
```

## Datasets and Preparation

Currently, the code is only implemented for the CIFAR10 dataset, which I will use to describe the data preparation recipe I followed. For information on how to implement your own datasets, as well as your own attack, victim, and shadow models, check out [this section](#Using-Custom-Datasets-and-Models).

The CIFAR10 dataset consists of 60000 images equally split among 10 classes, with 50000 and 10000 images in the train and test sets respectively. 
<ol>
 <li>The trainset is first split into 2 sets of size 15000 and 35000 reserved for the victim training and shadow model training respectively. The splits are prepared such that they are disjoint and each split has an equal amount of data from each class.</li>
 <li>The 35000 shadow set images are further randomly split into 2 sets of equal sizes, each set here serving as the pool of train images and test images that each shadow model will choose from as its train and test images.</li>
 <li>During the shadow training phase, the shadow models are trained sequentially. Each shadow model picks "cfg.SHADOW.TRAIN.SET_SIZE" (see attack configs) images from the train and test image pool for training and testing the shadow model respectively.</li>
 <li>At the end of training a shadow model, we make a single pass through all the train and test images for that shadow model to get all the posterior probability outputs. The train images are labelled 1 for "in" and the test images are labelled 0 for "out". Hence, each record containing the posterior probability and the membership label acts as the training data for the subsequent attack models. This is done at the end of training of each shadow model. The way this data is stored for the attack model to train on later is described in more detail in the next step.</li>
 <li>When storing the posterior outputs and memebership labels obtained after training each shadow model, we use a dictionary, the keys of which are the class labels (0-9 for CIFAR10), and the value is the list of data records obtained in the previous step whose groundtruth belongs to each respective class label. This dictionary is filled up as we train each shadow model. At the end of shadow training, the dictionary is stored as a binary file using pickle.</li>
 <li>The data obtained in the previous step is used to train each attack model (10 attack models for CIFAR10). For instance, the dictionary values associated with the key "0" are used to train attack model 0, where all the values belonging to this key belong groundtruth class label 0. Thus each attack model is trained on data belonging to a different class.</li>
 <li>Finally, for the test data of the attack models, we use the original 15000 records used to train the victim model in step 1 as the "in" data, and the original CIFAR10 testset as the "out" data. These records are passed into the trained victim model to obtain the posterior probabilities, and is assumed to belong to the class to which it assigned the highest probability. This record is sent to the attack model trained on the same class to determine its membership status.</li>
</ol>

Steps 2-6 are carried out during runtime. The data preparation for step 1 and 7 will have to be carried out manually prior to the attack. An example for the preparation of each of these steps is shown in [./data_prep_examples](https://github.com/aneezJaheez/MIA/tree/main/data_prep_examples).

## Running the Attack

After installing the required dependencies and preparing the datasets for training the shadow models according to the recipe highlighted in the paper or in [Datasets and Preparations](#Datasets-and-Preparation), the attack can be run from the root directory of the repository via the command 

```
python run_attack.py
```

In addition, the attack includes a list of configurations that can be tuned according to the attack requiremenets. A full list of configurations ar can be found in [configs/defaults.py](https://github.com/aneezJaheez/MIA/blob/main/configs/defaults.py). The configurations can either be tuned directly in the file, or stated in the initial command to run the attack. For instance specifying the shadow model architecture, optimizer and its hyperparameters, and running the attack using the GPU can be achieved via the following command:

```
python run_attack.py SHADOW.MODEL.ARCH simplecnn SHADOW.OPTIMIZER.NAME sgd SHADOW.OPTIMIZER.ALPHA 0.001 GPU 1
```

The configurations listed on the command line in the above manner take precendence over the configurations in the main configurations file.

## Using Custom Datasets and Models

Although the attack has been tested only on a CNN trained on CIFAR10, the attack can be extended to work on any classification model and dataset. 

### Implementing new models

Each class of model architectures ([victim](https://github.com/aneezJaheez/MIA/blob/main/victims/victim_backbones.py), [shadow](https://github.com/aneezJaheez/MIA/blob/main/shadow_models/shadow_backbones.py), and [attack](https://github.com/aneezJaheez/MIA/blob/main/attack_models/attack_backbones.py)) are organized in a similar manner, with the file pertaining to each containing the model backbone class definitions, and a get_model() function to retrieve a model by its assigned name. 

For instance, to implement your own shadow model, say, "ExampleModel":

1. Assign a name to your model and add it to the list of available models in the [shadow_backbones.py](https://github.com/aneezJaheez/MIA/blob/main/shadow_models/shadow_backbones.py) file.
```
available_shadow_backbones = ["simplecnn", "examplemodel"]
```

2. Describe the new model architecture as a class that extends PyTorch's module class, and include it in the same file.

```
class ExampleModel(nn.Module):
    def __init__(self, args):
        super(ExampleModel, self).__init__()

    def forward(self, x):
        return x
```

3. After your model has been added to the backbone file, and the name assigned to it has been added to the list, add and if-else block to the corresponding get_model() function in the same file so that it can be found during runtime. 

```
def get_shadow_model(model, num_classes=10, input_features=3, checkpoint_path=None, device=torch.device("cpu")):
    assert model in available_shadow_backbones, "You have specified a shadow model that has not been implemented. Please choose a model from " + str(available_shadow_backbones)

    if model == "simplecnn":
        model = SimpleCNN(num_classes=num_classes, input_features=input_features)
    elif model == "examplemodel":
        model = ExampleModel(args)
    
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        try:
            model.load_state_dict(checkpoint["state_dict"])
        except KeyError:
            model.load_state_dict(checkpoint)
        
    model.to(device)
    return model
```

4. Before running the attack, specify the model to be used in the [configurations](https://github.com/aneezJaheez/MIA/blob/main/configs/defaults.py) file using its assigned name in the list. For instance, to specify our "ExampleModel" as the shadow model for the attack:

```
_C.SHADOW.MODEL.ARCH = "examplemodel" #Name of the shadow model architecture. This will be used when selecting the model function.
```

A similar sequence of steps can be followed to implement a new attack model or victim model. 

### Implementing new datasets

The datasets in this repository are prepared extending the PyTorch dataset class. The attack requires a dataset class that supplies data for the following:
1. The data pool for the shadow models.
2. The classwise data to train and test the attack model.

In this case, for CIFAR10, we have supplied these 3 splits using two dataset modules, one to provide data to the shadow model and victim model and one for the attack model, which can be found in [datasets/make_dataset.py](https://github.com/aneezJaheez/MIA/blob/main/datasets/make_dataset.py).

After preparing the required data for your required datasets according to steps 1 and 7 in the previous [section](#Datasets-and-Preparation), implement the dataset classes that can be used to access each of the splits mentioned above. In the implemented case for CIFAR10, we have:

1. A "Cifar10" class that provides the data pool for shadow model training and testing.

Note : We have a single pool of data for the shadow model, which is split into disjoint halves during runtime to serve as the train and test pools. Hence, the entire shadow model data pool is loaded in at once without regard to the train and test splits.

2. A "Cifar10Attack" class that provides the train and test data for the attack model. The attack model dataset requires selective access to the records for each class. Hence, this dataset contains an argument to specify the specific class for which the data is required, which is specified during training of each attack model.

```
class Cifar10Attack(Dataset):
    def __init__(self, root, cifar_class_label, split, transforms):
        super(Cifar10Attack, self).__init__()
```

To summarize, you will need a dataset class for the shadow model data that loads in all the shadow model data, and a dataset class for the attack model which is capable of loading in data for each individual class for training and testing separately to train and test each individual attack model assined to that respective class.

After these dataset classes have been implemented, include it in and add a conditional block to the [train_shadow.py](https://github.com/aneezJaheez/MIA/blob/main/utils/train_shadow.py) and [train_attacker.py](https://github.com/aneezJaheez/MIA/blob/main/utils/train_attacker.py) files so that it can be located during runtime. 

In the train_shadow.py file:

```
from datasets.make_dataset import yourshadowdataset

    if cfg.SHADOW.TRAIN.DATASET == "cifar10":
        transform_func = dataset_to_transforms["cifar"]["test"]
        shadow_dataset = Cifar10(
            root=cfg.SHADOW.TRAIN.DATA_PATH, 
            data_split="train", 
            model_split="shadow", 
            transforms=transform_func
        )
    elif cfg.SHADOW.TRAIN.DATASET == "yourshadowdataset":
        .
        .
        .
```

and in the train_attacker.py file:

```
from datasets.make_dataset import yourattackdataset

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
        elif cfg.ATTACKER.TEST.DATASET == "yourattackdataset":
            .
            .
            .
```

In additiion, you can also specify your transform functions in the [datasets/__init__.py](https://github.com/aneezJaheez/MIA/blob/main/datasets/__init__.py) file.

## Attack Architecture
### Victim Model

Currently, this implementation only supports classification models.

As stated in the paper, we assume blackbox access to a trained victim classifier. The victim model takes in an input record x, and computes the posterior probability vector y. The victim model is trained on a classification dataset which we assume no knowledge of. 

A few victim models can be found in [victims/victim_backbones.py](https://github.com/aneezJaheez/MIA/blob/main/victims/victim_backbones.py).

### Shadow Model

An instance of the attack may consist of multiple shadow models, the goal of which is to mimic the target model as closely as possible. The desired performance is best achieved when the shadow models are reasonably similar or even greater in complexity in comparison to the victim models. However, such functionality extraction is a difficult problem in and of itself in the case of more complex deep networks, a scenario that is not discussed in the paper under which the attack does not turn out to be feasible in its current state. 

The difference between the shadow model and the victim model is that we know the training dataset of the shadow model and its ground-truth labels. Given that the shadow models learn to mimic the victim model in their functionality, we can use the inputs and outputs of the shadow model to teach the main attack models how to distinguish between members and non-members of the victims training dataset. The number of shadow models is a hyperparameter that can be tuned. According to the findings in the paper, the attack accuracy increases with an increase in the number of shadow models. 

We query each shadow model with its own training dataset and a disjoint testset.  The training data of the shadow models are labelled "in", and the test data are labelled "out". These resulting "in" and "out" records alongside the posterior probabilities correspond to the training data for the final attack models. The training set and test set of each shadow model is of the same size but is disjoint.

The shadow model used in this instance can be found in [shadow_models/shadow_backbones.py](https://github.com/aneezJaheez/MIA/blob/main/shadow_models/shadow_backbones.py).

### Attack Model

The attack model is the final component of the architecture, task of which is to distinguish the target victim model's behaviour on the training inputs from its behaviour on the inputs that it did not encounter during training. As such, the attack model is trained as a binary classifier, the training data for this model is the labelled inputs and outputs of the shadow models.

There are multiple attack models, just like there are multiple shadow models. The number of attack models is equal to the number of output classes in the victim's dataset. Hence, the number of attack models is predetermined depending the the number of classes the victim classifier is trained on. For each label y in the victim dataset, we train a separate attack model that, given y, predicts the in or out membership status for x. In doing so the each attack model learns the output distribution produced by the the victim models (by actually learning it through the reasonably similar performing shadow models) for each specific class label.

Here, we stick to a fully connected model with varying number of hidden layers and layer dimensions as the attack model, which can be found in [attack_models/attack_backbones.py](https://github.com/aneezJaheez/MIA/blob/main/attack_models/attack_backbones.py).



