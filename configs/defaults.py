import os
import time
from fvcore.common.config import CfgNode

#Config Definition
_C = CfgNode()

#======================================#
#Shadow model options
#======================================#
_C.SHADOW = CfgNode()

#Number of shadow models
_C.SHADOW.NUM_MODELS = 1

#Shadow Model architecture
_C.SHADOW.MODEL = CfgNode()
_C.SHADOW.MODEL.ARCH = "simplecnn" #Name of the shadow model architecture. This will be used when selecting the model function.

#Shadow model optimizer name, learning rate, and weight decay
_C.SHADOW.OPTIMIZER = CfgNode()
_C.SHADOW.OPTIMIZER.NAME = "adam" #Name of the optimizer which will be used when loading the optimizer function
_C.SHADOW.OPTIMIZER.ALPHA = 1e-3 #learning rate of the optimizer
_C.SHADOW.OPTIMIZER.GAMMA = 5e-4 #Weight decay to apply to the optimizer
_C.SHADOW.OPTIMIZER.BETAS = (0.9, 0.999) #Beta values to be used with the adam optimizer. This configuration is not required for sgd
_C.SHADOW.OPTIMIZER.MOMENTUM = 0. #Momentum for SGD. Does not apply to adam.

#Shadow model criterion
_C.SHADOW.CRITERION = "ce"

#Complete path including filename where the output data from the shadow model which will be fed to the attack models are stored
_C.SHADOW.DATA_OUT_PATH = "/export/home/aneezahm001/libraries/papers-to-code/MI/datasets/cifar-10-batches-py/attack_data/attack_data"

#Shadow model training options
_C.SHADOW.TRAIN = CfgNode()
_C.SHADOW.TRAIN.ENABLE = True #Train new shadow models : True, Skip shadow training and run attack on existing data : False
_C.SHADOW.TRAIN.DATASET = "cifar10" #Name of the dataset on which the shadow model is trained. This is used when selecting the dataset for shadow training
_C.SHADOW.TRAIN.DATA_PATH = "/export/home/aneezahm001/libraries/papers-to-code/MI/datasets/cifar-10-batches-py/shadow_data/" #Path to the dataset stated in SHADOW.TRAIN.DATASET
_C.SHADOW.TRAIN.NUM_EPOCHS = 1 #Number of epochs to train each shadow model
_C.SHADOW.TRAIN.BATCH_SIZE = 64 #Batch size of dataloader for shadow model training
_C.SHADOW.TRAIN.CHECKPOINT_DIR = "/export/home/aneezahm001/libraries/papers-to-code/MI/shadow_models/checkpoints/cifar10/" #Directory to store shadow model checkpoints
_C.SHADOW.TRAIN.SET_SIZE = 5000 #Size of train and test sets for training each shadow model. This value should be smaller than the size of the base dataset. The size of the train and test sets for each shadow model is the same.


#======================================#
#Attack model options
#======================================#
_C.ATTACKER = CfgNode()

#The class numbers of the victim model dataset on which the attack model will be trained and tested. Data from classes not specified will not be included in the attacker train and test sets.
_C.ATTACKER.VICTIM_CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

#Attack model options
_C.ATTACKER.MODEL = CfgNode()
_C.ATTACKER.MODEL.ARCH = "fcnet" #Name of the attack model backbone. used when selecting the attack model to load for training
_C.ATTACKER.MODEL.LAYER_DIMS = [64, 32, 16, 8] #For a fully connected model, specify the dimensions of each layer. This also serves to specify the number of layers to use in teh fc model.

_C.ATTACKER.OPTIMIZER = CfgNode()
_C.ATTACKER.OPTIMIZER.NAME = "adam" #name of the optimizer to be used to train the attack model
_C.ATTACKER.OPTIMIZER.ALPHA = 1e-4 #learning rate for the attacker optimizer
_C.ATTACKER.OPTIMIZER.GAMMA = 5e-4 #Weight decay for the attack model optimizer
_C.ATTACKER.OPTIMIZER.BETAS = (0.9, 0.999) #Betas to be used with the adam optimizer. Does not need to be specified if using sgd
_C.ATTACKER.OPTIMIZER.MOMENTUM = 0. #Momentum to apply to the sgd optimizer. does not need to be specified for adam optimizer

#Loss function to train the attack model
_C.ATTACKER.CRITERION = "bce"

#Probability Value above which a record will be determined to be in the trainset of the victim model
_C.ATTACKER.PRED_THRESHOLD = 0.5

#Attack model training options
_C.ATTACKER.TRAIN = CfgNode()
_C.ATTACKER.TRAIN.ENABLE = True #True to train a new set of attack models, False to use an existing checkpoint to directly test the attack accuracy
_C.ATTACKER.TRAIN.DATASET = "cifar10" #Name of the dataset used for training the attack model. 
_C.ATTACKER.TRAIN.DATA_PATH = "" #Path to the attack model training and testing data directory. This will be set according to the log dir
_C.ATTACKER.TRAIN.NUM_EPOCHS = 1 #Number of epochs to train each attack model for
_C.ATTACKER.TRAIN.BATCH_SIZE = 64 #Batch size when training the attack models
_C.ATTACKER.TRAIN.CHECKPOINT_DIR = "" #Path to the directory where the attack model checkpoints will be stored. This will be set according to the log directory

#Settings for testing the attack model on the victim train and test sets
_C.ATTACKER.TEST = CfgNode()
_C.ATTACKER.TEST.ENABLE = True #True to enable testing after trianing is complete, false to disable testing and stop after training
_C.ATTACKER.TEST.DATASET = "cifar10" #Name of the dataset used to train the victim model. This value will be used to conditionally select the dataloaders.
_C.ATTACKER.TEST.DATA_PATH = "./datasets/cifar-10-batches-py/attack_data" #Path to the directory containing the victim models train and test sets
_C.ATTACKER.TEST.LABELS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] #A list containing the labels for which you would like to test the attack. Only classes included here will be tested
_C.ATTACKER.TEST.BATCH_SIZE = 64 #Batch size for the testloders when testing the attack model
_C.ATTACKER.TEST.CHECKPOINT_DIR = None #leave empty if you want to test the attack model with the models trained in the current run. If you want to test the models trained from a previous run, enter the path to the follder containing 
                                        #the attack model folders with the checkpoint for each attack model.

#======================================#
#Victim options
#======================================#
_C.VICTIM = CfgNode()

#Number of classes in the victim model
_C.VICTIM.NUM_CLASSES = 10

#Number of input features in the victim training dataset. For example, 3 for RGB, 1 for greyscale
_C.VICTIM.IN_FEATURES = 3

#======================================#
#Misc. options
#======================================#

#0 for cpu, any value greater than 1 to use GPU.
_C.USE_GPU = 1

#Number of workers to use for the dataloaders
_C.NUM_WORKERS = 2

#Directory to save the attack logs.
_C.LOG_DIR = "./log/"


def assert_and_infer_cfg(cfg):
    """
    Include any checks that the configurations need to pass to be used in the attack.
    """
    log_folder = str(round(time.time())) + "_" + cfg.SHADOW.MODEL.ARCH + "_" + cfg.ATTACKER.MODEL.ARCH
    cfg.LOG_DIR = os.path.join(cfg.LOG_DIR, cfg.SHADOW.TRAIN.DATASET, log_folder)
    cfg.LOG_FILE = os.path.join(cfg.LOG_DIR, "attack.log.tsv")

    cfg.ATTACKER.TRAIN.CHECKPOINT_DIR = os.path.join(cfg.LOG_DIR, "attacker_files")
    cfg.ATTACKER.TRAIN.DATA_PATH = os.path.join(cfg.LOG_DIR, "attacker_files")

    if cfg.ATTACKER.TEST.CHECKPOINT_DIR == None:
        cfg.ATTACKER.TEST.CHECKPOINT_DIR = cfg.ATTACKER.TRAIN.CHECKPOINT_DIR

    cfg.SHADOW.DATA_OUT_PATH = os.path.join(cfg.LOG_DIR, "attacker_files")
    cfg.SHADOW.TRAIN.CHECKPOINT_DIR = os.path.join(cfg.LOG_DIR, "shadow_files")

    if not os.path.exists(cfg.LOG_DIR):
        print(cfg.LOG_DIR, "does not exist. Creating it now...")    
        os.makedirs(cfg.LOG_DIR)
        os.makedirs(cfg.ATTACKER.TRAIN.CHECKPOINT_DIR)
        os.makedirs(cfg.SHADOW.TRAIN.CHECKPOINT_DIR)

    with open(cfg.LOG_FILE, "a") as af:
        write_string = str(cfg)
        af.write(write_string + "\n\n")

    return cfg

def get_cfg():
    """
    Return a copy of the configs.
    """
    return _C.clone()