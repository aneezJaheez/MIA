# MIA
A torch-based implementation of the Membership Inference Attack described in the paper [Membership Inference Attacks against Machine Learning Models](https://arxiv.org/pdf/1610.05820.pdf)

## Index
* [Overview](#Overview)
* [Dependencies and Environment](#Dependencies and Environment)
* [Datasets and Preparation]
* [Attack Architecture]
  * [Victim Model]
  * [Shadow Model]
  * [Attack Model]
  * [Error Metrics]
* [Running the Attack]
* [Using Custom Datasets and Models]
* [Functionality to be added]

## Overview
A membership inference attack involves involves an adversarial ML classifier, which, given a data record and a blackbox access to a victim classifier, tries to determine whether or not the data record was part of the victim's training dataset. The attack succeeds if the adversary correctly determines if said data record is part of the victim model's training dataset. The objective of the attacker is to recognise such differences in the victim model's behaviour and use it to distinguish members from non-members of the victim's training dataset based solely on the victim's posterior probability outputs.

## Dependencies and Environment

The architecture is implemented using PyTorch 1.9.1. A full list of dependencies are listed in the [environment](https://github.com/aneezJaheez/MIA/blob/main/environment.yml) file. Run the command below to install the required packages in a conda environment. 

```
conda env create -f environment.yml
```

