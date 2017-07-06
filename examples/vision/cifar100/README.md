# CIFAR-100 Examples

---

## Overview

The examples listed below demonstrate several deep learning algorithms on CIFAR-100 dataset, which is one of the most popular image classification datasets in the machine learning community. The CIFAR-100 dataset will be automatically downloaded when running any of the examples.

---

## Multi Device Classification task (`multi_device_classification.py`)

This example shows the naive `Data Parallel Distributed Training` for the object regognition task using CIFAR-100 dataset and 23-layers ResNet with [NCCL v1](https://github.com/NVIDIA/nccl). 

NOTE that if you would like to run this example, please follow the bulid instruction to enable the multi-gpu training and make sure to prepare environment where you can use multiple GPUs. 

When you run the script like the following, 

```
python multi_device_classification.py --context "cuda.cudnn" -bs 64 -n 4

```

you can execute the training of 23-layers ResNet in the `Data Parallel Distributed Training` manner with batch size being 64 and 4 GPUs.
