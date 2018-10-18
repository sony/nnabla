# Training Example using CIFAR-10 and CIFAR-100

---

## Overview

The examples listed below demonstrate several deep learning algorithms on
CIFAR-10 dataset and CIFAR-100 dataset, which are one of the most popular image
classification datasets in the machine learning community.
The datasets will be automatically downloaded when running any of the examples.

---


## Classification task (`classification.py`)

This example demonstrates the training of image classification on
CIFAR-10 dataset(CIFAR-100 dataset). The convolutional neural network takes
32x32 pixel color images as input and outputs the predictions of 10-way
classification(100-way classification).  By default, CIFAR-10 dataset is selected.

When you run the example by

```
python classification.py

```

you will see the training progress in the console (decreasing training and
validation error).

By default, the script will be executed with GPU.
If you prefer to run with CPU, try

```
python classification.py -c cpu
```

After the learning completes successfully, the results will be saved in
"tmp.monitor/". In this folder you will find model files "\*.h5" and result
files "\*.txt".

The classification example provides two choices of neural network architectures
to train, CIFAR10 dataset with 23-layers ResNet and CIFAR100 with 23-layers ResNet.
You can select it with the `-n` option. For more details see the source code and
the help produced by running with the `-h` option.
