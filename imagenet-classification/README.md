# ImageNet classification example

---

## Overview

The examples are written in python. These examples demonstrate learning on Tiny ImageNet dataset.
Tiny ImageNet dataset will be cached by running the example script.

---

## classification.py

In this example, "Residual Neural Network" (also called "ResNet") is trained on a [Tiny Imagenet](https://tiny-imagenet.herokuapp.com/) dataset.

The following line executes the Tiny ImageNet training (with the setting the we recommended you to try first. It requires near 6GB memory available in the CUDA device. See more options in the help by the `-h` option.).

```
python classification.py -c cuda.cudnn -a4 -b64 -L34
```

Tiny ImageNet consists of 200 categories and each category has 500 of 64x64 size images in training set.
The ResNet trained here is almost equivalent to the one used in ImageNet.
The differences are the strides in both the first conv and the max pooling are removed.

After the learning completes successfully, the results will be saved in "tmp.montors.imagenet".

In this folder you will find model files "\*.h5" and result files "\*.txt"
