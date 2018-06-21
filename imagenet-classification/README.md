# ImageNet classification example

---

## Overview

The examples are written in python. These examples demonstrate learning on ImageNet and Tiny ImageNet dataset.
In case of ImageNet, We need to get the dataset and to create the cache file by yourself.
Tiny ImageNet dataset will be cached by running the example script.

---

## classification.py

In this example, "Residual Neural Network" (also called "ResNet") is trained on a [Imagenet](https://imagenet.herokuapp.com/) and [Tiny Imagenet](https://tiny-imagenet.herokuapp.com/) dataset.

### ImageNet

ImageNet consists of 1000 categories and each category has 1280 of images in training set.
The ImageNet dataset(training and validation) requires 150[GBytes] of disk capacity.
To create catche files requires approximately 400[GBytes] of disk capacity.

1. Prepare the data of ImageNet (You can get ImageNet dataset from the [link](https://imagenet.herokuapp.com/). The following setup procedure requires the following two files.
  - Training dataset: ILSVRC2012_img_train.tar
  - Validation dataset: ILSVRC2012_img_val.tar

2. Create a directory for the data set.
  - For the trainning data.
      - mkdir "directory name"
          - [ex):mkdir train_data]
      - python create_train_dir.py -t "tar file(trainning) of ImageNet" -o "directory name"
          - [ex):python create_train_dir.py -t ILSVRC2012_img_train.tar -o train_data]
  - For the validation data.
      - mkdir "directory name"
          - [ex):mkdir val_data]
      - python create_val_dir.py -t "tar file(validation) of ImageNet" -o "directory name"
          - [ex):python create_val_dir.py -t ILSVRC2012_img_val.tar -o val_data]

3. Create the cache files of the datasets that improve the disk I/O overhead.
  - For the trainning data.
      - mkdir "directory name"
          - [ex):mkdir train_cache]
      - python create_cache_file.py -i "directory of the trainning data" -o "directory of the trainning cache file" -w "width of output image" -g "height of output image" -m "shaping mode (trimming or padding)" -s "shuffle mode (true or false)"
          - [ex):[python create_cache_file.py -i train_data -o train_cache -w 320 -g 320 -m trimming -s true]
  - For the validation data.
      - mkdir "directory name"
          - [ex):mkdir val_cache]
      - python create_cache_file.py -i "directory of the validation data" -o "directory of the validatio cache file" -w "width of output image" -g "height of output image" -m "shaping mode (trimming or padding)" -s "shuffle mode (true or false)"
          - [ex):python create_cache_file.py -i val_data -o val_cache -w 320 -g 320 -m trimming -s false]

The following line executes the ImageNet training (See more options in the help by the `-h` option.).

```
4-1．Execute the example of ImageNet about Single GPU.
  - python classification.py -c "device id" -b"batch size" -a"accumulate gradient" -L"number of layers" -T "directory of the trainning cache file" -V "directory of the validation cache file"
    [ex):python classification.py -c cudnn -b64 -a4 -L34 -T train_cache -V val_cache]

4-2．Execute the example of ImageNet about Multi GPU.
  - mpirun -n "Number of GPUs" multi_device_multi_process_classification.py -b"batch size" -a"accumulate gradient" -L"number of layers" -l"learning rate" -i"max iteration of training" -v"validation interval" -j"mini-batch iteration of validation" -s"interval of saving model parameters" -D"interval of learning rate decay" -T "directory of the trainning cache file" -V "directory of the validation cache file"
    [ex):mpirun -n 4 python multi_device_multi_process_classification.py -b 32 -a 2 -L 50 -l 0.1 -i 2000000 -v 20004 -j 1563 -s 20004 -D 600000 -D 1200000 -D 1800000 -T train_cache -V val_cache]
```

After the learning completes successfully, the results will be saved in "tmp.montors.imagenet".

In this folder you will find model files "\*.h5" and result files "\*.txt"

### Tiny Imagenet

The training script for ImageNet also works on Tiny ImageNet dataset, which
consists of 200 categories and each category has 500 of 64x64 size images in training set.
The ResNet trained here is almost equivalent to the one used in ImageNet.
The differences are the strides in both the first conv and the max pooling are removed.

The following line executes the Tiny ImageNet training (with the setting the we recommended you to try first. It requires near 6GB memory available in the CUDA device. See more options in the help by the `-h` option.).

```
python classification.py -c cudnn -a4 -b64 -L34 -M true
```
