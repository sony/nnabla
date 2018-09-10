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

#### Prepare datasets

To create cache files requires approximately 400[GBytes] of disk capacity.

1. Prepare the data of ImageNet (You can get ImageNet dataset from the [link](https://imagenet.herokuapp.com/). The following setup procedure requires the following three files.
  - Development kit: `ILSVRC2012_devkit_t12.tar.gz`
  - Training dataset: `ILSVRC2012_img_train.tar`
  - Validation dataset: `ILSVRC2012_img_val.tar`

2. Create the cache files of the datasets that improve the disk I/O overhead.

We provides the tool for creating dataset cache.

    usage: create_cache.py [-h] -D DEVKIT
                           [-W WIDTH] [-H HEIGHT]
                           [-m {trimming,padding}]
                           [-S {True,False}]
                           [-N FILE_CACHE_SIZE]
                           [-C {h5,npy}]
                           [--thinning THINNING]
                           input [input ...] output

    positional arguments:
      input                 Source file or directory.
      output                Destination directory.

    optional arguments:
      -h, --help            show this help message and exit
      -D DEVKIT, --devkit DEVKIT
                            Devkit filename
      -W WIDTH, --width WIDTH
                            width of output image (default:320)
      -H HEIGHT, --height HEIGHT
                            height of output image (default:320)
      -m {trimming,padding}, --mode {trimming,padding}
                            shaping mode (trimming or padding) (default:trimming)
      -S {True,False}, --shuffle {True,False}
                            shuffle mode if not specified, train:True, val:False. Otherwise specified value will be used for both.
      -N FILE_CACHE_SIZE, --file-cache-size FILE_CACHE_SIZE
                            num of data in cache file (default:100)
      -C {h5,npy}, --cache-type {h5,npy}
                            cache format (h5 or npy) (default:npy)
      --thinning THINNING   Thinning rate

This tools creates dataset cache directory from ImageNet data tar
archive.

It auto detect contents of tar archive, so you can change image file
name and add/remove images from train tar archive.  Currently,
converting validation archive includes hard coded information, then
you can rename but cannot change contents order of tar archive.

For example

```
$ python create_cache.py \
    -D ImageNet/ILSVRC2012_devkit_t12.tar.gz \
    ImageNet/ILSVRC2012_img_train.tar \
    ImageNet/ILSVRC2012_img_val.tar \
    ImageNet/imagenet-320-320-trimming-npy
```

It will create cache data from

- ImageNet/ILSVRC2012_devkit_t12.tar.gz
    - Meta data
- ImageNet/ILSVRC2012_img_train.tar
    - Training images
- ImageNet/ILSVRC2012_img_val.tar
    - Validation images

Outputs are followings

- ImageNet/imagenet-train-320-320-trimming-npy/synsets_id_name.csv
    - SYNSET_ID name list.
- ImageNet/imagenet-train-320-320-trimming-npy/synsets_id_word.csv
    - SYNSET_ID word list.
- ImageNet/imagenet-train-320-320-trimming-npy/train
    - Cache for training dataset. Shuffled.
- ImageNet/imagenet-train-320-320-trimming-npy/val
    - Cache for validation dataset. Not shuffled.

#### Training

The following line executes the ImageNet training (See more options in the help by the `-h` option.).

```
4-1．Execute the example of ImageNet about Single GPU.
  - python classification.py -c "device id" -b"batch size" -a"accumulate gradient" -L"number of layers" -T "directory of the trainning cache file" -V "directory of the validation cache file"
    [ex):python classification.py -c cudnn -b64 -a4 -L34 -T train_cache -V val_cache]

4-2．Execute the example of ImageNet about Multi GPU.
  - mpirun -n "Number of GPUs" multi_device_multi_process_classification.py -b"batch size" -a"accumulate gradient" -L"number of layers" -l"learning rate" -i"max iteration of training" -v"validation interval" -j"mini-batch iteration of validation" -s"interval of saving model parameters" -D"interval of learning rate decay" -T "directory of the trainning cache file" -V "directory of the validation cache file"
    [ex):mpirun -n 4 python multi_device_multi_process_classification.py -b 32 -a 2 -L 50 -l 0.1 -i 2000000 -v 20004 -j 1563 -s 20004 -D 600000,1200000,1800000 -T train_cache -V val_cache]
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
