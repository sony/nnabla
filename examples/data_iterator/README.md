# Data iterator example.

## Overview

This is demonstration of DataIterator in NNabla. We tested on Ubuntu 20.04 LTS and Python 3.8.10.

## dataiterator.py

### Usage

This example can test many functionality of data_iterator.

```
$ python examples/data_iterator/dataiteretor.py --help
2017-06-23 13:25:08,205 [nnabla][INFO]: Initializing CPU extension...
usage: dataiteretor.py [-h] [-m] [-f] [-S] [-b BATCH_SIZE] [-s CACHE_SIZE]
                       [-B BUFFER_SIZE] [-M MEMORY_SIZE] [-o OUTPUT] [-n]
                       [-e MAX_EPOCH] [-w WAIT] [-i IN_PLACE_CACHE]
                       uri

Data iterator sample.

positional arguments:
  uri                   PATH to CSV_DATASET format file, "MNIST_TRAIN",
                        "MNIST_TEST", "TINY_IMAGENET_TRAIN" or
                        "TINY_IMAGENET_VAL"

optional arguments:
  -h, --help            show this help message and exit
  -m, --memory_cache
  -f, --file_cache
  -S, --shuffle
  -b BATCH_SIZE, --batch_size BATCH_SIZE
  -s CACHE_SIZE, --cache_size CACHE_SIZE
  -B BUFFER_SIZE, --buffer_size BUFFER_SIZE
  -M MEMORY_SIZE, --memory_size MEMORY_SIZE
  -o OUTPUT, --output OUTPUT
  -n, --normalize
  -e MAX_EPOCH, --max_epoch MAX_EPOCH
  -w WAIT, --wait WAIT
  -i IN_PLACE_CACHE, --in_place_cache IN_PLACE_CACHE
```

You can use your own data with CSV_DATASET format, MNIST or TINY
IMAGENET download on-demand from internet.

You need to copy `mnist_data.py` and `tiny_imagenet_data.py` files from the [nnabla-examples](https://github.com/sony/nnabla-examples) repository to the same grade directory as the `dataiterator.py` file. 
Download path:

`nnabla-examples/image-classification/mnist-collection/mnist_data.py` 

`nnabla-examples/image-classification/imagenet/obsolete/tiny_imagenet_data.py`

Detailed specification of CSV_DATASET format will be coming soon.
Before document comes, please see python/src/nnabla/utils/data-iterator.py


### Example

Here is demonstration with MNIST dataset.

```
$ python examples/data_iterator/dataiterator.py MNIST_TRAIN
2017-06-23 13:23:24,711 [nnabla][INFO]: Initializing CPU extension...
2017-06-23 13:23:24,904 [nnabla][INFO]: DataSource with shuffle(False)
2017-06-23 13:23:24,904 [nnabla][INFO]: Getting label data from http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz.
train-labels-idx1-ubyte.gz: 100%|██████████████████████████████████████| 28881/28881 [00:00<00:00, 1810941.59it/s]
2017-06-23 13:23:26,146 [nnabla][INFO]: Getting label data done.
2017-06-23 13:23:26,146 [nnabla][INFO]: Getting image data from http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz.
train-images-idx3-ubyte.gz: 100%|██████████████████████████████████| 9912422/9912422 [00:02<00:00, 4028632.33it/s]
2017-06-23 13:23:29,168 [nnabla][INFO]: Getting image data done.
2017-06-23 13:23:29,169 [nnabla][INFO]: Using DataIterator
2017-06-23 13:23:29,170 [nnabla][INFO]: 60000
2017-06-23 13:23:29,170 [nnabla][INFO]: Epoch 0
100%|██████████████████████████████████████████████████████████████████▉| 59968/60000 [00:00<00:00, 465012.79it/s]
2017-06-23 13:23:29,299 [nnabla][INFO]: Epoch 1
100%|██████████████████████████████████████████████████████████████████▉| 59968/60000 [00:00<00:00, 476473.23it/s]
2017-06-23 13:23:29,426 [nnabla][INFO]: Epoch 2
60032it [00:00, 473461.41it/s]
```
