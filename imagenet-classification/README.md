# ImageNet ILSVRC2012 classification


This is an official training and inference example of [ImageNet ILSVRC2012](http://www.image-net.org/) classification with NNabla.
We provide a bunch of different network architectures that can be trained using ILSVRC2012 dataset with our very fast and efficient training code.
Note that the training completely relies on NVIDIA's GPUs, and uses NVIDIA's data processing library called [DALI](https://docs.nvidia.com/deeplearning/sdk/dali-developer-guide/docs/index.html) which runs only on Linux.

The previous implementation (with lower performance both in speed & accuracy) has been moved to [`obsolete/`](./obsolete).

## Setting up

### Clone repository

```shell
git clone git@github.com:sony/nnabla-examples.git
```

### Set up dependencies

We recommend you to follow [our Docker workflow](../doc/docker.md) to set up a training environment.
If you would like to manually install all the requirements, install the following.

* Python >= 3.6
* CUDA (10.0 is recommended)
* nnabla and a CUDA extension (with [multi-GPU](https://nnabla.readthedocs.io/en/latest/python/pip_installation_cuda.html#installation-with-multi-gpu-supported) support is recommended)

## Inference with a pretrained model

You can use a pretrained weights provided by us or obtained after training for inference as following. See help with `python infer.py -h` for more options.

```shell
python infer.py {input image file} {h5 parameter file} -a {network architecture name such as `resnet50` and `se_resnext50}
```

You can find links to pretrained parameter files at a section ["Training results"](#training-results). Most of the pretrained parameter file maintain weight tensors as NHWC memory layout. Please refer to a section ["Memory layout conversion"](#memory-layout-conversion) for how to convert to NCHW format.

## Training

### Additional dependencies for training

* [DALI](https://docs.nvidia.com/deeplearning/sdk/dali-developer-guide/docs/index.html) (works with 0.16 or higher)
* For distributed multi-GPU training:
  * OpenMPI
  * NCCL2

### Preparing ImageNet dataset

#### Download ILSVRC2012

(You can skip this part if you already have ILSVRC2012 dataset on your machine.)

Download archived image files for both training and validation, `ILSVRC2012_img_train.tar` and `ILSVRC2012_img_val.tar`, from [the download page](http://image-net.org/download-images) (registration is required), and extract files as following.

```bash
# cd to dataset root and set path to tar files
TRAIN_TAR=<path to tar>/ILSVRC2012_img_train.tar
VAL_TAR=<path to tar>/ILSVRC2012_img_val.tar

# Create a directory structure
# We will extract images to `train/{WordNet ID; e.g. n01440764}/`
tar tf $TRAIN_TAR | sed 's/.tar//g' | xargs -i mkdir -p train/{}

# Untar training images
tar xvf $TRAIN_TAR --to-command='tar xvf - -C train/${TAR_FILENAME%.*}'

# Untar validation images
mkdir val
tar xvf $VAL_TAR -C val
```

#### Create a image list file

Next, we create two text files which contain a list of image paths associated with their image category IDs. The files are used as inputs of the training script later. Run the following to obtain those.

```shell
python create_input_files.py -T <path/to/ILSVRC2012_img_train>
```

You are going to get the files like the following.

- train_label
```
n02172182/n02172182_4324.JPEG 625
n02172182/n02172182_6601.JPEG 625
n02172182/n02172182_1015.JPEG 625
n02172182/n02172182_3296.JPEG 625
...
```

- val_label
```
ILSVRC2012_val_00000001.JPEG 490
ILSVRC2012_val_00000002.JPEG 361
ILSVRC2012_val_00000003.JPEG 171
...
```

Note that the category IDs are ranging from 0 to 999, and the numbers are sequentially assigned by an alphabetical order of WordNet IDs (e.g., `n02172182`).

## Run training

The following is a command used when we run distributed training with 4 V100 GPUs.

```shell
mpirun -n 4 python train.py \
  -a resnet50 \
  -b 192 \
  -t half --channel-last \
  -C cfg/cos90.yaml
  -T <path/to/ILSVRC2012 training data directory> \
  -V <path/to/ILSVRC2012 validation data directory>
```

Training results including logs and parameters will be produced in `tmp.monitor.{datatime}`. Given the generated logs, you can visualize training curves of training loss and validation error as images by the following commands

```shell
nnabla_cli plot_series Train-loss.series.txt Validation-loss.series.txt -l training -l validation -o rn50-mp-cos90-loss.png -x Epochs -y "Loss"
nnabla_cli plot_series Train-error.series.txt Validation-error.series.txt -l training -l validation -o rn50-mp-cos90-error.png -x Epochs -y "Top-1 error rate"
```

and those look like as following.

| Loss | Error |
|:---:|:---:|
| ![Loss](results/rn50-mp-cos90-loss.png) | ![Error](results/rn50-mp-cos90-error.png) |


**Options**: You can specify some options for training as summarized here. Please run `python train.py -h` to see more options.

* `-a` specifies a network archicture type such as `'resnet50'` and `'se_resnext50'`.
* `-b` specifies the number of batch size. If you see memory allocation error during execution, please adjust this to fit your training into your GPU.
* `-t half` enables mixed precision training, which saves memory and also gives speedup with GPUs with NVIDIA's TensorCore.
* `--channel-last` trains your model with NHWC memory layout. This reduces overheads due to transpose operations for each TensorCore execution. It also utilizes fused batch normalization operation which combines batch normalization, addition, and activation into a single kernel. It gives some advantages for speed and memory cost.
* `-C` can specify a training configuration file. Available configurations are summarized in the table below. 
* If you want to run it on a single GPU, just omit `mpirun -n 4`. You can specify gpu ID by `-d` option when single GPU mode.


**Training configuratios**:
We provide some of preset training configurations as config files summerized below (see the details by opening the files). You can use it by specifying it with `-C` option for `train.py`. You can also create your own configuration to be used.

| Name | File | Note |
|:---:|:---:|:---:|
| Step90 | [cfg/step90.yaml](./cfg/step90.yaml) | Similar setting with ResNet author's experiment. It trains for 90 epochs with step learning rate decay scheduler with warmup, label smoothing, and so on. | 
| Cos90 | [cfg/cos90.yaml](./cfg/cos90.yaml) | It uses cosine annealing rate decay schedular, and some modified hyperparameters such as base learning rate and weight decay from Step90. |
| Cos350 | [cfg/cos350.yaml](./cfg/cos350.yaml) | All is the same as Cos90 except for epochs as 350. |
| Mixup350 | [cfg/mixup350.yaml](./cfg/mixup350.yaml) | It trains for more epochs (350 epochs) based on Cos90, and uses [mixup](https://arxiv.org/abs/1710.09412) regularization technique to prevent overfitting. |
| Mixup250 | [cfg/mixup250.yaml](./cfg/mixup250.yaml) | All is the same as Mixup350 except for epochs as 250.  |
| Mixup90 | [cfg/mixup90.yaml](./cfg/mixup90.yaml) | All is the same as Mixup350 except for epochs as 90. |

Some of these settings are compatible with the settings found in [NVIDIA's repository](https://github.com/NVIDIA/DeepLearningExamples)  for deep learning examples.

### Training results (ResNet Family)

Training results are summarized as follows.

| Arch. | GPUs | MP*1 | Config*2 | Batch size per GPU | Training time (h) | Validation error (%) | Pretrained parameters (Click to download) | Note |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| ResNet18 | 4 x V100 | Yes | Step90 Mixup250 | 256 | 6.67 16.08 | 29.42 **28.25**| Download: [Step90](https://nnabla.org/pretrained-models/nnabla-examples/ilsvrc2012/rn18-nhwc.h5) [**Mixup250**](https://nnabla.org/pretrained-models/nnabla-examples/ilsvrc2012/rn18-nhwc_mixup250.h5) | |
| ResNet34 | 4 x V100 | Yes | Step90 Mixup250 | 256 | 7.63 18.91 | 26.44 **24.73** | Download: [Step90](https://nnabla.org/pretrained-models/nnabla-examples/ilsvrc2012/rn34-nhwc.h5) [**Mixup250**](https://nnabla.org/pretrained-models/nnabla-examples/ilsvrc2012/rn34-nhwc_mixup250.h5) | |
| ResNet50 | 4 x V100 | No | Step90 | 112 | 23.25 | 23.27 | [Download](https://nnabla.org/pretrained-models/nnabla-examples/ilsvrc2012/rn50-nchw.h5) | |
| ResNet50 | 4 x V100 | Yes | Step90 Cos90 Mixup250 | 192 | 7.29 7.44 20.19 | 23.28 22.98 **21.57** | Download: [Step90](https://nnabla.org/pretrained-models/nnabla-examples/ilsvrc2012/rn50-nhwc.h5) [Cos90](https://nnabla.org/pretrained-models/nnabla-examples/ilsvrc2012/rn50-nhwc_cos90.h5) [**Mixup250**](https://nnabla.org/pretrained-models/nnabla-examples/ilsvrc2012/rn50-nhwc_mixup250.h5) | |
| ResNet101 | 4 x V100 | Yes | Step90 | 128 | 10.85 | 21.89 | [Download](https://nnabla.org/pretrained-models/nnabla-examples/ilsvrc2012/rn101-nhwc.h5) | |
| ResNet152 | 4 x V100 | Yes | Cos90 | 96 | 16.01 | 21.06 | [Download](https://nnabla.org/pretrained-models/nnabla-examples/ilsvrc2012/rn152-nhwc_cos90.h5) | |
| ResNeXt50 | 4 x V100 | Yes | Step90 Cos90 Mixup250 | 96 | 11.85 12.06 33.84 | 22.46 22.41 **20.98** | Download: [Step90](https://nnabla.org/pretrained-models/nnabla-examples/ilsvrc2012/resnext50_nhwc.h5) [Cos90](https://nnabla.org/pretrained-models/nnabla-examples/ilsvrc2012/resnext50_nhwc_cos90.h5) [**Mixup250**](https://nnabla.org/pretrained-models/nnabla-examples/ilsvrc2012/resnext50_nhwc_mixup250.h5) | |
| SE-ResNet50 | 4 x V100 | Yes | Step90 Cos90 Mixup250 | 128 | 14.59 15.03 42.22 | 22.77 22.58 **21.19** | Download: [Step90](https://nnabla.org/pretrained-models/nnabla-examples/ilsvrc2012/se_resnet50_nhwc.h5) [Cos90](https://nnabla.org/pretrained-models/nnabla-examples/ilsvrc2012/se_resnet50_nhwc_cos90.h5) [**Mixup250**](https://nnabla.org/pretrained-models/nnabla-examples/ilsvrc2012/se_resnet50_nhwc_mixup250.h5) | |
| SE-ResNeXt50 | 4 x V100 | Yes | Step90 Cos90 Mixup250 | 96 | 19.76 19.689 55.70 | 21.72 21.53 **20.08** | Download: [Step90](https://nnabla.org/pretrained-models/nnabla-examples/ilsvrc2012/se_resnext50_nhwc.h5) [Cos90](https://nnabla.org/pretrained-models/nnabla-examples/ilsvrc2012/se_resnext50_nhwc_cos90.h5) [**Mixup250**](https://nnabla.org/pretrained-models/nnabla-examples/ilsvrc2012/se_resnext50_nhwc_mixup250.h5) | |

* *1 Mixed precision training with NHWC layout  (`-t half --channel-last`).
* *2 We used training configuration summarized above.

You can also find pretrained weights that are provided by some authors and converted to nnabla's weight format for performance evaluatation.

| Arch. | MP | Validation error (%) | Pretrained parameters | Author's page | Note |
|:---:|:---:|:---:|:---:|:---:|:---|
| SE-ResNet50 | No | 22.42 (22.37 *1) | [Download](https://nnabla.org/pretrained-models/nnabla-examples/ilsvrc2012/se_resnet50_by_author.h5) | [GitHub](https://github.com/hujie-frank/SENet) | Use `-n senet_author` in `infer.py` to specify how to normalize an input image. |
| SE-ResNeXt50 | No | 20.98 (20.97 *1) | [Download](https://nnabla.org/pretrained-models/nnabla-examples/ilsvrc2012/se_resnext50_by_author.h5) | [GitHub](https://github.com/hujie-frank/SENet) | Use `-n senet_author` in `infer.py` to specify how to normalize an input image. |

* *1 Numbers reported in [the author's repository](https://github.com/hujie-frank/SENet#trained-models).

### Training results (MobileNet Family)

Training results are summarized as follows.

| Arch. | GPUs | MP*1 | Config*2 | Batch size per GPU | Training time (h) | Validation error (%) | Pretrained parameters (Click to download) | Note |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| MobileNet-V1 | 4 x V100 | No | Cos90 | 256 | 8.63 | 27.77  | Download: [Cos90](https://nnabla.org/pretrained-models/nnabla-examples/ilsvrc2012/mbnv1_nchw_cos90.h5) | |
| MobileNet-V2 | 4 x V100 | No | Cos90 | 256 | 9.66 | 28.06  | Download: [Cos90](https://nnabla.org/pretrained-models/nnabla-examples/ilsvrc2012/mbnv2_nchw_cos90.h5) | |
| MobileNet-V3-Large | 4 x V100 | No | Cos90 | 256 | 10.92 | 26.11 | Download: [Cos90](https://nnabla.org/pretrained-models/nnabla-examples/ilsvrc2012/mbnv3_large_nchw_cos90.h5) | |
| MobileNet-V3-Samll | 4 x V100 | No | Cos90 | 256 | 5.32 | 33.87  | Download: [Cos90](https://nnabla.org/pretrained-models/nnabla-examples/ilsvrc2012/mbnv3_small_nchw_cos90.h5) | |
| MobileNet-V1 | 4 x V100 | Yes | Cos90 | 256 | 6.35  | 27.63  | Download: [Cos90](https://nnabla.org/pretrained-models/nnabla-examples/ilsvrc2012/mbnv1_nhwc_cos90.h5) | |
| MobileNet-V2 | 4 x V100 | Yes | Cos90 Mixup350 | 256 | 7.14  | 28.32  | Download: [Cos90](https://nnabla.org/pretrained-models/nnabla-examples/ilsvrc2012/mbnv2_nhwc_cos90.h5) [Mixup350](https://nnabla.org/pretrained-models/nnabla-examples/ilsvrc2012/mbnv2_nhwc_mixup350.h5) | |
| MobileNet-V3-Large | 4 x V100 | Yes | Cos90 Mixup350 | 256 | 7.63  | 26.09 | Download: [Cos90](https://nnabla.org/pretrained-models/nnabla-examples/ilsvrc2012/mbnv3_large_nhwc_cos90.h5) [Mixup350](https://nnabla.org/pretrained-models/nnabla-examples/ilsvrc2012/mbnv3_large_nhwc_mixup350.h5) | |
| MobileNet-V3-Samll | 4 x V100 | Yes | Cos90 Mixup350 | 256 | 5.49  | 33.49  | Download: [Cos90](https://nnabla.org/pretrained-models/nnabla-examples/ilsvrc2012/mbnv3_small_nhwc_cos90.h5) [Mixup350](https://nnabla.org/pretrained-models/nnabla-examples/ilsvrc2012/mbnv3_small_nhwc_mixup350.h5) | |

* *1 Mixed precision training with NHWC layout  (`-t half --channel-last`).
* *2 We used training configuration summarized above.

### Training results (EfficientNet Family)

Training results are summarized as follows.

| Arch. | GPUs | MP*1 | Config*2 | Batch size per GPU | Training time (h) | Validation error (%) | Pretrained parameters (Click to download) | Note |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| EfficientNet-B0 | 4 x V100 | Yes | Cos350 Mixup350 | 64 | 49.31 | 23.76  | Download: [Cos350](https://nnabla.org/pretrained-models/nnabla-examples/ilsvrc2012/efficientnet_b0_nchw_cos340.h5) [Mixup350](https://nnabla.org/pretrained-models/nnabla-examples/ilsvrc2012/efficientnet_b0_nchw_cos350.h5) | Input spatial size = (224, 224) |
| EfficientNet-B1 | 4 x V100 | Yes | Cos350 Mixup350 | 64 | 78.41 | 22.11  | Download: [Cos350](https://nnabla.org/pretrained-models/nnabla-examples/ilsvrc2012/efficientnet_b1_nchw_cos350.h5) [Mixup350](https://nnabla.org/pretrained-models/nnabla-examples/ilsvrc2012/efficientnet_b1_nchw_cos350.h5) | Input spatial size = (240, 240) |
| EfficientNet-B2 | 4 x V100 | Yes | Cos350 Mixup350 | 64 | 95.18 | 21.27  | Download: [Cos350](https://nnabla.org/pretrained-models/nnabla-examples/ilsvrc2012/efficientnet_b2_nchw_cos350.h5) [Mixup350](https://nnabla.org/pretrained-models/nnabla-examples/ilsvrc2012/efficientnet_b2_nchw_cos340.h5) | Input spatial size = (260, 260) |
| EfficientNet-B3 | 4 x V100 | Yes | Cos350 Mixup350 | 64 | 149.89 | 20.08  | Download: [Cos350](https://nnabla.org/pretrained-models/nnabla-examples/ilsvrc2012/efficientnet_b3_nchw_cos350.h5) [Mixup350](https://nnabla.org/pretrained-models/nnabla-examples/ilsvrc2012/efficientnet_b3_nchw_cos350.h5)| Input spatial size = (300, 300) |

* *1 Mixed precision training with NHWC layout  (`-t half --channel-last`).
* *2 We used training configuration summarized above.

### Evalutation of pretrained model

You can obtain Top-1 validation error rate with standard evaluation setting (256 scaling with center crop 224) of a trained model by using a command like below.

```bash
python validation.py -b <batch size> --arch <architecture name> -V <validation data folder> <weight file> -t half
```

### Memory layout conversion

You may want to change the memory layout of trained parameters from NHWC (trained with `channel_last=True`) to NCHW and vice versa, for fine-tuning on differnt tasks for example.
You may also want to the remove the 4-th channel in the first convolution which was padded to RGB input during training for speed advantage.

The following command converts a parameter file to a desired configuration.

```shell
python convert_parameter_format.py {input h5 file} {output h5 file} -m {layout either nchw or nhwc} -3
```

See options with `python convert_parameter_format.py -h`.
