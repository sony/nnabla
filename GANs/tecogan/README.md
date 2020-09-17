# TecoGAN: TEmporal COherent GAN for video super resolution.
This repository contains the code (in NNabla) for "[ LEARNING TEMPORAL COHERENCE VIA SELF-SUPERVISION
FOR GAN-BASED VIDEO GENERATION](https://arxiv.org/pdf/1811.09393.pdf)"
paper by [Mengyu Chu et al.](https://github.com/thunil/TecoGAN)

## Introduction
This paper focuses on temporal self-supervision for GAN-based video generation tasks and targets Video Super Resolution (VSR) and Unpaired Video Translation (UVT). 
The VSR part has been showcased in the current implementation. Major contributions made in this paper are:
* Temporal adversarial training for RNN part (recurrent neural network) to achieve temporal coherency without compromising spatial details.
* Use of spatio-temporal discriminator for realistic and coherent video generation.
* Use of ping pong loss for long term temporal consistency.

__Result Examples__

| Input                  | Output                  |
| :--------------------: | :---------------------: |
| ![](results/cropped_lr_city.gif) | ![](results/cropped_sr_city.gif) |
| ![](results/cropped_lr_bridge.gif) | ![](results/cropped_sr_bridge.gif) |
| ![](results/cropped_lr_robo.gif) | ![](results/cropped_sr_robo.gif) |
| ![](results/cropped_lr_face.gif) | ![](results/cropped_sr_face.gif) |

__Evaluation Results__

Evaluation has been done on [*VID4 dataset*](https://ge.in.tum.de/download/data/TecoGAN/vid4_HR.zip), which contains HR frames for 4 different categories.
Various evaluation metrics used in this paper are:<br>
PSNR: Pixel-wise accuracy<br>
LPIPS: [Perceptual distance to the ground truth (AlexNet)](https://github.com/sony/nnabla-examples/tree/master/utils/neu/metrics/lpips)<br>
tOF: Pixel-wise distance of estimated motions<br>
tLP: Perceptual distance between consecutive frames<br>
SSIM: Structural similarity<br>
&#8595; : Stands for, the smaller, the better<br>
&#8593; : Stands for, the bigger, the better

| | PSNR &#8593;| LPIPS &#8595;| tOF &#8595;| tLP &#8595;| SSIM &#8593;|
|---|---|---|---|---|---|
| Reported in the Tecogan paper | 25.570 | 0.162 | 0.189 | 0.668 | NA |
| Tecogan author's pretrained weights | 25.587 | 0.162 | 0.189 | 0.673 |0.787 |
| NNabla* | 25.667 | 0.152 | 0.199 | 0.394 |0.782 |

*The TecoGAN model takes around 10 days for training on ` Nvidia GeForce RTX 2080 Ti` GPU.

## Prerequisites
* tensorflow >= 1.8.0
* nnabla >= 1.9.0
## Inference
The pre-trained TecoGAN weights can be used to generate High-Resolution frames from the given Low-Resolution frames. The pre-trained weights can be downloaded from the links provided in the below table:

### Pre-trained Weights
| TecoGAN weights | FRVSR weights |
|---|---|
|[TecoGAN pre-trained weights](https://nnabla.org/pretrained-models/nnabla-examples/GANs/tecogan/tecogan_model.h5)|[FRVSR pre-trained weights](https://nnabla.org/pretrained-models/nnabla-examples/GANs/tecogan/frvsr_model.h5)|

### Inference using the downloaded pre-trained weights.
Clone the nnabla-examples [repository](https://github.com/sony/nnabla-examples.git) and run the following command to download test data and ground-truth data
```
cd nnabla-examples/GANs/TecoGAN
python download_test_data.py 
```
Run the following command to generate HR images from a given sample of LR images
```
python generate.py --model {path to downloaded TecoGAN NNabla weight file} --input_dir_LR {input directory} --output_dir {path to output directory}
```
## Dataset preparation
We would like to attribute credits of data download and sequence preparation to original authors of the paper "[ LEARNING TEMPORAL COHERENCE VIA SELF-SUPERVISION FOR GAN-BASED VIDEO GENERATION](https://arxiv.org/pdf/1811.09393.pdf)" and code (https://github.com/thunil/TecoGAN).
Training dataset can be downloaded with the following commands into a chosen directory `TrainingDataPath`.  
Note: online video downloading requires youtube-dl package.  

```bash
# Install youtube-dl for online video downloading
pip install --user --upgrade youtube-dl

# take a look of the parameters first:
python authors_scripts/dataPrepare.py --help

# To be on the safe side, if you just want to see what will happen, the following line won't download anything,
# and will only save information into log file.
# TrainingDataPath is still important, it is the directory where logs are saved: TrainingDataPath/log/logfile_mmddHHMM.txt
python authors_scripts/dataPrepare.py --start_id 2000 --duration 120 --disk_path TrainingDataPath --TEST

# This will create 308 subfolders under TrainingDataPath, each with 120 frames, from 28 online videos.
# It takes a long time. Please note that some videos might become unavailable in future.
python authors_scripts/dataPrepare.py --start_id 2000 --duration 120 --REMOVE --disk_path TrainingDataPath

```
Once dataset is ready, please update the parameter TrainingDataPath in config.yaml or as argument for --input_video_dir flag to run train.py, and then you can start training with the downloaded data! 

PS: Authors claim that most of the data (272 out of 308 sequences) downloaded by the script are the same as the ones they used for the published models. As of June 30, 2020,  303 out of those 308 sequences are available. Training is conducted on first 250 sequences for the results shown here (same # of sequences used by authors for training). 
## Training
TecoGAN training is divided in two steps:
1. A FRVSR (frame recurrent video super resolution) model having 10 residual blocks is trained with L2 loss on flow estimator and generator network.
2. Later the TecoGAN network with 16 residual blocks and discriminator is finetuned using the pre-trained FRVSR model. 
### Training a FRVSR model 
All the experiments are done using a scaling factor of 4 between LR and HR images.
Use the below code to start the training.
#### Single GPU training
```
python train.py \
     --input_video_dir {path to training images} \
     --output_dir {path to save trained model} \
     --num_resblock 10 \
     --max_iter 500000 \
     --tecogan False \
```
#### Distributed Training
For distributed training [install NNabla package compatible with Multi-GPU execution](https://nnabla.readthedocs.io/en/latest/python/pip_installation_cuda.html#pip-installation-distributed). Use the below code to start the distributed training.
```
export CUDA_VISIBLE_DEVICES=0,1,2,3 {device ids that you want to use}
mpirun -n {no. of devices} python train.py \
     --input_video_dir {path to training images} \
     --output_dir {path to save trained model} \
     --num_resblock 10 \
     --max_iter 500000 \
     --tecogan False \
```
### Training TecoGAN
TecoGAN uses feature maps from a pre-trained VGG19 network to encourage generator to produce similar feature maps as ground truth.
So, we require a pre-trained VGG19 model for this purpose. Download the VGG19 NNabla weights 
from [here](https://nnabla.org/pretrained-models/nnabla-examples/tecogan/vgg19.h5). If you want to convert tensorflow VGG19 weights to NNabla h5
format, download the VGG19 weights from [here](http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz) and then convert these weights to .h5 format using the below command:
```
python convert_vgg19_nnabla_tf --pre_trained_model {pre-trained tensorflow vgg19 weights} --save-path {path to save the converted model}
```

The pre-trained FRVSR model is used for finetuning the TecoGAN network. 
To obtain this you can train a FRVSR network or use our [pre-trained FRVSR weights.](https://nnabla.org/pretrained-models/nnabla-examples/GANs/tecogan/frvsr_model.h5)
Use the code below to train TecoGAN:
#### Single GPU training
```
python train.py \
     --input_video_dir {path to training images} \
     --output_dir {path to save trained model} \
     --num_resblock 16 \
     --max_iter 900000 \
     --vgg_pre_trained_weights {path to VGG19 trained model} \
     --pre_trained_frvsr_weights {path to FRVSR trained model}
     --tecogan True \
```
#### Distributed training
```
export CUDA_VISIBLE_DEVICES=0,1,2,3 {device ids that you want to use}
mpirun -n {no. of devices} python train.py \
     --input_video_dir {path to training images} \
     --output_dir {path to save trained model} \
     --num_resblock 16 \
     --max_iter 900000 \
     --vgg_pre_trained_weights {path to VGG19 trained model} \
     --pre_trained_frvsr_weights {path to FRVSR trained model}
     --tecogan True \
```

#### Evaluation
Multiple metrics have been reported in the paper. Code for all of them is provided on authors' repository [here](https://github.com/thunil/TecoGAN).