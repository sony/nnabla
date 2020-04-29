# TecoGAN: Temporal Coherent GAN
This repository contains the code (in NNabla) for "[ LEARNING TEMPORAL COHERENCE VIA SELFSUPERVISION
FOR GAN-BASED VIDEO GENERATION](https://arxiv.org/pdf/1811.09393.pdf)"
paper by [Mengyu Chu et al.](https://github.com/thunil/TecoGAN).

## Introduction
This paper focuses on temporal self-supervision for GAN-based video generation tasks and targets Video Super Resolution (VSR) and Unpaired Video Translation (UVT).

### Prerequisites

* tensorflow   
* nnabla 

### Inference

The pre-trained TecoGAN models can be used to generate High-Resolution samples from the given Low-Resolution samples. Author's pre-trained weights converted to NNabla format can be downloaded from the below link:
### Pre-trained Weights :
| TecoGAN |
|---|
|[TecoGAN pre-trained weights](https://nnabla.org/pretrained-models/nnabla-examples/GANs/TecoGAN/model.h5)|

### Inference using the downloaded pre-trained weights.
Clone the nnabla-examples [repository](https://github.com/sony/nnabla-examples.git).
```
cd nnabla-examples/GANs/TecoGAN
```
Run the following command to download test data and ground-truth data
```
python download_test_data.py 
```
Run the following command to generate High-Resolution samples
```
python generate.py --model {path to downloaded TecoGAN NNabla weight file} --input_dir_LR {input directory} --output_dir {path to output directory}
```
### Inference using pre-trained weights provided by original authors
See the following [link](./authors_weights_inference.md) to use the original author's pre-trained weights for inference.