# VQ-VAE Nnabla

This is implementation of [VQ-VAE](https://arxiv.org/abs/1711.00937) in Nnabla. 

## Requirements

- [Nnabla](https://nnabla.readthedocs.io/en/latest/python/install_on_linux.html#installation) (along with [cuda](https://nnabla.readthedocs.io/en/latest/python/pip_installation_cuda.html#pip-installation-cuda) and [distributed execution](https://nnabla.readthedocs.io/en/latest/python/pip_installation_cuda.html#pip-installation-distributed))
- [DALI](https://docs.nvidia.com/deeplearning/sdk/dali-developer-guide/docs/quickstart.html) (For Imagenet Training)
- NumPy
- MatPlotLib

## Datasets tested

- MNIST
- Cifar10
- ImageNet

## Instructions

To start training, execute:  

`python main.py --data imagenet`     

 This will start training on Imagenet dataset on 1 gpu (path to the dataset must be specified in configs/imagenet.yaml). For multi-gpu execution:
 `mpirun -n 4 python main.py --data imagenet`    
 will start training on 4 gpus.

To train on cifar10 or mnist, just replace `imagenet` with `cifar10` and `mnist` respectively. For cifar10 and mnist, the dataset will automatically be downloaded if 
it did not exist previously.

The model and solver parameters along with the average epoch loss and reconstructions of training and validation dataset will be 
stored in tmp.monitor folder by default. 

## Configuration file

The config folder has a yaml file for each of the datasets (imagenet.yaml, cifar10.yaml, mnist.yaml) which allows for editing of the following: 
- model configurations (hidden dimension. embedding dimension, number of embedding layers, commitment cost, rng, 
checkpoint to start retraining and saving models directory)
- training configuration (batch size, number of epochs, weight decay, learning rate, learning rate decay, solver type, 
logger interval)
- monitor path for training and validation loss and reconstruction
- dataset related parameters (path, dali threads)
- extension module and device id

## Results

- Mnist
<p float="center">
  <img src="results/mnist.png" />
</p> 

- Cifar10
<p float="center">
  <img src="results/cifar10.png"/>
</p> 

- Imagenet
<p float="center">
  <img src="results/imagenet.png"/>
</p> 
