# Neural Network Libraries - Examples

This repository contains working examples of [Neural Network Libraries](https://github.com/sony/nnabla/).
Before running any of the examples in this repository, you must install the Python package for Neural Network Libraries. The Python install guide can be found [here](https://nnabla.readthedocs.io/en/latest/python/installation.html).

Before running an example, also run the following command inside the example directory, to install additional dependencies:

```
cd example_directory
pip install -r requirements.txt
```


## Docker image

If you have [Docker](https://www.docker.com/) and [NVIDIA Docker](https://www.docker.com/) on your system, you can build and run a CUDA-compatible Docker image which contains dependencies for running most of the examples in this repository.

The following command creates the Docker image.

```shell
docker build -t local/nnabla-examples . --build-arg CUDA_VER=9.2 --build-arg CUDNN_VER=7 --build-arg PYTHON_VER=3.6
```

The options followed by `--build-arg` specify the versions of some dependent software. You can ommit these arguments if you use the default. You can find the default versions at the lines containing `ARG` commands in the [Dockerfile](./Dockerfile).


A Docker container of the created image can be launched by;

```shell
nvidia-docker run {options} local/nnabla-examples {command}
```

*Note*:

* You must use a CUDA toolkit version compatible with your CUDA driver version on your host system.
