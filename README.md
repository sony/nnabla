# Gallery of Neural Network Libraries

This repository contains working examples of [Neural Network Liraries](https://github.com/sony/nnabla/).
Before running any of examples in this repository, you must install NNabla Python package.

```
pip install -U nnabla
```

Some examples may be highly computationally expensive. We recommend you to have a CUDA-capable GPU installed on your machine, and install NNabla's CUDA extension ([installation guide](https://nnabla.readthedocs.io/en/latest/python/installation.html)).

```
pip install -U nnabla-ext-cuda
```

Each folder usually contains:

* A README file.
* One or more example scripts and helper Python modules.
* A Python package requirement file, `requirements.txt`.

Before running a script in a example folder, it is recommended to read the README file, and install dependency packages with `pip install -U requirements.txt`. 
