# Neural Network Libraries - Examples

This repository contains working examples of [Neural Network Libraries](https://github.com/sony/nnabla/).
Before running any of the examples in this repository, you must install the Python package for Neural Network Libraries. The Python install guide can be found [here](https://nnabla.readthedocs.io/en/latest/python/installation.html).

Before running an example, also run the following command inside the example directory, to install additional dependencies:

```
cd example_directory
pip install -r requirements.txt
```


## Docker workflow

* Our Docker workflow offers an easy installation and setup of running environments of our examples.
* [See this page](doc/docker.md).


## Interactive Demos

We have prepared interactive demos, where you can play around without having to worry about the codes and the internal mechanism. You can run it directly on [Colab](https://colab.research.google.com/) from the links in the table below.


| Name        | Notebook           | Task  |
|:------------------------------------------------:|:-------------:|:-----:|
| ESR-GAN       | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sony/nnabla-examples/blob/master/interactive-demos/esrgan.ipynb) | Super-Resolution|
| Self-Attention GAN       | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sony/nnabla-examples/blob/master/interactive-demos/sagan.ipynb) | Image Generation|
| Face Alignment Network | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sony/nnabla-examples/blob/master/interactive-demos/fan.ipynb) | Facial Keypoint Detection |
| PSMNet | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sony/nnabla-examples/blob/master/interactive-demos/psmnet.ipynb) | Depth Estimation |
