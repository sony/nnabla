# Neural Network Libraries

Neural Network Libraries is a deep learning framework that is intended to be used for research,
development and production. We aim to have it running everywhere: desktop PCs, HPC
clusters, embedded devices and production servers.


* [Neural Network Libraries - CUDA extension](https://github.com/sony/nnabla-ext-cuda): An extension library of Neural Network Libraries that allows users to speed-up the computation on CUDA-capable GPUs.
* [Neural Network Libraries - Examples](https://github.com/sony/nnabla-examples): Working examples of Neural Network Libraries from basic to state-of-the-art.
* [Neural Network Libraries - C Runtime](https://github.com/sony/nnabla-c-runtime):  Runtime library for inference Neural Network created by Neural Network Libraries.
* [Neural Network Console](https://dl.sony.com/): A Windows GUI app for neural network development.


## Installation

Installing Neural Network Libraries is easy:

```
pip install nnabla
```

This installs the CPU version of Neural Network Libraries. GPU-acceleration can be added by installing the CUDA extension with following command.  
`pip install nnabla-ext-cuda101`  
for version 10.1 CUDA Toolkit.  

for other versions:  
`pip install nnabla-ext-cuda100` for CUDA version 10.0.  
`pip install nnabla-ext-cuda90` for CUDA version 9.0.  
`pip install nnabla-ext-cuda80` for CUDA version 8.0.  
  
CUDA ver. 9.1, ver. 9.2 are not supported now.  


For more details, see the [installation section](http://nnabla.readthedocs.io/en/latest/python/installation.html) of the documentation.

### Building from Source

See [Build Manuals](doc/build/README.md).

### Running on Docker
For details on running on Docker, see the [installation section](http://nnabla.readthedocs.io/en/latest/python/installation.html) of the documentation.

## Features

### Easy, flexible and expressive

The Python API built on the Neural Network Libraries C++11 core gives you flexibility and
productivity. For example, a two layer neural network with classification loss
can be defined in the following 5 lines of codes (hyper parameters are enclosed
by `<>`).

```python
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF

x = nn.Variable(<input_shape>)
t = nn.Variable(<target_shape>)
h = F.tanh(PF.affine(x, <hidden_size>, name='affine1'))
y = PF.affine(h, <target_size>, name='affine2')
loss = F.mean(F.softmax_cross_entropy(y, t))
```

Training can be done by:

```python
import nnabla.solvers as S

# Create a solver (parameter updater)
solver = S.Adam(<solver_params>)
solver.set_parameters(nn.get_parameters())

# Training iteration
for n in range(<num_training_iterations>):
    # Setting data from any data source
    x.d = <set data>
    t.d = <set label>
    # Initialize gradients
    solver.zero_grad()
    # Forward and backward execution
    loss.forward()
    loss.backward()
    # Update parameters by computed gradients
    solver.update()
```

The dynamic computation graph enables flexible runtime network construction.
Neural Network Libraries can use both paradigms of static and dynamic graphs,
both using the same API.

```python
x.d = <set data>
t.d = <set label>
drop_depth = np.random.rand(<num_stochastic_layers>) < <layer_drop_ratio>
with nn.auto_forward():
    h = F.relu(PF.convolution(x, <hidden_size>, (3, 3), pad=(1, 1), name='conv0'))
    for i in range(<num_stochastic_layers>):
        if drop_depth[i]:
            continue  # Stochastically drop a layer
        h2 = F.relu(PF.convolution(x, <hidden_size>, (3, 3), pad=(1, 1), 
                                   name='conv%d' % (i + 1)))
        h = F.add2(h, h2)
    y = PF.affine(h, <target_size>, name='classification')
    loss = F.mean(F.softmax_cross_entropy(y, t))
# Backward computation (can also be done in dynamically executed graph)
loss.backward()
```

### Command line utility

Neural Network Libraries provides a command line utility `nnabla_cli` for easier use of NNL.

`nnabla_cli` provides following functionality.

- Training, Evaluation or Inference with NNP file.
- Dataset and Parameter manipulation.
- File format converter
  - From ONNX to NNP and NNP to ONNX.
  - From ONNX or NNP to NNB or C source code.

For more details see [Documentation](doc/python/command_line_interface.rst)


### Portable and multi-platform

* Python API can be used on Linux and Windows
* Most of the library code is written in C++11, deployable to embedded devices

### Extensible

* Easy to add new modules like neural network operators and optimizers
* The library allows developers to add specialized implementations (e.g., for
  FPGA, ...). For example, we provide CUDA backend as an extension, which gives
  speed-up by GPU accelerated computation.

### Efficient

* High speed on a single CUDA GPU
* Memory optimization engine
* Multiple GPU support


## Documentation

<https://nnabla.readthedocs.org>

### Getting started

* A number of Jupyter notebook tutorials can be found in the [tutorial](https://github.com/sony/nnabla/tree/master/tutorial) folder.
  We recommend starting from `by_examples.ipynb` for a first
  working example in Neural Network Libraries and `python_api.ipynb` for an introduction into the
  Neural Network Libraries API.

* We also provide some more sophisticated examples at [`nnabla-examples`](https://github.com/sony/nnabla-examples) repository.

* C++ API examples are available in [`examples/cpp`](https://github.com/sony/nnabla/tree/master/examples/cpp).


## Contribution guide

The technology is rapidly progressing, and researchers and developers often want to add their custom features to a deep learning framework.
NNabla is really nice in this point. The architecture of Neural Network Libraries is clean and quite simple.
Also, you can add new features very easy by the help of our code template generating system.
See the following link for details.

* [Contribution guide](CONTRIBUTING.md)
