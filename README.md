# Neural Network Libraries

Neural Network Libraries is a deep learning framework that is intended to be used for research,
development and production. We aim to have it running everywhere: desktop PCs, HPC
clusters, embedded devices and production servers.


* [Neural Network Console](https://dl.sony.com/), a Windows GUI app for neural network development, has been released.
* The GitHub repository of CUDA extension of Neural Network Libraries can be found [here](https://github.com/sony/nnabla-ext-cuda).


## Installation

Installing Neural Network Libraries is easy:

```
pip install nnabla
```

This installs the CPU version of Neural Network Libraries. GPU-acceleration can be added by installing the CUDA extension with `pip install nnabla-ext-cuda`.


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

### Setup

<https://nnabla.readthedocs.io/en/latest/python/installation.html>


### Getting started

* A number of Jupyter notebook tutorials can be found in the [tutorial](https://github.com/sony/nnabla/tree/master/tutorial) folder.
  We recommend starting from `by_examples.ipynb` for a first
  working example in Neural Network Libraries and `python_api.ipynb` for an introduction into the
  Neural Network Libraries API.

* We also provide some more sophisticated examples in the [`examples`](https://github.com/sony/nnabla/tree/master/examples) folder.

* C++ API examples are avaiailable in [`exampels/cpp`](https://github.com/sony/nnabla/tree/master/examples/cpp).
