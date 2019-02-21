# Python-like C++ training collection using MNIST dataset.

## Introduction

These examples demonstrate the workflow to design and train models for MNIST in C++.
Although these are only tested on Ubuntu 16.04 so far,
a similar procedure to build the system should work on other operating systems with little effort.

## Install C++ libraries

Please follow [the installation manual](https://github.com/sony/nnabla/blob/master/doc/build/build_cpp_utils.md).

Note: this example requires zlib and NNabla Python package installed.

Also MNIST dataset is required in the same directory.
It can be downloaded from the following URLs.
* Training images : http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
* Training labels : http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
* Test images : http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
* Test labels : http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

## Build training example in C++ code
You can find executable files `train_lenet_classifier`, `train_resnet_classifier`, `train_vae`, `train_siamese`, `train_dcgan` and `train_vat`  under the build directory located at nnabla/build/bin.
If you want to build it yourself using makefile you can refer to the following command in linux environments.

```shell
make lenet
```

The above command generates an executable `train_lenet_classifier` at the current directory.

The makefile `GNUmakefile` is simple.
It links `libnnabla.so` and `libz.so` with the executable generated from `train_lenet_classifier.cpp`, and compiles with C++11 option `-std=c++11`.

You can also compile other executables by using make command with proper options.
Also you can compile an executable `training_lenet_classifier_cuda` and other cuda versions that run computation on your CUDA device.
Please download and refer to `nnabla-ext-cuda` repository for details.

## Run training example
By running the generated example without any arguments.

```shell
./train_lenet_classifier
```

Output:
```
iter: 9, tloss: 2.061232, verr: 0.501562
iter: 19, tloss: 1.466847, verr: 0.282031
iter: 29, tloss: 0.900860, verr: 0.189063
iter: 39, tloss: 0.620210, verr: 0.186719
...
```

You can also run other examples by using each executable.


## Walk through the example code

[train_lenet_classifier.cpp][1]
[1]:train_lenet_classifier.cpp
1. Add NNabla headers.
```c++
  #include <nbla/context.hpp>
```

2. Create an execution engine context.
```c++
  nbla::Context ctx{{"cpu:float"}, "CpuCachedArray", "0"};
```

3. Execute training.
```c++
  lenet_training(ctx);
```

[lenet_training.hpp][2]
[2]:lenet_training.hpp
1. Add NNabla headers.
```c++
  #include <nbla/computation_graph/computation_graph.hpp>
  #include <nbla/solver/adam.hpp>
  using namespace nbla;
  using std::make_shared;

  #include <nbla/functions.hpp>
  #include <nbla/global_context.hpp>
  #include <nbla/parametric_functions.hpp>
  namespace f = nbla::functions;
  namespace pf = nbla::parametric_functions;
```

2. Designe Lenet model
```c++
  auto h = pf::convolution(x, 1, 16, {5, 5}, parameters["conv1"]);
  h = f::max_pooling(h, {2, 2}, {2, 2}, true, {0, 0});
  h = f::relu(h, false);
  h = pf::convolution(h, 1, 16, {5, 5}, parameters["conv2"]);
  h = f::max_pooling(h, {2, 2}, {2, 2}, true, {0, 0});
  h = f::relu(h, false);
  h = pf::affine(h, 1, 50, parameters["affine3"]);
  h = f::relu(h, false);
  h = pf::affine(h, 1, 10, parameters["affine4"]);
```
 Inputs of functions are CgVariablePtr and outputs are vector<CgVariablePtr>.
 Similar to nnabla-python, f is namespace of functions without trainable parameters
 and pf is namespace of functions with trainable parameters.

3. Load dataset to data iterator, modify this part depending on your purpose.
```c++
  #include "mnist_data.hpp"
  MnistDataIterator train_data_provider("train");
  MnistDataIterator test_data_provider("test");
```
  This sample works only for the mnist training dataset downloaded to this directory

4. Build training model and
```c++
  SingletonManager::get<GlobalContext>()->set_current_context(ctx);
  ParameterDirectory params;
  int batch_size = 128;
  auto x = make_shared<CgVariable>(Shape_t({batch_size, 1, 28, 28}), false);
  auto t = make_shared<CgVariable>(Shape_t({batch_size, 1}), false);
  auto h = model(x, params);
  auto loss = f::mean(f::softmax_cross_entropy(h, t, 1), {0, 1}, false);
  auto err = f::mean(f::top_n_error(h, t, 1, 1), {0, 1}, false);
```

5. Create solver and set parameters.
```c++
  auto adam = create_AdamSolver(ctx, 0.001, 0.9, 0.999, 1.0e-8);
  adam->set_parameters(params.get_parameters());
```

6. Provide minibatch in training loop
   Copy data and label inside "mnist_data.hpp"
```c++
  float_t *x_d = x->variable()->cast_data_and_get_pointer<float_t>(cpu_ctx, true);
  uint8_t *t_d = t->variable()->cast_data_and_get_pointer<uint8_t>(cpu_ctx, true);
```
In order to sync with the memory of the GPU, cast processing should be inside the iteration loop.

7. Execute training loop with forward, backward and update.
```c++
  adam->zero_grad();
  loss->forward(/*clear_buffer=*/false, /*clear_no_need_grad=*/false);
  loss->variable()->grad()->fill(1);
  loss->backward(/*NdArrayPtr grad =*/nullptr, /*bool clear_buffer = */false);
  adam->update();
```
8. Show mean loss.
```c++
  float_t *t_loss_d = loss->variable()->cast_data_and_get_pointer<float_t>(cpu_ctx, false);
```

9. Show mean error.
```c++
  float_t *v_err_d = err->variable()->cast_data_and_get_pointer<float_t>(cpu_ctx, false);
```