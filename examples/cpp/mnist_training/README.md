# C++ training with MNIST classification model.

## Introduction

This example demonstrates the workflow to train a classification model in C++.
Although this example is only tested on Ubuntu 16.04 so far,
a similar procedure to build the system should work on other operating systems with little effort.
We will add more useful examples in near future.

# Install C++ libraries

Please follow [the installation manual](https://github.com/sony/nnabla/blob/master/doc/build/build_cpp_utils.md).

Note: this example requires zlib and NNabla Python package installed.

Also MNIST dataset is required in the same directory.
It can be downloaded from the following URLs.
* Training images : http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
* Training labels : http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz

## Create NNP file of an initialized model for MNIST classification.
This sample requires initialized model parameters and a network definition saved as an NNP file.
We provide an example script which creates the NNP file from a classification example in mnist-example collections.

```shell
python create_initialized_model.py
```

This script imports definition of the network, and creates initialized parameters and a network architecture definition into a NNP file.
Following code specifies the information necessary for the network definition.

```python
    runtime_contents = {
        'networks': [
            {'name': 'training',
             'batch_size': args.batch_size,
             'outputs': {'loss': loss},
             'names': {'x': x, 'y': y}}]}
    nn.utils.save.save(nnp_file, runtime_contents)
```

In the above code, the network structure and initialized parameters are saved into the NNP file `lenet_initialized.nnp`
You can see the contents by unzipping the file.

The network structure contents are described in a JSON like format.
In the `networks` field, a network is given a name `training`. It has a default batch size. 
The computation graph can be set by the output variable `loss` in the `outputs` field. 
At the same time, the input variables `x` and `y` of the computation graph are registered in `names` field. 
To query an input or intermediate variable in the computation graph via the C++ interface, you should set a filed `names` in a format of `{<name>: <Variable>}`.

## Build MNIST training example in C++ code
You can find an executable file 'mnist_training' under the build directory located at nnabla/build/bin.
If you want to build it yourself using makefile you can refer to the following commands in linux environments.

```shell
make
```

The above command generates an executable `mnist_training` at the current directory.

The build file `GNUmakefile` is simple.
It links `libnnabla.so`, `libnnabla_utils.so` and `libz.so` with the executable generated from `main.cpp`, and compiles with C++11 option `-std=c++11`.

You can also compile an executable `mnist_training_cuda` that runs computation on your CUDA device.
Please download and refer to `nnabla-ext-cuda` repository for details.

## Handwritten digit training
By running the generated example with no argument, you can see the usage documentation.

```shell
./mnist_training
```

Output:
```

Usage: ./mnist_training model.nnp

  model.nnp : model file with initialized parameters.


```

The following command executes the training of the initialized model `lenet_initialized.nnp` on MNIST dataset.

```shell

./mnist_training lenet_initialized.nnp

```

The output file named `parameter.protobuf` contains the learned parameters.

Following process is temporary and at a later date, we will prepare a save function for nnp.

```shell
 cp lenet_initialized.nnp lenet_learned.nnp
 unzip lenet_learned.nnp
 zip lenet_learned.nnp nnp_version.txt network.nntxt parameter.protobuf
```

You will be asked "replace parameter.protobuf?" when unzipping, so please answer "n".

After getting learned.nnp, you can use it as a model file for "mnist_runtime".


## Walk through the example code

[main.cpp][1]
[1]:main.cpp
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
  mnist_training(ctx, argv[1]);
```

[mnist_training.hpp][2]
[2]:mnist_training.cpp
1. Add NNabla headers.
```c++
  #include <nbla_utils/nnp.hpp>
  #include <nbla/computation_graph/variable.hpp>
  #include <nbla/computation_graph/function.hpp>
  #include <nbla/solver/adam.hpp>
```

3. Create `Nnp` object and set nnp file.
```c++
  nbla::utils::nnp::Nnp nnp(ctx);
  nnp.add(nnp_file);
```

4. Get a network instance and set batchsize.
```c++
  auto net = nnp.get_network("training");
  net->set_batch_size(batch_size);
```

5. Load dataset to data iterator, modify this part depending on your purpose.
```c++
  MnistDataIterator data_iterator();
```
  This sample works only for the mnist training dataset downloaded to this directory

6. Create solver and set parameters.
```c++
  auto adam = create_AdamSolver(ctx, 0.001, 0.9, 0.999, 1.0e-6);
  auto parameters = nnp.get_parameters();
  adam->set_parameters(parameters);
```

7. Get input data as a CPU array.
```c++
  nbla::CgVariablePtr x = net->get_variable("x");
  nbla::CgVariablePtr y = net->get_variable("y");
  nbla::CgVariablePtr loss = net->get_variable("loss");
  float *x_d = x->variable()->cast_data_and_get_pointer<float>(ctx);
  int *y_d = y->variable()->cast_data_and_get_pointer<int>(ctx);
```
8. Provide minibatch in training loop
```c++
  float *x_d = x->variable()->cast_data_and_get_pointer<float>(ctx);
  int *y_d = y->variable()->cast_data_and_get_pointer<int>(ctx);
```
In order to sync with the memory of the GPU, cast processing should be inside the iteration loop.

9. Execute training loop with forward, backward and update.
```c++
  adam->zero_grad();
  loss->forward(/*clear_buffer=*/false, /*clear_no_need_grad=*/false);
  loss->variable()->grad()->fill(1);
  loss->backward(/*NdArrayPtr grad =*/nullptr, /*bool clear_buffer = */false);
  adam->update();
```

10. Show mean loss.
```c++
  float *loss_d = loss->variable()->cast_data_and_get_pointer<float>(ctx);
  mean_loss += loss_d[0];
  if ((iter + 1) % n_val_iter == 0) {
    mean_loss /= n_val_iter;
    std::cout << "iter: " << iter + 1 << ", loss: " << loss_d[0] << std::endl;
  }
```
