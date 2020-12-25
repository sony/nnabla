# C++ inference with MNIST classification model.

## Introduction

This example demonstrates the workflow to train a classification model in Python and to execute it in C++. Although this example shows how to use it on Ubuntu 16.04 so far, a similar procedure should work on the other operating systems with little effort for the build scripts adapting to your OS. We will add some of more useful examples in the near future.

# Install C++ libraries

Follow [the installation manual of C++ utility library](../../../doc/build/build.md)

Note: this example requires NNabla Python package also be installed.

## Train a classification model in Python
At first, you will train an MNIST classification model in Python-side. The example scripts of the MNIST classification training are provided in [NNabla Examples repository](https://github.com/sony/nnabla-examples). Clone or download it, then you can run train a classification model by the following commands.

```shell
# at nnabla-examples/mnist-collection/
python classification.py  # Optionally you can use -c cudnn option.
```

After training finishes, you can find a parameter file created in the `tmp.monitor` folder with name `lenet_params_010000.h5`

## Create NNP file

In order to execute your trained model on C++ code, the trained model parameters must be converted to a NNabla file format (NNP) with a network definition. NNabla file format can store the information of network definitions, parameters, and executor settings etc. We provide an example script (found in this folder) which creates a NNP file from learned parameters and a Python script of model definition.

```shell
# at .
NNABLA_EXAMPLES_ROOT=<your local path to nnabla-examples> python save_nnp_classification.py
```

It reads parameter file, and creates a computation graph using loaded parameters. The computation graph is only used to dump the network structure into NNP file.

```python
    runtime_contents = {
        'networks': [
            {'name': 'runtime',
             'batch_size': args.batch_size,
             'outputs': {'y': pred},
             'names': {'x': image}}],
        'executors': [
            {'name': 'runtime',
             'network': 'runtime',
             'data': ['x'],
             'output': ['y']}]}
    nn.utils.save.save(nnp_file, runtime_contents)
```
In the above code, the network structure containing parameters and the execution configuration is saved into the NNP file `lenet_010000.nnp`. The contents is described in a JSON like format. In the `networks` field, you add a network with a name of `runtime`. It has a default batch size. The computation graph can be set by the output variable `pred` in the `outputs` field. At the same time, the output variable `pred` of the computation graph is registered as a name `y`. To query an input or intermediate variable in the computation graph via the C++ interface, you should set a filed `names` in a format of `{<name>: <Variable>}`.

The named variables in the network are referenced by the `executors` config. The executor config is used in C++ for executing a network in a more simpler way. The executor `runtime` is added where the network `runtime` is executed. The input and output variables are specified by names that are registered in the `networks` field.

## Build MNIST inference example C++ code
You can find an executable file 'mnist_runtime' under the build directory located at nnabla/build/bin.
If you want to build it yourself using Makefile you can refer to the following process in linux environments.
Also you can build an executable file 'mnist_runtime_cuda', that is not in the build directory,  by following process.


```shell
make
```

The above command generates an executable `mnist_runtime` at the current directly.

The build file `GNUmakefile` is really simple. It links `libnnabla.so` and `libnnabla_utils.so` with the executable generated from `mnist_runtime.cpp`, and compiles with C++11 option `-std=c++11`.

```shell
CUDA_VERSION_SUFFIX=-100_7 make cuda
```
`CUDA_VERSION_SUFFIX` depends on the cuda library version you are using, you may check it in /usr/local/lib.

You can also compile an executable `mnist_runtime_cuda` that runs computation on your CUDA device by the above command if you install `nnabla-ext-cuda` in a right path. See `GNUmakefile` for details.


## Execute handwritten digit classification

By running the generated example with no argument, you can see the usage documentation.

```shell
./mnist_runtime
```

Output:
```
Usage: ./mnist_runtime nnp_file input_pgm

Positional arguments:
  nnp_file  : .nnp file created by examples/vision/mnist/save_nnp_classification.py.
  input_pgm : PGM (P5) file of a 28 x 28 image where pixel values < 256.
```

Sample images that I created using GIMP editor are located in this folder.

0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9
:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:
![0](./original_images/0.png "0")|![1](./original_images/1.png "1")|![2](./original_images/2.png "2")|![3](./original_images/3.png "3")|![4](./original_images/4.png "4")|![5](./original_images/5.png "5")|![6](./original_images/6.png "6")|![7](./original_images/7.png "7")|![8](./original_images/8.png "8")|![9](./original_images/9.png "9")

The following command executes image classification with the trained model `lenet_010000.nnp` given an input image.

![5](./original_images/5.png "5")

```shell
./mnist_runtime lenet_010000.nnp 5.pgm
```

The output shows it makes a prediction. In my case, it's correct.
```
Prediction scores: -24.1875 -14.0103 -13.2646 7.52215 -13.7401 31.1683 -0.501035 -4.69472 6.2626 1.87513
Prediction: 5
```
NOTE: The recognition performance is not perfect for the real hand-written digit images (i.e. the digit images contained in this example). For example, it often misclassifies the digit 6 as 5.

## Walk through the example code

1. Add include for the NNabla header files.
```c++
#include <nbla/logger.hpp>
#include <nbla_utils/nnp.hpp>
```

2. Create a execution engine context. Following configuration enables our cached (memory pool) cpu array as array backend.
```c++
nbla::Context ctx{"cpu", "CpuCachedArray", "0", "default"};
```

3. Create `Nnp` object with the default context.
```c++
nbla::utils::nnp::Nnp nnp(ctx);
```

4. Set nnp file to the `Nnp` object. It immediately parses the file format and stores the extracted info.
```c++
nnp.add(nnp_file);
```

5. Get an executor instance. The above `save_nnp_classification.py` script saved an executor named `runtime`.
```c++
auto executor = nnp.get_executor("runtime");
```

5. Overwrite batch size as 1. This example always takes input for each image.
```c++
executor->set_batch_size(1); // Use batch_size = 1.
```

6. Get input data as a CPU array. See [computation_graph/variable.hpp](../../../include/nbla/computation_graph/variable.hpp) and [variable.hpp](../../../include/nbla/variable.hpp) for API manual written in the headers.
```c++
nbla::CgVariablePtr x = executor->get_data_variables().at(0).variable;
uint8_t *data = x->variable()->cast_data_and_get_pointer<uint8_t>(ctx);
```

7. Read input pgm file and store image data into the CPU array. The `read_pgm_mnist` is implemented above the `main` function.
```c++
read_pgm_mnist(input_bin, data);
```

8. Execute prediction.
```c++
executor->execute();
```

9. Get output as an CPU array.
```c++
nbla::CgVariablePtr y = executor->get_output_variables().at(0).variable;
const float *y_data = y->variable()->get_data_pointer<float>(ctx);
```

10. Show prediction scores and the most likely predicted number of the input image.
```c++
int prediction = 0;
float max_score = -1e10;
std::cout << "Prediction scores:";
for (int i = 0; i < 10; i++) {
  if (y_data[i] > max_score) {
	prediction = i;
	max_score = y_data[i];
  }
  std::cout << " " << std::setw(5) << y_data[i];
}
std::cout << std::endl;
std::cout << "Prediction: " << prediction << std::endl;
```
