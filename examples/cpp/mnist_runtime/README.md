# C++ inference with MNIST classification model.

## Introduction

This example demonstrates the workflow to train a classification model in Python and to execute it in C++. Although this example is only tested on Ubuntu 16.04 so far, a similar procedure should work on the other operating systems with little effort to the build system depending on the OSs. We will add some of more useful examples in the near future.


## Train a classification model in Python
First, you run a Python training script provided in the MNIST example folder in order to get a trained model for MNIST digit classification. 

```shell
# at examples/vision/mnist
python classification.py  # Optionally you can use -c cuda.cudnn option.
```

After training finishes, you can find a parameter file created in the `tmp.monitor` folder with name `lenet_params_010000.h5`

### Create NNP file

In order to execute your trained model on C++ code, the trained model parameters must be converted to a NNabla file format (NNP) with a network definition. NNabla file format can store the information of network definitions, parameters, and executor settings etc. We provide an example script which creates a NNP file from learned parameters and a Python script of model definition.

```shell
# at examples/vision/mnist
python save_nnp_classification.py
```

It reads parameter file, and creates a computation graph using loaded parameters. The computation graph is only used for dumping the network structure into NNP file.

```pyton
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
In the above code, the network structure containing parameters and the execution configuration is saved into the NNP file. The contents is described in a JSON like format. In the `netoworks` field, a newtork with a name `runtime`. It has a default batch size. The computation graph can be set by the output variable `pred` in the `outputs` field. At the same time, the output variable `pred` of the computation graph is registered as a name `y`. To query an input or intermediate variable in the computation graph via the C++ interface, you should set a filed `names` in a format of `{<name>: <Variable>}`.

The named variables are actually reference at the `executors` config. The executor config is used in C++ for executing network in a more simpler way. The executor `runtime` is added where the newtork `runtime` is executed. The input and output variables are specified by names that are registered in the `networks` field.

TODO: EDITING.


