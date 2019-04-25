# C++ training cli for MNIST classification model.

## Introduction

This example demonstrates the workflow to train a classification model in C++ training cli.

# Install C++ libraries

Please follow [the installation manual](https://github.com/sony/nnabla/blob/master/doc/build/build_cpp_utils.md).

Note: this example requires zlib and NNabla Python package installed.

## Download MNIST dataset and create cached files.
This example requires cached MNIST dataset files.
We provide an example script which creates them with utils in mnist-example collections.
```shell
python create_mnist_cached.py
```
This command create 'cache' directory in the current directory.

## Create NNP file of an initialized model for MNIST classification.
This example requires initialized model parameters and a network definition saved as an NNP file.
We provide an example script which creates the NNP file from a classification example in mnist-example collections.

```shell
export NNABLA_DIR=../../../../nnabla_examples/
python create_initialized_model.py
```

This script creates a NNP file including initialized parameters and information of configurations, networks, optimizers, monitors and datasets,
Following code specifies the information necessary for the network definition.

```python
    nnp_file = '{}_initialized.nnp'.format(args.net)
    training_contents = {
        'global_config': {'default_context': ctx},
        'training_config':
            {'max_epoch': args_added.max_epoch,
             'iter_per_epoch': args_added.iter_per_epoch,
             'save_best': True},
        'networks': [
            {'name': 'training',
             'batch_size': args.batch_size,
             'outputs': {'loss': loss_t},
             'names': {'x': x, 'y': t, 'loss': loss_t}},
            {'name': 'validation',
             'batch_size': args.batch_size,
             'outputs': {'loss': loss_v},
             'names': {'x': x, 'y': t, 'loss': loss_v}}],
        'optimizers': [
            {'name': 'optimizer',
             'solver': solver,
             'network': 'training',
             'dataset': 'mnist_training',
             'weight_decay': 0,
             'lr_decay': 1,
             'lr_decay_interval': 1,
	     'update_interval': 1
             }],
        'datasets': [
            {'name': 'mnist_training',
             'uri': 'MNIST_TRAINING',
             'cache_dir': args_added.cache_dir + '/mnist_training.cache/',
             'variables': {'x': x, 'y': t},
             'shuffle': True,
             'batch_size': args.batch_size,
             'no_image_normalization': True},
            {'name': 'mnist_validation',
             'uri': 'MNIST_VALIDATION',
             'cache_dir': args_added.cache_dir + '/mnist_test.cache/',
             'variables': {'x': x, 'y': t},
             'shuffle': False,
             'batch_size': args.batch_size,
             'no_image_normalization': True
             }],
        'monitors': [
            {'name': 'training_loss',
             'network': 'validation',
             'dataset': 'mnist_training'},
            {'name': 'validation_loss',
             'network': 'validation',
             'dataset': 'mnist_validation'}],
    }

    nn.utils.save.save(nnp_file, training_contents)

```

In the above code, the initialized parameters and other configurations are saved into the NNP file `lenet_initialized.nnp`
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
nbla train lenet_initialized.nnp result
```

The above command creates result directory and we can see the logs of the training operation.
