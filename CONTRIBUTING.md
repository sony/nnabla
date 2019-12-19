# Contributing Guide

## Issue Tracker

We use an [issue tracker](https://github.com/sony/nnabla/issues) hosted in the [NNabla GitHub repository](https://github.com/sony/nnabla).
If you encounter any bugs, or come up with any feature requests, please first search if there is an already existing issue. If you can not find any, please feel free to post a new issue.

Note:
    This issue tracker is only used for development issues such as a bug report and a new feature request or proposal.
    Please do **NOT** post usage, installation and neural network modeling questions to the issue tracker.
    Such questions are welcome at the [NNabla user group](https://groups.google.com/forum/#!forum/nnabla).


## Contributing by Pull Request


We appreciate contributors in the community, that are willing to improve NNabla.
We basically follow the development style used in many GitHub repositories.

1. Search existing issues and/or pull request in
   [the GitHub repository](https://github.com/sony/nnabla).

2. If it doesn't exist, post an issue for the feature proposal.

3. Fork the repository, and develop your feature in the forked repo.

4. Format your code using auto format. (Refer: Auto format guidelines section below)

5. Create a pull request of your development branch to NNabla's `master` branch.
   Our maintainers will then review your changes.

6. Once your change is finalized, the maintainer will merge your change.


## Auto Formatting Guidelines

Auto formatting  is an inbuilt command to automatically format Python code to confirm to the PEP 8 style guide and C/C++/Objective-C code to adhere clang-format style.
It is highly recommended to format your changed code before opening pull requests, which will save your and the reviewers' time.

To apply auto formatting, move to your branch, choose one of the below options of applying auto formatting.
 
### Auto Formatting Using Dockers

Run following command to apply auto formatting for code.

For nnabla repository:
```shell
cd {nnabla repository root}
make bwd-nnabla-auto-format
```
For nnabla-ext-cuda repository:
```shell
cd {nnabla-ext-cuda repository root}
make bwd-nnabla-ext-cuda-auto-format
```

### Auto Formatting Without Using Dockers

#### Prerequisites

Please install following packages.
1. Python3 and packages [refer here](https://nnabla.readthedocs.io/en/latest/python/install_on_linux.html#prerequisites)
2. clang-format-3.8
3. autopep8 package

For Ubuntu 16.04 platform:

To install clang-format use following command,
```shell
sudo apt-get update
sudo apt-get install clang-format-3.8
```
To install autopep8 use following command,
```shell
pip install --upgrade autopep8
```
#### Running auto format command

Run following command to apply auto formatting for code.

For nnabla repository:
```shell
cd {nnabla repository root}
make nnabla-auto-format
```
For nnabla-ext-cuda repository:
```shell
cd {nnabla-ext-cuda repository root}
make nnabla-ext-cuda-auto-format
```
## Development Guide

* Architecture overview (available soon).
* [Adding a new function (layer implementation)](doc/contributing/add_function.md).
* [Adding a new solver (gradient descent algorithm implementation)](doc/contributing/add_solver.md).
* [Contributing to NNabla CUDA extension](https://github.com/sony/nnabla-ext-cuda/blob/master/CONTRIBUTING.md).
* Adding a new extension (available soon).
