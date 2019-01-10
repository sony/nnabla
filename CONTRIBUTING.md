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

4. Create a pull request of your development branch to NNabla's `master` branch.
   Our maintainers will then review your changes.

5. Once your change is finalized, the maintainer will merge your change.


## Development Guide

* Architecture overview (available soon).
* [Adding a new function (layer implementation)](doc/contributing/add_function.md).
* [Adding a new solver (gradient descent algorithm implementation)](doc/contributing/add_solver.md).
* [Contributing to NNabla CUDA extension](https://github.com/sony/nnabla-ext-cuda/blob/master/CONTRIBUTING.md).
* Adding a new extension (available soon).
