C++ API
=======

The C++ libraries currently provide:

* APIs to execute an inference of a trained model created by `Python APIs <http://nnabla.readthedocs.io/en/latest/python.html>`_
  and `Neural Network Console <https://dl.sony.com>`_, a Sony's GUI neural network IDE.
* A command line interface written in C++ which executes an inference.
* An example of how to use C++ API with a trained model.

We are still preparing a well-formatted C++ API reference manual, however you can read through the header files where most of classes and functions are documented in `Doxygen <http://www.doxygen.org/>`_ format. The header files can be found under `include <https://github.com/sony/nnabla/tree/master/include/>`_ directory.

The example `MNIST runtime <https://github.com/sony/nnabla/tree/master/examples/cpp/mnist_runtime>`_ is a good starting point to understand how to use C++ API for neural network inference.


.. toctree::
    :maxdepth: 1

    cpp/installation.rst
    cpp/command_line_interface.rst
    cpp/examples.rst
