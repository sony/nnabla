========================
Neural Network Libraries
========================

Neural Network Libraries is deep learning framework that is intended to be used for research, development, and production. We aim it running everywhere like desktop PCs, HPC clusters, embedded devices and production servers.

This document describes how to use the Python API and C++ API, the contribution guide for developers, and the license term of this software. The Python API is more suitable for fast prototyping and experimentation of deep learning systems, while the C++ API is for deploying inference or training algorithms into embedded systems and servers (The documentation is not available so far. We will make it available soon). The framework is designed modularity and extensibility in mind. Community contributors can add a new operator or optimizer module of neural networks, and a specialized implementation of neural network modules for a specific target device as an extension.


.. toctree::
    :maxdepth: 2

    python.rst
    cpp.rst
    data_exchange_file_format
    format.rst
    python/file_format_converter/file_format_converter.rst
    contributing.rst
    license.rst


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
