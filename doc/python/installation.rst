.. _python-package-installation:

Python Package Installation
===========================

There are three ways to install NNabla Python package.


Install with pip command
----------------------

The NNabla python packages are hosted on PYPI for many platforms. For people who are familiar with Python and its package management system ``pip`` (and  optionally CUDA, but recommended), the following pip installation guide will be satisfactory when you install NNabla Python. To see the a bit more detailed OS specific setup guide, go to the next section.

.. toctree::
    :maxdepth: 1

    pip_installation.rst


.. _pip_os_specific:

OS specific workflows
~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :maxdepth: 1

    install_on_linux.rst
    install_on_windows.rst
    install_on_macos.rst


Install NNabla package compatible with Multi-GPU execution
----------------------------------------------------------

To enable multi-GPU execution such as distributed training on NNabla, you have to install a special edition of NNabla package. See :ref:`pip-installation-distributed` for installation.


Install from source
-------------------

Documentation of build from source has been moved to `Github repository <https://github.com/sony/nnabla-ext-cuda/tree/master/doc/build>`_ (`build <https://github.com/sony/nnabla/tree/master/doc/build/build.md>`_ or `build_distributed <https://github.com/sony/nnabla/tree/master/doc/build/build.md>`_).


Running on Docker
-----------------

.. toctree::
    :maxdepth: 1

    docker.rst


