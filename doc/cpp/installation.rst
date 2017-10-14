.. _cpp-lib-installation:

Build C++ libraries
===================


.. contents::
   :local:
   :depth: 1


Note: The C++ inference with CUDA extension is not covered in this build manual. We'll add it very soon.

This document describes how to build and install C++ libraries, headers and executabls on Ubuntu 16.04 using CMake build system.
We successfully build C++ libraries on macOS too in a similar way (the differences are at the installation of some dependencies). We may add build instractions on another platform in the future.

Most of the procedure is the same as :ref:`python_build_on_linux`.


Requirements
------------

* G++
* CMake>=3.1: ``sudo apt-get install cmake``
* Python: ``sudo apt-get install python`` (Used by code generator)
* LibArchive: ``sudo apt-get install libarchive-dev``
* HDF5 (Optional): ``sudo apt-get instasll libhdf5-dev``
* Protobuf >=3: The following snippet running on your terminal will build and install protobuf-3.1.0 from source. See the following NOTE to prevent overwriting previously installed Protobuf on your system (The distributed version of Protobuf by apt on Ubuntu 16.04 is 2.6).

.. code-block:: shell

    curl -L https://github.com/google/protobuf/archive/v3.1.0.tar.gz -o protobuf-v3.1.0.tar.gz
    tar xvf protobuf-v3.1.0.tar.gz
    cd protobuf-3.1.0
    mkdir build && cd build
    cmake -DCMAKE_POSITION_INDEPENDENT_CODE=ON -Dprotobuf_BUILD_TESTS=OFF ../cmake
    make
    sudo make install

NOTE: Installing protobuf on your system with sudo may harm a protobuf library previously installed on your system. It is recommended to install a protobuf to a user folder. Sepecify a cmake option ``-DCMAKE_INSTALL_PREFIX=<installation path>` to set your preferred installation path (e.g. ``$HOME/nnabla_build_deps``), and run ``make install` without sudo. To help CMake system find the protobuf, pass the installation path to an environment variable ``CMAKE_FIND_ROOT_PATH``. NNabla's CMake script is written to use the value (multiple paths can be passed with a delimiter ``;``, e.g. ``CMAKE_FIND_ROOT_PATH="<path A>;<path B>";${CMAKE_FIND_ROOT_PATH}``) as search paths to find some packages (e.g. protobuf, hdf5, libarchive) on your system. Please don't forget to set environment variables such as a executable path and library paths (e.g. ``PATH``, ``LD_LIBRARY_PATH`` etc) if you have runtime dependencies at your custom installation path.


Build
-----

.. code-block:: shell

    git clone https://github.com/sony/nnabla
    mkdir -p nnabla/build && cd nnabla/build
    cmake .. -DBUILD_CPP_UTILS=ON -DBUILD_PYTHON_PACKAGE=OFF -DNNABLA_UTILS_WITH_HDF5=ON
    make

If you want to disable the HDF5 support, set ``-DNNABLA_UTILS_WITH_HDF5=OFF``.

The following command will install the libraries, the command line executables and the include header files to your system.

.. code-block:: shell

   sudo make install
