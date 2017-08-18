.. _cpp-lib-installation:

Build C++ libraries
===================


.. contents::
   :local:
   :depth: 1


Note: The C++ inference with CUDA extension is not covered in this build manual. We'll add it very soon.

This document describes how to build and install C++ libraries, headers and executabls on Ubuntu 16.04 using CMake build system.
We will add the installation manual on the other platforms.


Most of the procedure is the same as :ref:`python_build_on_linux`.


Requirements
------------

Some additional dependencies are required on the Python intallation.


* HDF5, LibArchive: `sudo apt-get install libhdf5-dev libarchive-dev`

* Protobuf >=3: The following snippet running on your terminal will build and install protobuf-3.1.0 from source.

.. code-block:: shell

    curl -L https://github.com/google/protobuf/archive/v3.1.0.tar.gz -o protobuf-v3.1.0.tar.gz
    tar xvf protobuf-v3.1.0.tar.gz
    cd protobuf-3.1.0
    mkdir build && cd build
    cmake -DCMAKE_POSITION_INDEPENDENT_CODE=ON -Dprotobuf_BUILD_TESTS=OFF ../cmake
    make
    sudo make install

Build
-----

.. code-block:: shell

    git clone https://github.com/sony/nnabla
    mkdir -p nnabla/build && cd nnabla/build
    cmake .. -DBUILD_CPP_UTILS=ON -DBUILD_PYTHON_API=OFF
    make


The following command will install the libraries, the command line executables and the include header files to your system.

.. code-block:: shell

   sudo make install
