.. _cpp-lib-installation-cross-compile:

Cross Compile
=============

.. contents::
   :local:
   :depth: 1

Requirements
------------

Additional requirements to :ref:`cpp-lib-installation`

* Cross compiler C/C++ toolchain
* Cross compiled Protobuf, libArchive and zlib

Build
-----

.. code-block:: shell

    git clone https://github.com/sony/nnabla
    mkdir -p nnabla/build && cd nnabla/build
    cmake -DBUILD_PYTHON_PACKAGE=OFF -DBUILD_CPP_UTILS=ON -DBUILD_CPP_UTILS_LIB_ONLY=ON -DCMAKE_CXX_COMPILER=<path_to_g++> -DCMAKE_C_COMPILER=<path_to_gcc> -DLibArchive_LIBRARY=<path_to_libArchive> -DZLIB_LIBRARY=<path_to_zlib> -DPROTOBUF_LIBRARY=<path_to_libProtobuf> -DNBLA_CROSS_INCLUDE_DIRS=<path_to_libProtobuf_inc>;<path_to_libArchive_inc>;<path_to_zlib_inc> ..
    make
