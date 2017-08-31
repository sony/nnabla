.. _python-build-on-macos:

Build on macOS
--------------


.. contents::
   :local:
   :depth: 1

Prerequisites
^^^^^^^^^^^^^

Our build system requires:

* `CMake <https://cmake.org/>`_ (>=3.1)
* macOS 10.9 or later
* Clang compilers
* Python 2.7/3.4/3.5/3.6: Tested on Miniconda3.
* Protobuf compiler: Find a binary executable for OS x86_64 architecture at <https://github.com/google/protobuf/releases>. Download and extract the zip file, then locate the binary executable ``protoc`` to the location where path is set.

NOTE: Building with CUDA extension is not supported so far.

Build and installation
^^^^^^^^^^^^^^^^^^^^^^

Once the requirements described above are set up, the following procedure is exactly same as that in the build on Linux except CUDA installation is not supported on macOS. See :ref:`linux-build-and-install`.
