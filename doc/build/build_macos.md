# Build Python Package on macOS

## Prerequisites

Our build system requires:

* [CMake](https://cmake.org/) (>=3.14.3)
* macOS 10.9 or later
* Clang compilers
* Python >= 3.7: Tested on Python 3.8.9.
* Protobuf compiler: Find a binary executable for OS x86_64 architecture [here](https://github.com/google/protobuf/releases). Download and extract the zip file, then locate the binary executable ``protoc`` to the location where path is set. Please note that nnabla has python package dependency of [protobuf](https://github.com/nnabla/nnabla/blob/master/python/setup.py#L43), if the ``protoc`` binary you used is not compatible, please try another version of ``protoc``. 

## Build and installation

Once the requirements described above are set up, the rest of the procedure is exactly same as that in the build on Linux. See [Build Python Package from Source](build.md#build-and-installation).
