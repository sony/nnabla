# Build Python Package on macOS

## Prerequisites

Our build system requires:

* [CMake](https://cmake.org/) (>=3.1)
* macOS 10.9 or later
* Clang compilers
* Python >= 3.6: Tested on Miniconda3.
* Protobuf compiler: Find a binary executable for OS x86_64 architecture [here](https://github.com/google/protobuf/releases). Download and extract the zip file, then locate the binary executable ``protoc`` to the location where path is set.

## Build and installation

Once the requirements described above are set up, the rest of the procedure is exactly same as that in the build on Linux. See [Build Python Package from Source](build.md).
