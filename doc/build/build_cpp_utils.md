# Build C++ utility libraries

This document describes how to build and install C++ libraries,
headers and executables that can be used for C++ standalone inference
and training. The following instruction demonstrates the build
procedure on Ubuntu 16.04, but we successfully build C++ libraries on
macOS and Windows ([manual](./build_cpp_utils_windows.md)) too in a
similar way (the differences lie in the installation of some
dependencies). We may add build instructions on another platform in
the future.

Most of the procedure is the same as [Build Python Package from
Source](build.md).

## Requirements

* G++: `sudo apt-get install build-essential`
* CMake>=3.1: `sudo apt-get install cmake`
* Python: `sudo apt-get install python python-pip`(Used by code generator)
  * Python packages: PyYAML and MAKO: `sudo -H pip install pyyaml mako`
* LibArchive: `sudo apt-get install libarchive-dev`
* HDF5 (Optional): `sudo apt-get install libhdf5-dev`
* Protobuf >=3: See below.


### Installing protobuf3 C++ libraries and tools

#### Install from source.

Unlike [Python Package compilation](./build.md) which requires
`protoc` compiler only, the NNabla C++ utility library requires
protobuf C++ library too.  The following snippet running on your
terminal will build and install protobuf-3.1.0 from source.

```shell
curl -L https://github.com/google/protobuf/archive/v3.1.0.tar.gz -o protobuf-v3.1.0.tar.gz
tar xvf protobuf-v3.1.0.tar.gz
cd protobuf-3.1.0
mkdir build && cd build
cmake -DCMAKE_POSITION_INDEPENDENT_CODE=ON -Dprotobuf_BUILD_TESTS=OFF ../cmake
make
sudo make install  # See a note below if you want your system clean.
```

NOTE: Installing protobuf on your system with sudo may harm a protobuf
library previously installed on your system. It is recommended to
install a protobuf to a user folder. Specify a cmake option
`-DCMAKE_INSTALL_PREFIX=<installation path>` to set your preferred
installation path (e.g. `$HOME/nnabla_build_deps`), and run `make
install` without sudo. In the NNabla C++ library compilation described
below, to help CMake system find the protobuf, pass the installation
path to an environment variable `CMAKE_FIND_ROOT_PATH`. NNabla's CMake
script is written to use the value (multiple paths can be passed with
a delimiter `;`, e.g. `CMAKE_FIND_ROOT_PATH="<path A>;<path
B>";${CMAKE_FIND_ROOT_PATH}`) as search paths to find some packages
(e.g. protobuf, hdf5, libarchive) on your system. Please don't forget
to set environment variables such as a executable path and library
paths (e.g. `PATH`, `LD_LIBRARY_PATH` etc) if you have runtime
dependencies at your custom installation path.

#### Install from PPA package.

Here is the procedure using an informal PPA package. If you can not
trust unofficial packages, please use the procedure to build from the
source shown above.

```shell
sudo add-apt-repository ppa:maarten-fonville/protobuf
sudo apt install protobuf-compiler libprotoc-dev libprotobuf-dev
```

## Build

```shell
git clone https://github.com/sony/nnabla
mkdir -p nnabla/build && cd nnabla/build
cmake .. -DBUILD_CPP_UTILS=ON -DBUILD_PYTHON_PACKAGE=OFF -DNNABLA_UTILS_WITH_HDF5=ON
make
```

Some optional arguments for `cmake`:

* `-DNNABLA_UTILS_WITH_HDF5=OFF` to turn off HDF5 feature if you stacked in HDF5 installation.
* `-DNNABLA_UTILS_WITH_NPY=ON` to turn on NPY feature if you want to use *.npy cache files.
* `-DBUILD_PYTHON_PACKAGE=ON` to build Python package too.

The following command will install the libraries, the command line
executables and the include header files to your system.

```shell
sudo make install
```
