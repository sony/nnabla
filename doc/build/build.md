# Build Python Package from Source

This document demonstrates how to build NNabla Python package from source on Ubuntu 16.04 LTS. It also works for many other linux distributions by replacing the installation commands that use `apt` in this example by commands for a package manager on your system (e.g. `yum`). 

For build instructions for Windows and macOS, go to:

* [Build on Windows](build_windows.md)
* [Build on macOS](build_macos.md)

For the build instruction for C++ utility libraries, go to:

* [Build C++ utility libraries](build_cpp_utils.md)


## Setup

The following command install some system and dev tools.

```shell
sudo apt-get install -y cmake curl g++ make git
```

Python dependencies are installed by:

```shell
sudo apt install python-dev python-pip python-setuptools
sudo pip install --upgrade pip  # bump to latest version
sudo pip install --upgrade setuptools  # bump to latest version
```

You can also use another distribution (e.g. [Miniconda](https://conda.io/miniconda.html) is recommended).

[Google Protocol Buffer](https://github.com/google/protobuf) compiler is also required to create NNabla's neural network format serializer/desrializer in Python or C++.

```shell
curl -L https://github.com/google/protobuf/releases/download/v3.1.0/protoc-3.1.0-linux-x86_64.zip -o /tmp/protoc-3.1.0-linux-x86_64.zip
sudo unzip -d /usr/local /tmp/protoc-3.1.0-linux-x86_64.zip && sudo chmod 755 /usr/local/bin/protoc
```

## Build and installation

Get source code from Github.

```shell
git clone https://github.com/sony/nnabla
```

Install python requirements.

```shell
cd nnabla
sudo pip install -U -r python/setup_requirements.txt
sudo pip install -U -r python/requirements.txt
```

Build the C++ core libraries and NNabla Python package.

```shell
mkdir build
cd build
cmake ../
make
```

If you want to install nnabla for python 3.x, you may need to add `-DPYTHON_COMMAND_NAME=python3.x` to `cmake`. Without it, the installation may fail.
So replace `cmake ../`. with

```shell
cmake ../ -DPYTHON_COMMAND_NAME=python3.6  # if you use python 3.6
```

Be careful if you have multiple python versions.

Finally, you get the NNabla Python package installer as a wheel file found at `./dist`. You can install nnabla using the wheel file using `pip` command.

```shell
cd dist
sudo pip uninstall -y nnabla
sudo pip install nnabla-<package version>-<package-arch>.whl # a name depends on a version and an environment
```

## Verify installation by unit testing

For unit testing, some additional requirements should be installed.

```shell
cd nnabla
sudo pip install -U -r python/test_requirements.txt
```

Then run(on nnabla directory):

```
py.test python/test
```
