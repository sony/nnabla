# Build Python Package on Windows

## Prerequisites

We tested on Windows8.1 64bit and Windows10 64bit.

Our build system requires:

* Git
* [CMake](https://cmake.org/) (>=3.1)
* Python >= 3.6: Devel, PIP
* Visual C++

## Setup build environment

### Git

You can use your favorite git client, but in this instruction, we use official binary download [here](https://git-scm.com/).

Get and install windows binary from [here](https://git-scm.com/download/win).

After installed, make sure that following system environment variables are set.

* PATH: `C:\\Program Files\\Git\\cmd`

### Visual C++

You can use installed Visual C++ 2015 (Community, Professional or Enterprise)

Otherwise, you can use [Visual C++ 2015 Build Tools](http://landinghub.visualstudio.com/visual-cpp-build-tools>).

NOTE: You cannot use Visual Studio Express Edition. Because it does not include x86_64 native library.

After installed, make sure that following system environment variables are set.

* VS90COMNTOOLS: `C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\Common7\\Tools`
* PATH: `C:\\Program Files (x86)\\MSBuild\\14.0\\Bin`

### CMake

You can find [CMake](https://cmake.org/) (>=3.1) for compilation.

Get and install windows binary from [here](https://cmake.org/files/v3.8/cmake-3.8.2-win64-x64.msi>).

After installed, make sure that following system environment variables are set.

* PATH: `C:\\Program Files\\CMake\\bin`

### Protoc

You can find protocol buffer compiler from [Protocol Buffers](https://developers.google.com/protocol-buffers/).

Get Latest Release (3.3) from [here](https://github.com/google/protobuf/releases/download/v3.3.0/protoc-3.3.0-win32.zip).
and extract this into `C:\\Utils\\protoc-3.3.0-win32\\`

After installed, make sure that following system environment variables are set.

* PATH: `C:\\Utils\\protoc-3.3.0-win32\\bin`

### Python (3.5 or higher)

In this instruction, we use [Miniconda](https://conda.io/miniconda.html).

Follow the link and get the installer. If you already have Python on your system, it should also work, but we don't check it works on any Python distribution other than Miniconda.

## Build


In this instruction we use [Conda environment](https://conda.io/docs/using/envs.html) to keep your environment clean.

Create and activate environment.

```bat
> conda create -n nnabla
> activate nnabla
(nnabla) > conda install cython numpy boto3 h5py tqdm futures mako PyYAML
(nnabla) > conda install scipy scikit-image matplotlib ipython pywin32
(nnabla) > conda install contextlib2
(nnabla) > pip install protobuf
```


If your network is using proxy and setup fails, configure proxy server with environment variable and try install again.

```bat
(nnabla) > SET HTTP_PROXY=http://(enter the address of the http proxy server here)
(nnabla) > SET HTTPS_PROXY=https://(enter the address of the https proxy server here)
```

Clone repository and prepare build directory.

```bat
(nnabla) > git clone https://github.com/sony/nnabla
(nnabla) > cd nnabla
(nnabla) > mkdir build
(nnabla) > cd build
```

Build and install.

```bat
(nnabla) > cmake -G "Visual Studio 14 Win64" ..
(nnabla) > msbuild ALL_BUILD.vcxproj /p:Configuration=Release
(nnabla) > cd dist
(nnabla) > pip uninstall -y nnabla
(nnabla) > pip install nnabla-<package version>-<package-arch>.whl
```

## Unit test

For unit testing, some additional requirements should be installed.

```bat
(nnabla) > pip install pytest
```

Then run:

```bat
(nnabla) > py.test nnabla\python\test
```

Deactivate the conda env.

```
(nnabla) > deactivate
```

## FAQ

* Q. Command line display becomes strange after executing `conda install scikit-image`.
  * Restart command prompt, and continue the remaining steps.
* Q. The compiled library and executable run on a compiled PC, but it does not work on another PC.
  * Confirm that the Microsoft Visual C++ 2015 Redistributable is installed on target PC.

* Q. Another version of Visual C++ was installed and python setup.py install failed.
  * Uninstall another version of Visual C++ or move the location once to setup.
* Q. I use WinPython, and test fails
  * Miniconda is recommended to build NNabla on Windows.
