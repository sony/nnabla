Build on Windows
----------------

.. contents::
   :local:
   :depth: 1


Prerequisites
^^^^^^^^^^^^^

We tested on Windows8.1 64bit and Windows10 64bit.

Our build system requires:

* Mandatory software.

  * Git
  * `CMake <https://cmake.org/>`_ (>=3.1)
  * Python 2.7: Devel, PIP
  * Visual C++

* Recommended

  * CUDA Toolkit 8.0 / cuDNN 6.0 (to build CUDA/cuDNN extension)
  * clang-format version 3.8
  * autopep8

.. _setup-build-env-windows:

Setup build environment
^^^^^^^^^^^^^^^^^^^^^^^

Git
"""

You can use your favorite git client, but in this instruction, we use official binary from `<https://git-scm.com/>`_

Get and install windows binary from `here <https://git-scm.com/download/win>`_.

After installed, make sure that following system environment variables are set.

============= ===========================================================================
 NAME          VALUE
============= ===========================================================================
PATH          `C:\\Program Files\\Git\\cmd`
============= ===========================================================================



Visual C++
""""""""""

You can use installed Visual C++ 2015 (Community, Professional or Enterprise)

Otherwise, you can use `Visual C++ 2015 Build Tools <http://landinghub.visualstudio.com/visual-cpp-build-tools>`_.

.. note:: You cannot use Visual Studio Express Edition. Because it does not include x86_64 native library.

After installed, make sure that following system environment variables are set.

============= ===========================================================================
 NAME          VALUE
============= ===========================================================================
VS90COMNTOOLS `C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\Common7\\Tools`
PATH          `C:\\Program Files (x86)\\MSBuild\\14.0\\Bin`
============= ===========================================================================


CMake
"""""
You can find `CMake <https://cmake.org/>`_ (>=3.1) to compilation.

Get and install windows binary from `here <https://cmake.org/files/v3.8/cmake-3.8.2-win64-x64.msi>`_

After installed, make sure that following system environment variables are set.

============= ===========================================================================
 NAME          VALUE
============= ===========================================================================
PATH          `C:\\Program Files\\CMake\\bin`
============= ===========================================================================

Protoc
""""""
You can find protocol buffer compiler from `Protocol Buffers <https://developers.google.com/protocol-buffers/>`_ .

Get Latest Release (3.3) from `here <https://github.com/google/protobuf/releases/download/v3.3.0/protoc-3.3.0-win32.zip>`_
and extract this into C:\\Utils\\protoc-3.3.0-win32\

After installed, make sure that following system environment variables are set.

============= ===========================================================================
 NAME          VALUE
============= ===========================================================================
PATH          `C:\\Utils\\protoc-3.3.0-win32\\bin`
============= ===========================================================================

Python2.7
"""""""""

In this instruction, we use `miniconda <https://conda.io/miniconda.html>`_.

Get and install windows binary from `here <https://repo.continuum.io/miniconda/Miniconda2-latest-Windows-x86_64.exe>`_


CUDA Toolkit 8.0 / cuDNN 6.0
""""""""""""""""""""""""""""

`CUDA Toolkit <https://developer.nvidia.com/cuda-downloads>`_

`cuDNN <https://developer.nvidia.com/cudnn>`_

To install cuDNN, copy bin, include and lib to C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0


Code Formatters
"""""""""""""""
You can find LLVM binaries at http://releases.llvm.org/download.html

Get and install windows binary from `here <http://releases.llvm.org/3.8.1/LLVM-3.8.1-win64.exe>`_


.. code-block:: shell

    > pip install autopep8


Build
^^^^^

In this instruction we use `conda environment <https://conda.io/docs/using/envs.html>`_
to keep your environment clean.


Create and activate environment.

.. code-block:: doscon

    > conda create -n nnabla
    > activate nnabla
    (nnabla) > conda install cython numpy boto3 h5py tqdm futures
    (nnabla) > conda install scipy scikit-image matplotlib ipython pywin32
    (nnabla) > conda install contextlib2
    (nnabla) > pip install protobuf
    (nnabla) > pip install autopep8


If your network is using proxy and setup fails, configure proxy server with environment variable and try install again.

.. code-block:: doscon

    (nnabla) > SET HTTP_PROXY=http://(enter the address of the http proxy server here)
    (nnabla) > SET HTTPS_PROXY=https://(enter the address of the https proxy server here)


Clone repository and prepare build directory.

.. code-block:: doscon

    (nnabla) > git clone https://github.com/sony/nnabla
    (nnabla) > cd nnabla
    (nnabla) > mkdir build
    (nnabla) > cd build

Build and install.

.. code-block:: doscon

    (nnabla) > cmake -G "Visual Studio 14 Win64" ..
    (nnabla) > msbuild ALL_BUILD.vcxproj /p:Configuration=Release
    (nnabla) > cd dist
    (nnabla) > pip install -U nnabla-<package version>-<package-arch>.whl


Build CUDA/cuDNN extension
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: doscon

    (nnabla) > git clone https://github.com/sony/nnabla-ext-cuda
    (nnabla) > cd nnabla-ext-cuda
    (nnabla) > mkdir build
    (nnabla) > cd build

Build and install.
You must add following options to cmake.

- -DNNABLA_DIR=<PATH to nnabla source directory>
- -DCPPLIB_LIBRARY=<PATH to libnnabla.so>

.. code-block:: doscon

    (nnabla) > cmake -G "Visual Studio 14 Win64" -DNNABLA_DIR=..\..\nnabla -DCPPLIB_LIBRARY=..\..\nnabla\build\bin\Release\nnabla.dll ..
    (nnabla) > msbuild ALL_BUILD.vcxproj /p:Configuration=Release
    (nnabla) > cd dist
    (nnabla) > pip install -U nnabla_ext_cuda-<package version>-<package-arch>.whl


Unit test
^^^^^^^^^

For unit testing, some additional requirements should be installed.

.. code-block:: doscon

    (nnabla) > pip install pytest

Then run:

.. code-block:: doscon

    (nnabla) > py.test nnabla\python\test


Deactivate and remove environment.

.. code-block:: doscon

    (nnabla) > deactivate
    > conda remove -n nnabla --all


FAQ
^^^

Q. Command line display becomes strange after executing 'sconda install scikit-image'.
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Restart command prompt, and continue the remaining steps.


Q. It runs on a compiled PC, but it does not work on another PC
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Confirm that the Microsoft Visual C++ 2015 Redistributable is installed on target PC.
If the installed GPU is different between build PC and target PC, try the following cmake option when you build nnabla-ext-cuda.

.. code-block:: doscon

    > cmake -G "Visual Studio 14 Win64" -D CUDA_SELECT_NVCC_ARCH_ARG:STRING="All" ..\

Q. Another version of Visual C++ was installed and python setup.py install failed.
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Uninstall another version of Visual C++ or move the location once to setup.

Q. I use WinPython, and test fails
""""""""""""""""""""""""""""""""""

Miniconda is recommended to build NNabla on Windows.

