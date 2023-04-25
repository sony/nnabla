.. _pip-installation-cuda:

NNabla CUDA extension package installation using PIP
====================================================

Note: please refer to the :ref:`pip_os_specific` for the OS specific dependencies setup.

In addition to NNabla's requirements, CUDA extension requires CUDA setup has done on your system. If you don't have CUDA on your system, follow the procedure described below.

Download and install `CUDA toolkit <https://developer.nvidia.com/cuda-downloads>`_ and `cuDNN library(Registration required) <https://developer.nvidia.com/rdp/cudnn-download>`_ (both runtime library and development library). Please follow the instruction in the document provided by NVIDIA. Do NOT see any instruction provided by any third party. They are often incorrect or based on old instructions, that could destroy your system.

By installing the NNabla CUDA extension package ``nnabla-ext-cuda``, you can accelerate the computation by NVIDIA CUDA GPU (CUDA must be setup on your environment accordingly).

Several pip packages of NNabla CUDA extension are provided for each CUDA version and its corresponding cuDNN version as following.

.. _cuda-cudnn-compatibility:

CUDA vs cuDNN Compatibility
---------------------------

================== ============ =====================
Package name       CUDA version cuDNN version
================== ============ =====================
nnabla-ext-cuda110 11.0.3       8.0(Linux & Win)
nnabla-ext-cuda116 11.6.2       8.4(Linux & Win)
================== ============ =====================

The latest CUDA version is always preferred if your GPU accepts.

Currently, for each NNabla CUDA extension package, it may be not compatible with some specific GPUs.

After nnabla-ext-cuda package is installed, you can manually check whether your GPU is usable.
For example, you can check GPU with device_id 0 by:

.. code-block:: python

   import nnabla_ext.cudnn
   device_id = '0'
   nnabla_ext.cudnn.check_gpu(device_id)

Above code will run successfully if your GPU is usable, otherwise, an error will be reported.

nnabla-ext-cuda package will also try to check the compatibility of your GPUs automatically when you use 'cuda' or 'cudnn' extension.
By default, it will list and check all gpus in your machine. Error will be reported if there is incompatible card.

You can set environment variable 'AVAILABLE_GPU_NAMES' to tell it which GPU is usable, 'AVAILABLE_GPU_NAMES' is a white list, GPU in 'AVAILABLE_GPU_NAMES' will not cause error.
For example, if you think GeForce RTX 3070 and GeForce RTX 3090 are usable, you can set environment variable as following:

.. code-block:: bash

	export AVAILABLE_GPU_NAMES="GeForce RTX 3070,GeForce RTX 3090"


Installation
------------

The following is an example of installing the extension for CUDA 11.0.3

.. code-block:: bash

	pip install nnabla-ext-cuda110

and check if all works.

.. code-block:: bash

  python -c "import nnabla_ext.cuda, nnabla_ext.cudnn"

.. code-block:: bash

  2018-06-26 15:20:36,085 [nnabla][INFO]: Initializing CPU extension...
  2018-06-26 15:20:36,257 [nnabla][INFO]: Initializing CUDA extension...
  2018-06-26 15:20:36,257 [nnabla][INFO]: Initializing cuDNN extension...

**Note**: If you want to make sure the latest version will be installed, try to uninstall previously installed one with ``pip uninstall -y nnabla nnabla-ext-cuda110`` beforehand.


.. _pip-installation-distributed:

Installation with Multi-GPU supported
-------------------------------------

Multi-GPU wheel package is only available on python3.7+.

.. _cuda-cudnn-compatibility-multi-gpu:

CUDA vs cuDNN Compatibility
---------------------------

=================================== ============ =============
Package name                        CUDA version cuDNN version
=================================== ============ =============
nnabla-ext-cuda110                  11.0.3       8.0
nnabla-ext-cuda116                  11.6.2       8.4
=================================== ============ =============

You can install as the following.

.. code-block:: bash

  pip install nnabla
  pip install nnabla-ext-cuda110


If you already installed NNabla, uninstall all of it, or start from a clean environment which you create using Anaconda, venv.


You should also install OpenMPI and NCCL in addition to CUDA and CuDNN.

If you are using Ubuntu 20.04 and choose mpi4.0.3, you can install mpi with following command.

.. code-block:: bash

  sudo apt install -y --no-install-recommends openmpi-bin libopenmpi-dev

Otherwise, you must install a version openmpi by supported on ubuntu 20.04. (e.g. 3.1.6 or 4.1.3). In theory, all versions of openmpi are supported.

.. code-block:: bash

  MPIVER=3.1.6
  curl -O https://download.open-mpi.org/release/open-mpi/v${MPIVER%.*}/openmpi-${MPIVER}.tar.bz2
  tar xvf openmpi-${MPIVER}.tar.bz2
  cd openmpi-${MPIVER}
  ./configure --with-sge
  make
  sudo make install


FAQ
---

Q. How do I install CUDA?
^^^^^^^^^^^^^^^^^^^^^^^^^

NNabla CUDA extension requires both CUDA toolkit and cuDNN library. You should select a proper CUDA version according to your CUDA device capability. See `the official installation guide <https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html>`_. NNabla supports CUDA versions later than 8.0. See :ref:`the table <cuda-cudnn-compatibility>` for the cuDNN compatibility with the specific CUDA versions.


Q. How do I install NCCL
^^^^^^^^^^^^^^^^^^^^^^^^

Please visit `NCCL <https://developer.nvidia.com/nccl>`_, then follow the instruction.


Q. How do I check proper version of cuDNN
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Enter the following command:

.. code-block:: bash

  python -c "import nnabla_ext.cuda, nnabla_ext.cudnn"

If there is a version mismatch on your machine, you can see proper versions in the error message.
Following is a sample error message.

.. code-block:: bash

  [nnabla][INFO]: Initializing CPU extension...
  Please install CUDA version 11.0.3.
    and cuDNN version 8.0
    Or install correct nnabla-ext-cuda for installed version of CUDA/cuDNN.
