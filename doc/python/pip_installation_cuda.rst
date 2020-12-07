.. _pip-installation-cuda:

NNabla CUDA extension package installation using PIP
====================================================

Note: please refer to the :ref:`pip_os_specific` for the OS specific dependencies setup.

By installing the NNabla CUDA extension package ``nnabla-ext-cuda``, you can accelerate the computation by NVIDIA CUDA GPU (CUDA must be setup on your environment accordingly).

Several pip packages of NNabla CUDA extension are provided for each CUDA version and its corresponding cuDNN version as following.

.. _cuda-cudnn-compatibility:

CUDA vs cuDNN Compatibility
---------------------------

================== ============ =====================
Package name       CUDA version cuDNN version
================== ============ =====================
nnabla-ext-cuda100 10.0         7.6(Linux & Win)
nnabla-ext-cuda102 10.2         7.6(Win)
nnabla-ext-cuda102 10.2         8.0(Linux)
nnabla-ext-cuda110 11.0         8.0(Linux)
================== ============ =====================

The latest CUDA version is always preferred if your GPU accepts.

Installation
------------

The following is an example of installing the extension for CUDA 10.2

.. code-block:: bash

	pip install nnabla-ext-cuda102

and check if all works.

.. code-block:: bash

  python -c "import nnabla_ext.cuda, nnabla_ext.cudnn"

.. code-block:: bash

  2018-06-26 15:20:36,085 [nnabla][INFO]: Initializing CPU extension...
  2018-06-26 15:20:36,257 [nnabla][INFO]: Initializing CUDA extension...
  2018-06-26 15:20:36,257 [nnabla][INFO]: Initializing cuDNN extension...

**Note**: If you want to make sure the latest version will be installed, try to uninstall previously installed one with ``pip uninstall -y nnabla nnabla-ext-cuda100`` beforehand.


.. _pip-installation-distributed:

Installation with Multi-GPU supported
-------------------------------------

Multi-GPU wheel package is only available on python3.6+.

.. _cuda-cudnn-compatibility:

CUDA vs cuDNN Compatibility
---------------------------

=================================== ============ =============
Package name                        CUDA version cuDNN version
=================================== ============ =============
nnabla-ext-cuda100-nccl2-mpi2-1-1  10.0         7.6
nnabla-ext-cuda100-nccl2-mpi3-1-6  10.0         7.6
nnabla-ext-cuda102-nccl2-mpi2-1-1  10.2         8.0
nnabla-ext-cuda102-nccl2-mpi3-1-6  10.2         8.0
nnabla-ext-cuda110-nccl2-mpi2-1-1  11.0         8.0
nnabla-ext-cuda110-nccl2-mpi3-1-6  11.0         8.0
=================================== ============ =============

You can install as the following.

.. code-block:: bash

  pip install nnabla
  pip install nnabla-ext-cuda100-nccl2-mpi2-1-1


If you already installed NNabla, uninstall all of it, or start from a clean environment which you create using Anaconda, venv.


You should also install OpenMPI and NCCL in addition to CUDA and CuDNN.

If you are using Ubuntu18.04 and choose mpi2.1.1, you can install mpi with following command.

.. code-block:: bash

  sudo apt install -y --no-install-recommends openmpi-bin libopenmpi-dev

Otherwise, you must install openmpi with following command.(MPIVER=3.1.6 or 2.1.1)

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
  Please install CUDA version 10.2.
    and cuDNN version 8.0
    Or install correct nnabla-ext-cuda for installed version of CUDA/cuDNN.
