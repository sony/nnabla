.. _pip-installation-cuda:

NNabla CUDA extension package installation using PIP
====================================================

Note: please refer to the :ref:`pip_os_specific` for the OS specific dependencies setup.

By installing the NNabla CUDA extension package ``nnabla-ext-cuda``, you can accelerate the computation by NVidia CUDA GPU (CUDA must be setup on your environment accordingly).

Several pip packages of NNabla CUDA extension are provided for each CUDA version and its corresponding CUDNN version as following.

.. _cuda-cudnn-compatibility:

CUDA vs CUDNN Compatibility
---------------------------

================== ============ =====================
Package name       CUDA version CUDNN version
================== ============ =====================
nnabla-ext-cuda80  8.0          7.1(Linux & Win)
nnabla-ext-cuda90  9.0          7.6(Linux & Win)
nnabla-ext-cuda100 10.0         7.6(Linux & Win)
nnabla-ext-cuda101 10.1         7.6(Linux & Win)
================== ============ =====================

The latest CUDA version is always preferred if your GPU accepts.

Installation
------------

The following is an example of installing the extension for CUDA 10.1

.. code-block:: bash

	pip install nnabla-ext-cuda101

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

Multi-GPU wheel package is available only on ubuntu16.04 and python3.5+.

.. _cuda-cudnn-compatibility:

CUDA vs CUDNN Compatibility
---------------------------

================================= ============ =============
Package name                      CUDA version CUDNN version
================================= ============ =============
nnabla-ext-cuda90_nccl2_ubuntu16  9.0          7.6
nnabla-ext-cuda100_nccl2_ubuntu16 10.0         7.6
nnabla-ext-cuda100_nccl2_ubuntu18 10.0         7.6
nnabla-ext-cuda101_nccl2_ubuntu16 10.1         7.6
nnabla-ext-cuda101_nccl2_ubuntu18 10.1         7.6
================================= ============ =============

You can install as the following.

.. code-block:: bash

  pip install nnabla-ubuntu16
  pip install nnabla-ext-cuda101-nccl2-ubuntu16


If you already installed NNabla, uninstall all of it, or start from a clean environment which you create using Anaconda, virtualenv, or pyenv.


You should also install OpenMPI,

.. code-block:: bash

  apt-get install libopenmpi-dev

and NCCL in addition to CUDA and CuDNN.


FAQ
---

Q. How do I install CUDA?
^^^^^^^^^^^^^^^^^^^^^^^^^

NNabla CUDA extension requires both CUDA toolkit and CUDNN library. You should select a proper CUDA version according to your CUDA device capability. See `the official installation guide <https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html>`_. NNabla supports CUDA versions later than 8.0. See :ref:`the table <cuda-cudnn-compatibility>` for the CUDNN compatibility with the specific CUDA versions.


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
  Please install CUDA version 10.1.
    and CUDNN version 7.6
    Or install correct nnabla-ext-cuda for installed version of CUDA/CUDNN.
