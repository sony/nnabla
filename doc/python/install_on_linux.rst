Installation on Linux
=====================

.. contents::
   :local:
   :depth: 1


Prerequisites
-------------

This installation instruction describes how to install NNabla using pip
on Ubuntu 16.04 (64bit).

* Required software.

  * Python 2.7 or Python>=3.4: PIP

* Recommended software (for NVIDIA GPU users).

  * CUDA Toolkit 8.0 / cuDNN 6.0

Note: Although this provides the instruction only on Ubuntu 16.04,
you can install NNabla using pip on many Linux with little extra dependencies installed.

Installation
------------

Install NNabla package via pip:

.. code-block:: bash

	sudo pip install -U nnabla

Then, check if it works by running:

.. code-block:: bash
	
	 python -c "import nnabla"

.. code-block:: bash

	 2017-06-22 08:47:27,023 [nnabla][INFO]: Initializing CPU extension...


If you are GPU user, follow the following instruction.
Before installing NNabla extension, make sure that
you have a machine/env CUDA and cuDNN are installed (See :ref:`install-cuda8-ubuntu16`).

Then,

.. code-block:: bash

	sudo pip install -U nnabla_ext_cuda

, and check if all works.

.. code-block:: bash
	
	 python -c "import nnabla_ext.cuda, nnabla_ext.cudnn"

.. code-block:: bash

	 2017-06-22 08:49:59,493 [nnabla][INFO]: Initializing CPU extension...
	 2017-06-22 08:49:59,699 [nnabla][INFO]: Initializing CUDA extension...
	 2017-06-22 08:50:02,684 [nnabla][INFO]: Initializing cuDNN extension...


Run an Example
--------------

`Get<https://github.com/sony/nnabla-examples/archive/master.zip>`_ (and unzip) or clone `NNabla Examples repository <https://github.com/sony/nnabla-examples/>`_, and go to the MNIST folder.

.. code-block:: shell

    cd nnabla-examples/mnist-collection/


Run MNIST classification.

.. code-block:: shell

    python classification.py


Run MNIST classification with CUDA/cuDNN.

.. code-block:: shell

    python classification.py -c cudnn



FAQ
---

.. _install-cuda8-ubuntu16:

Q. How do I install CUDA?
^^^^^^^^^^^^^^^^^^^^^^^^^

Install CUDA (CUDA 8.0)
""""""""""""""""""""""""

.. code-block:: bash

	wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
	sudo dpkg -i cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
	sudo apt-get update
	sudo apt install -y cuda-8.0

Install cuDNN (cuDNN version 6.0)
""""""""""""""""""""""""""""""""

Download cuDNN from `this page <https://developer.nvidia.com/cudnn>`_, then 

.. code-block:: bash

	tar zxvf cudnn-8.0-linux-x64-v6.0.tgz  # here, cudnn version is 6.0 for CUDA version 8.0
	sudo cp cuda/include/cudnn.h /usr/local/cuda/include/
	sudo cp -P cuda/lib64/* /usr/local/cuda/lib64/


Q. I use Anaconda, and the installation fails.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use libgcc 5 and numpy 1.13.0 or the greater, and note that `numba` depends on the older `numpy` so please uninstall `numba` first (The following is for Python2).

.. code-block:: bash

		conda create -n py2 python=2.7 anaconda  # if necessary
		source activate py2
		conda install libgcc
		conda install -c anaconda numpy=1.13.0

Then, you can follow the usual installation workflow.


Q. I don't have CUDNN6 in my environment.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When you got the error,

.. code-block:: text

	 ImportError: libcudnn.so.6: cannot open shared object file: No such file or directory

Please download cuDNN 6 for CUDA 8, put it in `/usr/local/cuda/lib/` or `/usr/local/cuda/lib64/` as the usual workflow, or set `LD_LIBRARY_PATH` as the following,

.. code-block:: bash
								
	 tar zxvf cudnn-8.0-linux-x64-v6.0.tgz
	 export LD_LIBRARY_PATH=$(pwd)/cuda/lib64:$LD_LIBRARY_PATH

Q. I do not have the root privilege.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you do not have the root privilege, please use virtualenv or Anaconda. After you downloaded cuDNN v6, do the following.

.. code-block:: bash
	tar zxvf cudnn-8.0-linux-x64-v6.0.tgz
	export LD_LIBRARY_PATH=$(pwd)/cuda/lib64:$LD_LIBRARY_PATH


Q. I want to use another linux distribution.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We actually tested other linux distributions and versions; Ubuntu 14.04, CentOS 6.9, 7.3, Fedora 23, 25, 26, and RHEL 7.3 on various environments; Baremetal server, AWS instance, and/or Docker machine. Thus, you can install in almost the same way described here. The details of how-to-install for each are comming soon.
