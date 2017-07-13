Build on Linux with Distributed Training 
----------------------------------------

.. contents::
   :local:
   :depth: 1

Prerequisites
^^^^^^^^^^^^^

Our build system requires:

* Required.

  * `CMake <https://cmake.org/>`_ (>=3.1)
  * Python 2.7: Devel, PIP
  * make, gcc, g++
  * CUDA Toolkit 8.0 / cuDNN 6.0 (to build CUDA/cuDNN extension for NVIDIA GPU)
  * Multiple GPUs
  * NCCL v1

Setup build environment
^^^^^^^^^^^^^^^^^^^^^^^

Follow :ref:`linux-setup-build-environment`.

Build and installation
^^^^^^^^^^^^^^^^^^^^^^

Build and install
"""""""""""""""""

Follow :ref:`linux-build-and-install`.

Build and install CUDA/cuDNN extension and NCCL
"""""""""""""""""""""""""""""""""""""""""""""""

In order to use Distributed Training, the only difference, when building, is 
the procedure described here. 

Download `nccl <https://github.com/NVIDIA/nccl>`_, build it, and set **NCCL_HOME** 
environment variable to enable to use NCCL v1 as the follows, 

.. code-block:: shell

	wget https://github.com/NVIDIA/nccl/archive/master.zip
	unzip master.zip
	cd nccl-master
	make -j 16 lib
	cd .. 
	export NCCL_HOME=$(pwd)
	

Distributed Training also depends on MPI, so install it as follows,

.. code-block:: shell

	sudo apt-get install libopenmpi-dev
	
then, set **MPI_HOME** like

.. code-block:: shell
  
  export MPI_HOME=/usr/lib/openmpi  # tipically here.
  
Note that **NCCL_HOME** and **MPI_HOME** is only used for building CUDA extension.

Follow :ref:`linux-build-and-install-cuda/cudnn-extension`, but when you do 
`cmake ../` you could see logs like, 

.. code-block:: shell

	...
	...
	CUDA libs: /usr/local/cuda/lib64/libcudart.so;/usr/local/cuda/lib64/libcublas.so;/usr/local/cuda/lib64/libcurand.so;/home/kzky/git/nccl/build/lib/libnccl.so;/usr/local/cuda/lib64/libcudnn.so
	CUDA includes: /usr/local/cuda/include;/home/kzky/git/nccl/build/include;/usr/local/cuda/include
	...

It shows NCCL include directory and library.

.. note::

	When we change terminals or re-login, set **NCC_HOME** again or 
	**LD_LIBRARY_PATH** like
	
.. code-block:: shell
	
	export LD_LIBRARY_PATH=${NCCL_HOME}/build/lib


Unit test
^^^^^^^^^

Follow CUDA/cuDNN test in :ref:`linux-unit-test`. Now you could see the communicater 
test passed.

.. code-block:: shell

	...
	...
	communicator/test_data_parallel_communicator.py::test_data_parallel_communicator PASSED
	...


You can use **Data Parallel Distributed Training** using multiple GPUs, please
go to CIFAR-10 example for how to use it.


