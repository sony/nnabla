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
  * OpenMPI

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
	
Note that **NCCL_HOME** is only used for building CUDA extension.


Distributed Training also depends on MPI, so install it as follows,

.. code-block:: shell

	sudo apt-get install libopenmpi-dev
	
Follow :ref:`linux-build-and-install-cuda/cudnn-extension` but 
with **WITH_NCCL**, when you do `cmake` like 

.. code-block:: shell

	cmake -D WITH_NCCL=ON ../                                                            

You can see nccl and mpi includes and dependencies,   

.. code-block:: shell

	...

	CUDA libs: /usr/local/cuda/lib64/libcudart.so;/usr/local/cuda/lib64/libcublas.so;/usr/local/cuda/lib64/libcurand.so;/home/kzky/git/nccl/build/lib/libnccl.so;/usr/lib/openmpi/lib/libmpi_cxx.so;/usr/lib/openmpi/lib/libmpi.so;/usr/local/cuda/lib64/libcudnn.so
	CUDA includes: /usr/local/cuda/include;/home/kzky/git/nccl/build/include;/usr/lib/openmpi/include/openmpi/opal/mca/event/libevent2021/libevent;/usr/lib/openmpi/include/openmpi/opal/mca/event/libevent2021/libevent/include;/usr/lib/openmpi/include;/usr/lib/openmpi/include/openmpi;/usr/local/cuda/include
	...


.. note::

	When we change terminals or re-login, set **NCCL_HOME** again or 
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


Now you can use **Data Parallel Distributed Training** using multiple GPUs, please
go to CIFAR-10 example for how to use it.


