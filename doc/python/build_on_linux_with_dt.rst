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
  * NCCL
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

In order to use the distributed training, the only difference, when building, is 
the procedure described here. 

Download `nccl <https://developer.nvidia.com/nccl/nccl-download>`_ according to your environemnt,
then install it manually in case of ubuntu16.04, 

.. code-block:: shell

	sudo dpkg -i nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb
	sudo update
	sudo apt-get install libnccl2 libnccl-dev 


For developer, if you want to use another nccl not publicly distributed, 
specify **NCCL_HOME** environment variable as the folloing.

.. code-block:: shell

	export NCCL_HOME=${path}/build
	
Here, we assume the directry structure,  

* ${path}/build/include
* ${path}/build/lib

Distributed training also depends on MPI, so install it as follows,

.. code-block:: shell

	sudo apt-get install libopenmpi-dev
	
Follow :ref:`linux-build-and-install-cuda/cudnn-extension` but 
with **WITH_NCCL**, when you do `cmake` like 

.. code-block:: shell

	cmake -DNNABLA_DIR=../../nnabla -DCPPLIB_LIBRARY=../../nnabla/build/lib/libnnabla.so -D WITH_NCCL=ON ../                                                            

You can see nccl and mpi includes and dependencies,   

.. code-block:: shell

	...

	CUDA libs: /usr/local/cuda-8.0/lib64/libcudart.so;/usr/local/cuda-8.0/lib64/libcublas.so;/usr/local/cuda-8.0/lib64/libcurand.so;/usr/lib/x86_64-linux-gnu/libnccl.so;/usr/lib/openmpi/lib/libmpi_cxx.so;/usr/lib/openmpi/lib/libmpi.so;/usr/local/cuda/lib64/libcudnn.so
	CUDA includes: /usr/local/cuda-8.0/include;/usr/lib/openmpi/include/openmpi/opal/mca/event/libevent2021/libevent;/usr/lib/openmpi/include/openmpi/opal/mca/event/libevent2021/libevent/include;/usr/lib/openmpi/include;/usr/lib/openmpi/include/openmpi;/usr/local/cuda-8.0/include
	...


Unit test
^^^^^^^^^

Follow CUDA/cuDNN test in :ref:`linux-unit-test`. Now you could see the communicater 
test passed.

.. code-block:: shell

	...
	...
	communicator/test_data_parallel_communicator.py::test_data_parallel_communicator PASSED
	...


Now you can use **Data Parallel Distributed Training** using multiple GPUs and multiple nodes, please
go to CIFAR-10 example for how to use it.


