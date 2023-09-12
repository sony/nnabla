Installation on Windows
-----------------------

.. contents::
   :local:
   :depth: 1


Prerequisites
^^^^^^^^^^^^^

We tested on Windows10 64bit, python3.10.

The following software are required for installation:

* Required software.

  * Python>=3.8: PIP

* Recommended.

  * CUDA Toolkit and cuDNN (if you have CUDA GPUs).


Setup environment
^^^^^^^^^^^^^^^^^

Python
""""""
If you don't install python, you can refer to the link below.

`python <https://www.python.org/downloads/windows/>`_

If your network is using proxy and setup fails, configure proxy server with environment variable and try install again.

.. code-block:: doscon

    > set http_proxy=http://(enter the address of the http proxy server here)
    > set https_proxy=http://(enter the address of the https proxy server here)


CUDA and cuDNN library
""""""""""""""""""""""

If you are using a NVIDIA GPU, execution speed will be drastically improved by installing the following software.

For the versions of CUDA/cuDNN we support, see :ref:`the table <cuda-cudnn-compatibility>`.

`CUDA Toolkit <https://developer.nvidia.com/cuda-downloads>`_

`cuDNN <https://developer.nvidia.com/cudnn>`_

To install cuDNN, copy bin, include and lib to C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v{CUDA_VERSION}


Install
^^^^^^^

See :ref:`pip-installation-workflow`.

FAQ
^^^

Q. Scikit-image installation takes a long time.
"""""""""""""""""""""""""""""""""""""""""""""""

Depending on the environment, it will take a long time.  Please wait.

