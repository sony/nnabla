Installation on Windows
-----------------------

.. contents::
   :local:
   :depth: 1


Prerequisites
^^^^^^^^^^^^^

We tested on Windows8.1 64bit and Windows10 64bit.

The following software are required for installation:

* Required software.

  * Python>=3.7: PIP
  * Microsoft Visual C++ 2015 Redistributable

* Recommended.

  * CUDA Toolkit and cuDNN (if you have CUDA GPUs).


Setup environment
^^^^^^^^^^^^^^^^^

Python
""""""

In this instruction, we use `Miniconda <https://conda.io/miniconda.html>`_.

Get and install the windows binary from `here <https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe>`_


And then install required packages from command prompt.

.. code-block:: doscon

    > conda install scipy scikit-image ipython


If your network is using proxy and setup fails, configure proxy server with environment variable and try install again.

.. code-block:: doscon

    > SET HTTP_PROXY=http://(enter the address of the http proxy server here)
    > SET HTTPS_PROXY=https://(enter the address of the https proxy server here)


Microsoft Visual C++ 2015 Redistributable
"""""""""""""""""""""""""""""""""""""""""

Get and install from `here <https://www.microsoft.com/en-us/download/details.aspx?id=52685>`_


CUDA and cuDNN library
""""""""""""""""""""""

If you are using a NVIDIA GPU, execution speed will be drastically improved by installing the following software.

`CUDA Toolkit <https://developer.nvidia.com/cuda-downloads>`_

`cuDNN <https://developer.nvidia.com/cudnn>`_

To install cuDNN, copy bin, include and lib to C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v{CUDA_VERSION}

See :ref:`a list of compatible cuDNN versions of CUDA extension packages <cuda-cudnn-compatibility>`.


Install
^^^^^^^

See :ref:`pip-installation-workflow`.

FAQ
^^^

Q. Scikit-image installation takes a long time.
"""""""""""""""""""""""""""""""""""""""""""""""

Depending on the environment, it will take a long time.  Please wait.

Q. Failed to install Scipy during installation.
"""""""""""""""""""""""""""""""""""""""""""""""

Please install scipy using "conda install" before "pip install nnabla".

