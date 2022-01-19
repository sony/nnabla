.. _pip-installation-workflow:

NNabla package installation using PIP
=====================================

Note: please refer to the :ref:`pip_os_specific` for the OS specific dependencies setup.

Install NNabla package via pip:

.. code-block:: bash

	pip install nnabla

**Note**: If you want to make sure the latest version will be installed, try to uninstall previously installed one with ``pip uninstall -y nnabla`` beforehand.

Then, check if it works by running:

.. code-block:: bash

        python -c "import nnabla"

.. code-block:: bash

   2018-06-26 15:20:16,759 [nnabla][INFO]: Initializing CPU extension...



NNabla CUDA extension package installation
------------------------------------------

See :ref:`pip-installation-cuda`.

Run an Example
--------------

Get `the examples <https://github.com/sony/nnabla-examples/archive/master.zip>`_ (, and unzip) or clone `NNabla Examples repository <https://github.com/sony/nnabla-examples/>`_, and go to the MNIST folder.

.. code-block:: shell

    cd nnabla-examples/image-classification/mnist-collection/


Run MNIST classification.

.. code-block:: shell

    python classification.py


Run MNIST classification with CUDA/cuDNN.

.. code-block:: shell

    python classification.py -c cudnn


