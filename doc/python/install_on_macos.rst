Installation on macOS
=====================

.. contents::
   :local:
   :depth: 1

NOTE: Our testing coverage in terms of environments and machines on macOS is very limited. Please submit an issue if you have any trouble.


Prerequisites
^^^^^^^^^^^^^

We test the installation on macOS Sierra.

The following software are required for installation:

* Python 2.7 or Python>=3.4 (We'd recommend you to setup Python using Anaconda or `Miniconda <https://conda.io/miniconda.html>`_).

  * pip (bundled in Conda Python)
  * wheel (bundled in Conda Python)
  * setuptools (bundled in Conda Python. You may need to upgrade the version of setuptools with ``pip install -U --no-deps setuptools``.)

Install
^^^^^^^

.. code-block:: shell

    pip install nnabla

NOTE: Binary package installation for the CUDA extension is not provided so far.

The following block naively checks if installation succeeds:

.. code-block:: shell

    python -c "import nnabla"


.. code-block::

    2017-09-21 15:01:43,035 [nnabla][INFO]: Initializing CPU extension...

Run an Example
^^^^^^^^^^^^^^

`Get<https://github.com/sony/nnabla-examples/archive/master.zip>`_ (and unzip) or clone `NNabla Examples repository <https://github.com/sony/nnabla-examples/>`_, and go to the MNIST folder.

.. code-block:: shell

    cd nnabla-examples/mnist-collection/

Then, run an MNIST classification:

.. code-block:: shell

    python classification.py
