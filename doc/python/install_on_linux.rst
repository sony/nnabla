Installation on Linux
=====================

.. contents::
   :local:
   :depth: 1


Prerequisites
-------------

This installation instruction describes how to install NNabla using pip
on almost any Linux 64-bit systems.

The supported Python versions for provided binary packages are 2.7, 3.5  3.6. It is recommended to use `Miniconda <https://conda.io/miniconda.html>`_ as a Python distribution. The following is a simple procedure to install Miniconda Python.

.. code-block:: shell

    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p {installation path e.g. ~/miniconda}
    # You have to set an environment variable PATH accordingly
    # to enable the installed ``Python`` and the ``conda`` system.
    echo 'export PATH=<installation path>:$PATH' > ~/.bashrc
    # Restart your bash or source ~/.bashrc

    # Switch the default Python version
    conda install -y python={version number e.g. 3.6}

Installation
------------

See :ref:`pip-installation-workflow`.

FAQ
---

Q. I use Anaconda, and the installation fails.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use libgcc 5 and numpy 1.13.0 or the greater, and note that `numba` depends on the older `numpy` so please uninstall `numba` first (The following is for Python2).

.. code-block:: bash

		conda create -n py2 python=2.7 anaconda  # if necessary
		source activate py2
		conda install libgcc
		conda install -c anaconda numpy=1.13.0

Then, you can follow the usual installation workflow.

Q. I want to use another linux distribution.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We actually tested other linux distributions and versions; Ubuntu 14.04, CentOS 6.9, 7.3, Fedora 23, 25, 26, and RHEL 7.3 on various environments; Baremetal server, AWS instance, and/or Docker machine. Thus, you can install in almost the same way described here. The details of how-to-install for each are coming soon.
