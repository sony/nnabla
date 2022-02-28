Installation on Linux
=====================

.. contents::
   :local:
   :depth: 1


Prerequisites
-------------

This installation instruction describes how to install NNabla using pip
on almost any Linux 64-bit systems.

The supported Python versions for provided binary packages are 3.7, 3.8, 3.9. It is recommended to use `Miniconda <https://conda.io/miniconda.html>`_ as a Python distribution. The following is a simple procedure to install Miniconda Python.

.. code-block:: shell

    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p {installation path e.g. ~/miniconda}
    # You have to set an environment variable PATH accordingly
    # to enable the installed ``Python`` and the ``conda`` system.
    echo 'export PATH=<installation path>/bin:$PATH' >> ~/.bashrc
    # Restart your bash or source ~/.bashrc

    # Switch the default Python version
    conda install -y python={version number e.g. 3.7}

Installation
------------

See :ref:`pip-installation-workflow`.

FAQ
---

Q. I want to use another linux distribution.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We actually tested other linux distributions and versions; Ubuntu 16.04, 18.04, 20.04, CentOS 7, 8 on various environments; Baremetal server, AWS instance, and/or Docker machine. Thus, you can install in almost the same way described here. The details of how-to-install for each are coming soon.
