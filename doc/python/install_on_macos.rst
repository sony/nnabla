Installation on macOS
=====================

.. contents::
   :local:
   :depth: 1

NOTE: Our testing coverage in terms of environments and machines on macOS is very limited. Please submit an issue if you face any issue.


Prerequisites
^^^^^^^^^^^^^

We test the installation on macOS 11.

The following software are required for installation:

* Python>=3.7 (You can also use another distribution. e.g. `pyenv <https://github.com/pyenv/pyenv>`_ or `Miniconda <https://conda.io/miniconda.html>`_).

  * pip
  * wheel (``pip install wheel``)
  * setuptools (You may need to upgrade the version of setuptools with ``pip install -U --no-deps setuptools``.)

Install
^^^^^^^

See :ref:`pip-installation-workflow` (note that the binary packages for the CUDA extension are not available for macOS. Please `build it from source <https://github.com/sony/nnabla/tree/master/doc/build/build.md>`_).
