.. _cpp-lib-installation-windows:

Build C++ libraries on Windows
==============================

Dependencies
------------

* Required (See :ref:`setup-build-env-windows` for how to setup the build requirement.)
  * 64-bit Windows.
  * Python 2.7 or >=3.4, 3.5 or 3.6
  * Visual C++ 2015 or Build tools
  * `CMake <https://cmake.org/>`_ (>=3.1)

Build
-----

First, clone `nnabla <https://github.com/sony/nnabla>`_ and go into the root folder of it.
There are some additional depending libraries that are installed automatically by running the following batch script.

.. code-block:: doscon

    build-tools\msvc\setup_cpp_utils_deps.bat

This will setup the following depending libraries of the NNabla C++ utility

* LibArchive
* ZLib
* Protobuf

into the ``third_party`` folder (HDF5 is not supported on Windows so far), and genearte a batch script ``cmake_cpp_utils.bat``. You can  build the NNabla core library and the C++ utility library (``nnabla.dll``, ``nnabla_utils.dll`` and their ``.lib`` and ``.exp`` files) by running:

.. code-block:: doscon

    build-tools\msvc\cmake_cpp_utils.bat

Your library or executable with NNabla
--------------------------------------

To build your libraries or executables, you need:

* setting ``<nnabla root>\include`` folder as include path
* setting ``nnabla.lib`` and ``nnabla_utils.lib`` as libraries

At runtime, you will need the following dynamic link libraries located in a right path.

* ``nnabla.dll`
* ``nnabla_utils.dll``
* ``zlibwapi.dll``
* ``archive.dll``

Please find these by searching at NNabla root folder.
