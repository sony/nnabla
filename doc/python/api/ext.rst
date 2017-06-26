Extensions
==========

NNabla offers easy extensibility for developers to add new device extensions.
The NNabla Python package officially supports the ``cuda.cudnn`` extension, which
dramatically accelerates computation
by leveraging NVIDIA CUDA GPUs with cuDNN computation primitives.

You can manually import extensions by:

.. code-block:: python

   import nnabla_ext.cuda.cudnn

See :ref:`python-package-installation` to install the CUDA extension.

Implementing an extension
-------------------------

All extension modules must have the following functions.

.. py:function:: context(*kw)

    Returns a default context descriptor of the extension module.
    This method takes optional arguments depending on the extension.
    For example, in the ``cuda.cudnn`` extension, it takes the ``device_id`` as an ``int``
    to specify the GPU where computation runs on.

.. py:function:: synchronize(*kw)

    This method is used to synchronize the device execution stream with respect to the host thread.
    For example, in CUDA, the kernel execution is enqueued into a stream, and is
    executed asynchronously w.r.t. the host thread. This function is only valid in devices that use
    such features. In the CPU implementation, this method is implemented as dummy function,
    and therefore calls to this function are ignored.
    The function in the ``cuda.cudnn`` extension takes the ``device_id`` as an optional argument,
    which specifies the device you want to synchronize with.
