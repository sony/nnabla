Out-of-core execution
=====================

The ``nnabla.lms`` package provides APIs that allow users to execute large-scale networks than allotted GPU memory by utilizing out-of-core algorithm.
`Out-of-core algorithm <https://en.wikipedia.org/wiki/External_memory_algorithm>`_, or external memory algorithm, is an algorithm that enables processing data that are too large to fit into a main memory at once.

SwapInOutScheduler
------------------
.. autoclass:: nnabla.lms.SwapInOutScheduler
    :members:
