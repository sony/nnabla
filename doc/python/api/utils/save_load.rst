.. _nnp-apis:

NNP save and load utilities
===========================

**IMPORTANT NOTICE**: To handle NNP file from Neural Network Console, if the network you want to save/load contains `LoopControl` functions `RepeatStart`_,  `RepeatEnd`_, `RecurrentInput`_, `RecurrentOutput`_ or `Delay`_, you must expand network with :any:`../../file_format_converter/file_format_converter`.

.. _RepeatStart: https://support.dl.sony.com/docs/layer_reference/#RepeatStart
.. _RepeatEnd: https://support.dl.sony.com/docs/layer_reference/#RepeatEnd
.. _RecurrentInput: https://support.dl.sony.com/docs/layer_reference/#RecurrentInput
.. _RecurrentOutput: https://support.dl.sony.com/docs/layer_reference/#RecurrentOutput
.. _Delay: https://support.dl.sony.com/docs/layer_reference/#Delay

.. autofunction:: nnabla.utils.save.save

.. autoclass:: nnabla.utils.nnp_graph.NnpLoader
    :members:

.. autoclass:: nnabla.utils.nnp_graph.NnpNetwork
    :members:

.. automodule:: nnabla.utils
