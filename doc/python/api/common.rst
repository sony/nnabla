Common
======

.. automodule:: nnabla

Logger
------

.. automodule:: nnabla.logger

.. autoclass:: logger

.. automodule:: nnabla._init

.. .. autofunction:: prefer_cached_array
.. .. autofunction:: reset_array_preference
.. .. autofunction:: array_classes
.. .. autofunction:: add_available_context
.. .. autofunction:: available_context

Auto-forward mode
-----------------

NNabla provides the dynamic computation graph feature, which enables automatic forward propagation during graph construction. This can be enabled using the :meth:`set_auto_forward` function. Backpropagation shall be manually executed on the dynamically constructed graph.

.. automodule:: nnabla
		
.. autofunction:: auto_forward
.. autofunction:: set_auto_forward
.. autofunction:: get_auto_forward


.. _context:

Context
-------

.. autoclass:: Context

.. _context-specifier:

Context Specifier API
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: context_scope
.. autofunction:: set_default_context
.. autofunction:: get_current_context
