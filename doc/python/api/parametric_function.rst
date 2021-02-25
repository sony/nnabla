Parametric Functions
====================

In NNabla, trainable models are created by composing functions that have optimizable parameters.
These functions are called parametric functions. 
Parametric functions are provided by :mod:`nnabla.parametric_functions`.


See also:
    `Python API Tutorial
    <http://nnabla.readthedocs.io/en/latest/python/tutorial/python_api.html>`_.



.. _parameter:

Parameter Management API
------------------------

.. automodule:: nnabla.parameter

The parameters registered by :ref:`parametric-functions`
can be managed using APIs listed in this section.


.. autofunction:: parameter_scope
.. autofunction:: get_current_parameter_scope
.. autofunction:: get_parameters
.. autofunction:: clear_parameters
.. autofunction:: save_parameters
.. autofunction:: load_parameters
.. autofunction:: get_parameter_or_create

.. _parametric-functions:

List of Parametric Functions
----------------------------

.. automodule:: nnabla.parametric_functions

Parametric functions are provided by :py:mod:`nnabla.parametric_functions` , as listed below.
Like functions listed in :ref:`functions`, they take :obj:`~nnabla.Variable` (s) as
first argument(s) followed by options specific to a parametric function. In addition,
they register parameter :obj:`~nnabla.Variable` (s) into the parameter scope.

The parameter variables are registered with ``need_grad`` properties specific
to a parametric function. The variables with ``need_grad=False`` flag will not
be updated by gradient descent. Hence, backward computation is not executed for
those variables. ``False`` is usually specified when the parameters are updated
during foward pass and/or backward pass, e.g., batch normalization.

All parametric functions take an optional argument ``fix_parameters=False``.
By giving ``True``, the associated parameter variables are connected to a
computation graph with a property ``need_grad=False`` regardless properties
of the registered variables, then backward gradient
computation is not executed for those variables. This is useful when you create
a computation graph for evaluation purpose, fixing parameters partially in a
graph, and so on.

All parametric functions listed below are decorated with the following decorator.

.. autofunction:: parametric_function_api

See :ref:`parameter` to know how to query and manipulate registered variables.

Here is the list of parametric functions.

.. autofunction:: affine
.. autofunction:: convolution
.. autofunction:: depthwise_convolution
.. autofunction:: deconvolution
.. autofunction:: depthwise_deconvolution
.. autofunction:: deformable_convolution
.. autofunction:: batch_normalization
.. autofunction:: fused_batch_normalization
.. autofunction:: sync_batch_normalization
.. autofunction:: mean_subtraction
.. autofunction:: layer_normalization
.. autofunction:: instance_normalization
.. autofunction:: group_normalization

.. autofunction:: rnn
.. autofunction:: lstm
.. autofunction:: gru

.. autofunction:: embed
.. autofunction:: prelu

.. autofunction:: svd_affine
.. autofunction:: svd_convolution
.. autofunction:: cpd3_convolution
.. autofunction:: binary_connect_affine
.. autofunction:: binary_connect_convolution
.. autofunction:: binary_weight_affine
.. autofunction:: binary_weight_convolution
.. autofunction:: inq_affine
.. autofunction:: inq_convolution

.. autofunction:: fixed_point_quantized_affine
.. autofunction:: fixed_point_quantized_convolution
.. autofunction:: min_max_quantized_affine
.. autofunction:: min_max_quantized_convolution
.. autofunction:: pow2_quantized_affine
.. autofunction:: pow2_quantized_convolution
.. autofunction:: pruned_affine
.. autofunction:: pruned_convolution
.. autofunction:: min_max_quantize

.. autofunction:: lstm_cell

.. autoclass:: LSTMCell

    .. automethod:: __call__(x, w_init, b_init, fix_parameters)

.. autofunction:: spectral_norm
.. autofunction:: weight_normalization
.. autofunction:: multi_head_attention
.. autofunction:: transformer
.. autofunction:: transformer_encode
.. autofunction:: transformer_decode

Parameter Initializer
---------------------

Some of the parametric functions optionally takes parameter initializer
listed below.

.. automodule:: nnabla.initializer

.. autoclass:: BaseInitializer

    .. automethod:: __call__(shape)

.. autoclass:: ConstantInitializer
    :show-inheritance:
       
.. autoclass:: NormalInitializer
    :show-inheritance:

.. autoclass:: UniformInitializer
    :show-inheritance:

.. autoclass:: UniformIntInitializer
    :show-inheritance:

.. autoclass:: RangeInitializer
    :show-inheritance:

.. autoclass:: OrthogonalInitializer
    :show-inheritance:

.. autofunction:: calc_normal_std_he_forward
.. autofunction:: calc_normal_std_he_backward
.. autofunction:: calc_normal_std_glorot
.. autofunction:: calc_uniform_lim_glorot
