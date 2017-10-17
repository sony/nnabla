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


.. autofunction:: parameter_scope(name)
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

All parametric functions listed below are decorated with the following decorator.

.. autofunction:: parametric_function_api

See :ref:`parameter` to know how to query and manipulate registered variables.

Here is the list of parametric functions.

.. autofunction:: affine
.. autofunction:: convolution
.. autofunction:: deconvolution
.. autofunction:: batch_normalization

.. autofunction:: embed
.. autofunction:: prelu

.. autofunction:: binary_connect_affine
.. autofunction:: binary_connect_convolution
.. autofunction:: binary_weight_affine
.. autofunction:: binary_weight_convolution
.. autofunction:: inq_affine
.. autofunction:: inq_convolution

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

.. autofunction:: calc_normal_std_he_forward
.. autofunction:: calc_normal_std_he_backward
.. autofunction:: calc_normal_std_glorot
.. autofunction:: calc_uniform_lim_glorot
