.. _function:

=========
Functions
=========

All NNabla functions are derived from the :class:`nnabla.function.Function` class.

Function
========

.. automodule:: nnabla.function
.. autoclass:: Function
   :members:

.. autoclass:: PythonFunction
   :members:

   .. automethod:: __init__

.. _functions:

List of Functions
=================

.. automodule:: nnabla.functions

The :mod:`nnabla.functions` module provides various types of functions listed below.
These functions takes input :class:`nnabla.Variable` (s) as its leading argument(s), followed by options
specific to each function.

.. note::

    The functions can also take :obj:`~nnabla.NdArray` (s) as inputs instead
    of :obj:`~nnabla.Variable` (s). It will execute the function operation immediately,
    and returns :obj:`~nnabla.NdArray` (s) as output(s) holding output values of the
    operation. We call this "Imperative Mode" (NdArray + Functions).


Neural Network Layers
---------------------

.. autofunction:: affine
.. autofunction:: convolution
.. autofunction:: depthwise_convolution
.. autofunction:: deconvolution
.. autofunction:: depthwise_deconvolution
.. autofunction:: adaptive_separable_convolution
.. TODO: List fused_convolution once it becomes useful.
.. autofunction:: max_pooling
.. autofunction:: average_pooling
.. autofunction:: global_average_pooling
.. autofunction:: sum_pooling
.. autofunction:: unpooling
.. autofunction:: embed
.. autofunction:: rnn
.. autofunction:: lstm
.. autofunction:: gru
.. autofunction:: multi_head_attention
.. autofunction:: patch_correlation


Neural Network Activation
-------------------------

.. autofunction:: sigmoid
.. autofunction:: swish
.. autofunction:: tanh
.. autofunction:: relu
.. autofunction:: softmax
.. autofunction:: log_softmax
.. autofunction:: elu
.. autofunction:: selu
.. autofunction:: crelu
.. autofunction:: celu
.. autofunction:: gelu
.. autofunction:: prelu
.. autofunction:: leaky_relu
.. autofunction:: relu6
.. autofunction:: hard_sigmoid
.. autofunction:: hard_tanh
.. autofunction:: log_sigmoid
.. autofunction:: softplus
.. autofunction:: softsign
.. autofunction:: tanh_shrink
.. autofunction:: sinc


Normalization
-------------

.. autofunction:: batch_normalization
.. autofunction:: fused_batch_normalization
.. autofunction:: sync_batch_normalization
.. autofunction:: mean_subtraction
.. autofunction:: clip_by_value
.. autofunction:: clip_grad_by_value
.. autofunction:: clip_by_norm
.. autofunction:: clip_grad_by_norm
.. autofunction:: layer_normalization
.. autofunction:: instance_normalization
.. autofunction:: group_normalization
.. autofunction:: weight_standardization


Reduction
---------

.. autofunction:: sum
.. autofunction:: mean
.. autofunction:: max
.. autofunction:: min
.. autofunction:: prod
.. autofunction:: reduce_sum
.. autofunction:: reduce_mean


Arithmetic
----------

.. autofunction:: add2
.. autofunction:: add_n
.. autofunction:: sub2
.. autofunction:: mul2
.. autofunction:: mul_n
.. autofunction:: div2
.. autofunction:: pow2
.. autofunction:: add_scalar
.. autofunction:: mul_scalar
.. autofunction:: pow_scalar
.. autofunction:: r_sub_scalar
.. autofunction:: r_div_scalar
.. autofunction:: r_pow_scalar

Logical
-------

.. autofunction:: equal
.. autofunction:: equal_scalar
.. autofunction:: greater
.. autofunction:: greater_equal
.. autofunction:: greater_equal_scalar
.. autofunction:: greater_scalar
.. autofunction:: less
.. autofunction:: less_equal
.. autofunction:: less_equal_scalar
.. autofunction:: less_scalar
.. autofunction:: logical_and
.. autofunction:: logical_and_scalar
.. autofunction:: logical_not
.. autofunction:: logical_or
.. autofunction:: logical_or_scalar
.. autofunction:: logical_xor
.. autofunction:: logical_xor_scalar
.. autofunction:: not_equal
.. autofunction:: not_equal_scalar
.. autofunction:: sign
.. autofunction:: minimum2
.. autofunction:: maximum2
.. autofunction:: minimum_scalar
.. autofunction:: maximum_scalar
.. autofunction:: isnan
.. autofunction:: isinf
.. autofunction:: reset_nan
.. autofunction:: reset_inf
.. autofunction:: where


Math
----

.. autofunction:: constant
.. autofunction:: arange
.. autofunction:: abs
.. autofunction:: exp
.. autofunction:: log
.. autofunction:: round
.. autofunction:: ceil
.. autofunction:: floor
.. autofunction:: identity
.. autofunction:: matrix_diag
.. autofunction:: matrix_diag_part
.. autofunction:: batch_matmul
.. autofunction:: sin
.. autofunction:: cos
.. autofunction:: tan
.. autofunction:: sinh
.. autofunction:: cosh
.. autofunction:: tanh
.. autofunction:: asin
.. autofunction:: acos
.. autofunction:: atan
.. autofunction:: atan2
.. autofunction:: asinh
.. autofunction:: acosh
.. autofunction:: atanh


Array Manipulation
------------------

.. autofunction:: concatenate
.. autofunction:: split
.. autofunction:: stack
.. autofunction:: slice
.. autofunction:: gather_nd
.. autofunction:: scatter_nd
.. autofunction:: pad
.. autofunction:: transpose
.. autofunction:: broadcast
.. autofunction:: broadcast_to
.. autofunction:: tile
.. autofunction:: flip
.. autofunction:: shift
.. autofunction:: sort
.. autofunction:: reshape
.. autofunction:: one_hot
.. autofunction:: batch_inv
.. autofunction:: batch_det
.. autofunction:: assign
.. autofunction:: top_k_data
.. autofunction:: top_k_grad


Stochasticity
-------------

.. autofunction:: rand
.. autofunction:: randint
.. autofunction:: randn
.. autofunction:: rand_binomial
.. autofunction:: rand_beta
.. autofunction:: rand_gamma
.. autofunction:: dropout
.. autofunction:: random_choice
.. autofunction:: random_crop
.. autofunction:: random_erase
.. autofunction:: random_flip
.. autofunction:: random_shift
.. autofunction:: image_augmentation


Loss Functions
--------------

.. autofunction:: sigmoid_cross_entropy
.. autofunction:: binary_cross_entropy
.. autofunction:: softmax_cross_entropy
.. autofunction:: categorical_cross_entropy
.. autofunction:: squared_error
.. autofunction:: absolute_error
.. autofunction:: huber_loss
.. autofunction:: epsilon_insensitive_loss
.. autofunction:: kl_multinomial

Signal Processing
-----------------

.. autofunction:: interpolate
.. autofunction:: fft
.. autofunction:: ifft
.. autofunction:: stft
.. autofunction:: istft

Geometric Neural Network Layers
-------------------------------
.. autofunction:: affine_grid
.. autofunction:: warp_by_grid
.. autofunction:: warp_by_flow

Quantized Neural Network Layers
----------------------------------

.. autofunction:: binary_sigmoid
.. autofunction:: binary_tanh
.. autofunction:: binary_connect_affine
.. autofunction:: binary_connect_convolution
.. autofunction:: binary_weight_affine
.. autofunction:: binary_weight_convolution
.. autofunction:: fixed_point_quantize
.. autofunction:: min_max_quantize
.. autofunction:: pow2_quantize
.. autofunction:: prune
.. autofunction:: inq_affine
.. autofunction:: inq_convolution
			  
   
Unsupported, Special Use
------------------------

.. autofunction:: vat_noise
.. autofunction:: unlink
.. autofunction:: sink
.. autofunction:: confusion_matrix


Image Object Detection
----------------------

.. autofunction:: nms_detection2d


Validation
----------

.. autofunction:: top_n_error
.. autofunction:: binary_error
