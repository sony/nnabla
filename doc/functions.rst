Function Definitions
====================

Neural Network Layer
--------------------

Affine
^^^^^^

Affine layer, also called as the fully connected layer. It calculates:

.. math::
    {\mathbf y} = {\mathbf A} {\mathbf x} + {\mathbf b}.

where :math:`{\mathbf x}` is the input and :math:`{\mathbf y}` is the output. 

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - Input N-D array with shape (:math:`M_0 \times ... \times M_{B-1} \times D_B \times ... \times D_N`). Dimensions before and after base_axis are flattened as if it is a matrix.
     - 
   * - weight
     - Weight matrix with shape (:math:`(D_B \times ... \times D_N) \times L`)
     - Parameter
   * - bias
     - Bias vector (:math:`L`)
     - Optional Parameter

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - base_axis
     - int64
     - 1
     - Base axis of Affine operation. Dimensions up to base_axis is treated as sample dimension.


* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - :math:`(B + 1)`-D array. (:math:`M_0 \times ... \times M_{B-1} \times L`)
     - 

Convolution
^^^^^^^^^^^

N-D Convolution with bias.

See references for dilated convolution (a.k.a. atrous convolution).

References:

    * `Chen et al., DeepLab: Semantic Image Segmentation with Deep Convolutional
      Nets, Atrous Convolution, and Fully Connected CRFs.
      <https://arxiv.org/abs/1606.00915>`_

    * `Yu et al., Multi-Scale Context Aggregation by Dilated Convolutions.
      <https://arxiv.org/abs/1511.07122>`_

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - :math:`(B + 1 + N)`-D array (:math:`M_1 \times ... \times M_B \times C \times L_1 \times ... \times L_N`).
     - 
   * - weight
     - :math:`(2 + N)`-D array (:math:`C' \times C \times K_1 \times ... \times K_N`).
     - Parameter
   * - bias
     - Bias vector (:math:`C'`).
     - Optional Parameter

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - base_axis
     - int64
     - 1
     - base axis :math:`B`.
   * - pad
     - Shape
     - (0,) * (len(x.shape) - (base_axis+1))
     - Padding sizes for dimensions.
   * - stride
     - Shape
     - (1,) * (len(x.shape) - (base_axis+1))
     - Stride sizes for dimensions.
   * - dilation
     - Shape
     - (1,) * (len(x.shape) - (base_axis+1))
     - Dilation sizes for dimensions.
   * - group
     - int64
     - 1
     - Number of groups of channels. This makes the connection across channels sparser, by grouping connections along the mapping direction.


* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - :math:`(B + 1 + N)`-D array (:math:`M_1 \times ... \times M_B \times C' \times L'_1 \times ... \times L'_N`).
     - 

DepthwiseConvolution
^^^^^^^^^^^^^^^^^^^^

N-D Depthwise Convolution with bias.

References:

    * `F. Chollet: Chollet, Francois. "Xception: Deep Learning with Depthwise Separable Convolutions.
      <https://arxiv.org/abs/1610.02357>`_

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - :math:`(B + 1 + N)`-D array (:math:`M_1 \times ... \times M_B \times C \times L_1 \times ... \times L_N`).
     -
   * - weight
     - :math:`(1 + N)`-D array (:math:`C \times K_1 \times ... \times K_N`).
     - Parameter
   * - bias
     - Bias vector (:math:`C`).
     - Optional Parameter

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - base_axis
     - int64
     - 1
     - base axis :math:`B`.
   * - pad
     - Shape
     - (0,) * (len(x.shape) - (base_axis+1))
     - Padding sizes for dimensions.
   * - stride
     - Shape
     - (1,) * (len(x.shape) - (base_axis+1))
     - Stride sizes for dimensions.
   * - dilation
     - Shape
     - (1,) * (len(x.shape) - (base_axis+1))
     - Dilation sizes for dimensions.
   * - multiplier
     - int64
     - 1
     - Number of output feature maps per input feature map.

* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - :math:`(B + 1 + N)`-D array (:math:`M_1 \times ... \times M_B \times C \times L'_1 \times ... \times L'_N`).
     -

Deconvolution
^^^^^^^^^^^^^

N-D deconvolution, also known as transposed convolution, with bias operates backward convolution (derivative of the output w.r.t. the input) plus channel-wise learned bias.

The weights are specified in the same manner as :meth:`~nnabla.functions.convolution` , as if it was an ordinary convolution function.
The forward operation of :meth:`~nnabla.functions.deconvolution` will then be operationally equivalent to the backward pass of :meth:`~nnabla.functions.convolution` .
Therefore, the number of input channels (can be seen as output channels of forward convolution) is specified in the first dimension, and the number of the output channels divided by the number of groups is specified in the second dimension.

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - :math:`(B + 1 + N)`-D array (:math:`M_1 \times ... \times M_B \times C \times L_1 \times ... \times L_N`).
     - 
   * - weight
     - :math:`(2 + N)`-D array (:math:`C' \times C \times K_1 \times ... \times K_N`).
     - Parameter
   * - bias
     - Bias vector (:math:`C'`).
     - Optional Parameter

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - base_axis
     - int64
     - 1
     - base axis :math:`B`.
   * - pad
     - Shape
     - (0,) * (len(x.shape) - (base_axis+1))
     - Padding sizes for dimensions.
   * - stride
     - Shape
     - (1,) * (len(x.shape) - (base_axis+1))
     - Stride sizes for dimensions.
   * - dilation
     - Shape
     - (1,) * (len(x.shape) - (base_axis+1))
     - Dilation sizes for dimensions.
   * - group
     - int64
     - 1
     - Number of groups of channels. This makes the connection across channels sparser, by grouping connections along the mapping direction.

* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - :math:`(B + 1 + N)`-D array (:math:`M_1 \times ... \times M_B \times C' \times L'_1 \times ... \times L'_N`).
     - 

DepthwiseDeconvolution
^^^^^^^^^^^^^^^^^^^^^^

Depthwise deconvolution computes the transposed depthwise convolution with bias for one-dimensional and two-dimensional input data.

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - :math:`(B + 1 + N)`-D array (:math:`M_1 \times ... \times M_B \times C \times L_1 \times ... \times L_N`).
     -
   * - weight
     - :math:`(1 + N)`-D array (:math:`C \times K_1 \times ... \times K_N`).
     - Parameter
   * - bias
     - Bias vector (:math:`C`).
     - Optional Parameter

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - base_axis
     - int64
     - 1
     - base axis :math:`B`.
   * - pad
     - Shape
     - (0,) * (len(x.shape) - (base_axis+1))
     - Padding sizes for dimensions.
   * - stride
     - Shape
     - (1,) * (len(x.shape) - (base_axis+1))
     - Stride sizes for dimensions.
   * - dilation
     - Shape
     - (1,) * (len(x.shape) - (base_axis+1))
     - Dilation sizes for dimensions.
   * - divisor
     - int64
     - 1
     - Number of input feature maps per output feature map.

* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - :math:`(B + 1 + N)`-D array (:math:`M_1 \times ... \times M_B \times C \times L'_1 \times ... \times L'_N`).
     -

MaxPooling
^^^^^^^^^^

Max pooling. It pools the maximum values inside the scanning kernel:

.. math::
    y_{i_1, i_2} = \max_{k_1, k_2 \in K} (x_{i_1 + k_1, i_2 + k_2})

where :math:`x_{i_1 + k_1, i_2 + k_2}` is the input and :math:`y_{i_1, i_2}` is the output.

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - Input variable.
     - 

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - kernel
     - Shape
     - 
     - Kernel sizes for each spatial axis.
   * - stride
     - Shape
     - kernel
     - Subsampling factors for each spatial axis.
   * - ignore_border
     - bool
     - True
     - If false, kernels covering borders are also considered for the output.
   * - pad
     - Shape
     - (0,) * len(kernel)
     - Border padding values for each spatial axis. Padding will be added both sides of the dimension.

* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - Maximum values variable
     - 

AveragePooling
^^^^^^^^^^^^^^

Average pooling. It pools the averaged values inside the scanning kernel:

.. math::
    y_{i_1, i_2} = \frac{1}{K_1 K_2} \sum_{k1} \sum_{k2} x_{i_1 + k_1, i_2 + k_2}

where :math:`x_{i_1 + k_1, i_2 + k_2}` is the input and :math:`y_{i_1, i_2}` is the output.

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - Input variable.
     - 

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - kernel
     - Shape
     - 
     - Kernel sizes for each spatial axis.
   * - stride
     - Shape
     - kernel
     - Subsampling factors for each spatial axis.
   * - ignore_border
     - bool
     - True
     - If false, kernels covering borders are also considered for the output.
   * - pad
     - Shape
     - (0,) * len(kernel)
     - Border padding values for each spatial axis. Padding will be added both sides of the dimension.
   * - including_pad
     - bool
     - True
     - If true, border padding values are considered for the output.


* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - Average values variable
     - 

GlobalAveragePooling
^^^^^^^^^^^^^^

Global average pooling. It pools an averaged value from the whole image

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - Input variable.
     - 

* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - Average values variable
     - 

SumPooling
^^^^^^^^^^

Sum pooling. It pools the summed values inside the scanning kernel:

.. math::
    y_{i_1, i_2} = \sum_{k1} \sum_{k2} x_{i_1 + k_1, i_2 + k_2}

where :math:`x_{i_1 + k_1, i_2 + k_2}` is the input and :math:`y_{i_1, i_2}` is the output.

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - Input variable.
     - 

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - kernel
     - Shape
     - 
     - Kernel sizes for each spatial axis.
   * - stride
     - Shape
     - kernel
     - Subsampling factors for each spatial axis.
   * - ignore_border
     - bool
     - True
     - If false, kernels covering borders are also considered for the output.
   * - pad
     - Shape
     - (0,) * len(kernel)
     - Border padding values for each spatial axis. Padding will be added both sides of the dimension.


* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - Summed values variable
     - 

Unpooling
^^^^^^^^^

Inverse operation of pooling. It spreads the input values:

.. math::
    y_{k_1 i_1 + j_1, k_2 i_2 + j_2} = x_{i_1, i_2}

where :math:`_{i_1, i_2}` is the input and :math:`y_{k_1 i_1 + j_1, k_2 i_2 + j_2}` is the output.

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - Input variable.
     - 

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - kernel
     - Shape
     - 
     - Kernel sizes for each spatial axis.


* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - Spread values variable
     - 

Embed
^^^^^

Embed slices of a matrix/tensor with indexing array/tensor.

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x0
     - Indices with shape :math:`(I_0, ..., I_N)`
     - Integer
   * - x1
     - Weights with shape :math:`(W_0, ..., W_M)`
     - 

* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - Output with shape :math:`(I_0, ..., I_N, W_1, ..., W_M)`
     - 

Neural Network Activation Functions
-----------------------------------

Sigmoid
^^^^^^^

Element-wise sigmoid function.

.. math::

    f(x) = \frac{1}{1 + \exp(-x)},

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - Input
     - 

* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - Output
     - 

Swish
^^^^^

Element-wise swish function, by Ramachandran et al. (2017).

.. math::

    y_i = \frac{x_i}{1 + \exp(-x_i)},

References:
    * `Prajit Ramachandran, Barret Zoph, and Quoc V. Le, Swish: a Self-Gated Activation Function, arXiv:1710.05941 [cs.NE]
      <https://arxiv.org/abs/1710.05941>`_
* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - Input
     - 

* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - Output
     - 

Tanh
^^^^

Element-wise hyperbolic tangent (tanh) function.

.. math::
    y_i = \tanh (x_i)

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - N-D array
     - 

* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - N-D array with the same shape as x
     - 

ReLU
^^^^

Element-wise Rectified Linear Unit (ReLU) function.

.. math::
    y_i = \max (0, x_i)

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - N-D array
     - 

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - inplace
     - bool
     - False
     - The output array is shared with the input array if True.

* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - N-D array with the same shape as x
     - 

LeakyReLU
^^^^^^^^^

Element-wise Leaky Rectified Linear Unit (ReLU) function.

It is defined as:

.. math::
    y_i = \alpha * \min(0, x_i) + \max (0, x_i)


* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - N-D array
     - 

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - alpha
     - float
     - 0.1
     - The slope value multiplied to negative numbers. :math:`\alpha` in the definition.

* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - N-D array with the same shape as x
     - 

Softmax
^^^^^^^

Softmax normalization. Calculates

.. math::
    y_i = \frac{\exp(x_i)}{\sum_j \exp(x_j)}

along the dimension specified by `axis`, where :math:`y_i` is the input and :math:`x_i` is the output.

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - N-D array. Typically indicates a score.
     - 

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - axis
     - int64
     - len(x.shape) - 1
     - Axis normalization is taken.


* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - N-D array with the same shape as x
     - 

ELU
^^^

Element-wise Exponential Linear Unit (ELU) function.

.. math::
    y_i= \left\{
    \begin{array}{ll}
    x_i & (x > 0)\\
    \alpha (\exp(x_i) - 1) & (x \leq 0)
    \end{array} \right..

References:
    * `Clevart et al., Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs).
      <http://arxiv.org/abs/1511.07289>`_

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - N-D array
     - 

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - alpha
     - double
     - 1.0
     - Coefficient for negative outputs. :math:`\alpha` in definition


* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - N-D array with the same shape as x
     - 

SELU
^^^^

Element-wise Scaled Exponential Linear Unit (SELU) function by Klambauer et al. (2017).

.. math::
    y_i= \lambda \left\{
    \begin{array}{ll}
    x_i & (x > 0)\\
    \alpha (\exp(x_i) - 1) & (x \leq 0)
    \end{array} \right..

The coefficients :math:`\lambda` and :math:`\alpha` default to the following values :math:`\lambda_{01}` and :math:`\alpha_{01}`, respectively, provided by Klambauer et al. (2017):

.. math::
    \begin{array}{lll}
      \lambda_{01} &=&  \left(  1 - \operatorname{erfc}\left( \frac{1}{\sqrt{2}} \right) \sqrt{e}  \right)
                  \sqrt{2 \pi} \\
                 && \left(
                      2 \operatorname{erfc} \left( \sqrt{2} \right) e^2
                      + \pi \operatorname{erfc}\left( \frac{1}{\sqrt{2}} \right)^2 e
                      \right. \\
                 && \left.
                      - 2(2 + \pi) \operatorname{erfc} \left( \frac{1}{\sqrt{2}} \right) \sqrt{e}
                      + \pi + 2
                 \right)^{-1/2}  \\
              &\approx& 1.0507 \\
      \alpha_{01} &=&  - \frac
                    {\sqrt {\frac {2}{\pi}}}
                    {\operatorname{erfc} \left( \frac{1}{\sqrt{2}} \right) \exp \left(\frac {1} {2} \right) - 1} \\
              &\approx& 1.67326
    \end{array}


References:
    * `Klambauer, G., Unterthiner, T., Mayr, A., & Hochreiter, S. (2017).
      Self-Normalizing Neural Networks. In Advances in Neural Information
      Processing Systems (NIPS). <https://arxiv.org/abs/1706.02515>`_

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - N-D array
     -

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - scale
     - double
     - 1.050700987355480
     - The coefficient :math:`\lambda` in the definition.
   * - alpha
     - double
     - 1.673263242354377
     - The coefficient :math:`\alpha` in the definition.

* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - N-D array with the same shape as x
     -

CReLU
^^^^^

Element-wise Concatenated Rectified Linear Unit (CReLU) function.
This function calculates the ReLU of :math:`x` and :math:`-x` , then concatenates the results together at a specified axis,
and returns the resulting array.


References:
    * `Wenling Shang, Kihyuk Sohn, Diogo Almeida, Honglak Lee.
      Understanding and Improving Convolutional Neural Networks
      via Concatenated Rectified Linear Units.
      <https://arxiv.org/abs/1603.05201>`_

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - N-D array.
     - 

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - axis
     - int64
     - 1
     - The ReLU activations of positive inputs and negative inputs are concatenated at axis.


* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - N-D array where axis dimension is doubled by concatenating.
     - 

CELU
^^^^

Element-wise Concatenated Exponential Linear Unit (CELU) function.
Concatenates ELU outputs of positive and negative inputs together at specified axis.

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - N-D array.
     - 

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - alpha
     - double
     - 1.0
     - Coefficient for negative outputs. :math:`\alpha` in definition.
   * - axis
     - int64
     - 1
     - The ELU activations of positive inputs and negative inputs are concatenated at axis.


* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - N-D array where axis dimension is doubled by concatenating.
     - 

PReLU
^^^^^

Element-wise Parametrized Rectified Linear Unit function. Calculates:

.. math::
    y_i = \max(0, x_i) + w_i \min(0, -x_i)

where negative slope :math:`w` is learned and can vary across channels (an
axis specified with `base_axis`).

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x0
     - (N-D array) Input
     - 
   * - x1
     - (N-D array) Weights
     - 

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - base_axis
     - int64
     - 1
     - Dimensions up to base_axis is treated as sample dimension.


* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - N-D array.
     - 

Normalization
-------------

BatchNormalization
^^^^^^^^^^^^^^^^^^

Batch normalization.

.. math::
    \begin{eqnarray}
      \mu &=& \frac{1}{M} \sum x_i \\
      \sigma^2 &=& \frac{1}{M} \sum \left(x_i - \mu\right)^2 \\
      \hat{x}_i &=& \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} \\
      y_i &=& \hat{x}_i \gamma + \beta.
    \end{eqnarray}


At testing time, the mean and variance values used are those that were computed during training by moving average.

References:

    * `Ioffe and Szegedy, Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift.
      <https://arxiv.org/abs/1502.03167>`_

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - N-D array of input.
     - 
   * - beta
     - N-D array of beta which is learned.
     - Parameter
   * - gamma
     - N-D array of gamma which is learned.
     - Parameter
   * - mean
     - N-D array of running mean (modified during forward execution).
     - Parameter
   * - variance
     - N-D array of running variance (modified during forward execution).
     - Parameter

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - axes
     - repeated int64
     - (1, )
     - Axes mean and variance are taken.
   * - decay_rate
     - float
     - 0.9
     - Decay rate of running mean and variance.
   * - eps
     - float
     - 1e-5
     - Tiny value to avoid zero division by std.
   * - batch_stat
     - bool
     - True
     - Use mini-batch statistics rather than running ones.


* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - N-D array
     - 

MeanSubtraction
^^^^^^^^^^^^^^^

It subtracts the mean of the elements of the input array,
and normalizes it to :math:`0`. Preprocessing arrays with this function has the effect of improving accuracy
in various tasks such as image classification.

At training time, this function is defined as

.. math::
    \begin{eqnarray}
      \mu &=& \frac{1}{M} \sum x_i \\
       rm &=& ({\rm decay\_rate}) rm + (1 - {\rm decay\_rate}) \mu \\
      y_i &=& x_i - rm
    \end{eqnarray}

At validation time, it is defined as

.. math::
    y_i = x_i - rm

Note:
    The backward performs an approximated differentiation that takes into account only the latest mini-batch.

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - N-D array of input.
     - 
   * - rmean
     - N-D array of running mean (modified during forward execution).
     - 
   * - t
     - Scalar of num of iteration of running mean (modified during forward execution).
     - 

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - base_axis
     - int64
     - 1
     - Base axis of Mean Subtraction operation. Dimensions up to base_axis is treated as sample dimension.
   * - update_running_mean
     - bool
     - True
     - Update running mean during forward execution.


* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - N-D array.
     - 

ClipGradByValue
^^^^^^^^^^^^^^^

In forward pass, the function behaves as the identity.

In backward pass, 

.. math::
	g_x = \begin{cases}
	        max & (g_y > max) \\
	        g_y & (otherwise) \\
	        min & (g_y < min)
	    \end{cases}.

ClipGradByNorm
^^^^^^^^^^^^^^^

In the forward pass, the function behaves like the identity.

In the backward pass, 

.. math::

    g_x = clip\_norm \times \frac{g_y}{\|g_y\|_2}


where :math:`g_x` is the gradient w.r.t the input, :math:`g_y` is the gradient w.r.t. the output, 
and :math:`clip\_norm` is the norm of :math:`g_y`. this is the case that `axes` is not set.  
When `axes` is set, the norm is computed over `axes`.

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - N-D array of input.
     - 
   * - min
     - N-D array of minimum input value by which the gradients of the `x` are clipped.
     - 
   * - max
     - N-D array of maximum input value by which the gradients of the `x` are clipped.
     - 

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - clip_norm
     - float
     - 1.0
     - Clip to the norm of input to `clip_norm` in the forward pass.
   * - axes
     - repeated int64
     - range(x.ndim)
     - Axes to be reduced. If empty list is given, all dimensions are reduced to scalar. This is used in the forward pass.

* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - N-D array.
     - 

Reduction
---------

Sum
^^^

Reduces a matrix along a specified axis with the sum function.

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - N-D array.
     - 

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - axes
     - repeated int64
     - range(x.ndim)
     - Axes to be reduced. If empty list is given, all dimensions are reduced to scalar.
   * - keep_dims
     - bool
     - False
     - Flag whether the reduced axis is kept.


* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - N-D array
     - 

Mean
^^^^

Reduces a matrix along a specified axis with the mean function.

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - N-D array.
     - 

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - axes
     - repeated int64
     - range(x.ndim)
     - Axes to be reduced.
   * - keep_dims
     - bool
     - False
     - Flag whether the reduced axis is kept.


* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - N-D array
     - 

Max
^^^

Reduction along axis or axes with max operation.

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - N-D array.
     - 

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - axes
     - repeated int64
     - range(x.ndim)
     - Axes to be reduced.
   * - keep_dims
     - bool
     - False
     - Flag whether the reduced axis is kept.


* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - N-D array
     - 

Min
^^^

Reduction along axis or axes with min operation.

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - N-D array.
     - 

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - axes
     - repeated int64
     - range(x.ndim)
     - Axes to be reduced.
   * - keep_dims
     - bool
     - False
     - Flag whether the reduced axis is kept.


* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - N-D array
     - 

Prod
^^^^

Reduction along axis or axes with product operation.

Note:
    Backward computation is not accurate in a zero value input.

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - N-D array.
     - 

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - axes
     - repeated int64
     - range(x.ndim)
     - Axes to be reduced.
   * - keep_dims
     - bool
     - False
     - Flag whether the reduced axis is kept.


* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - N-D array
     - 

ReduceSum
^^^^^^^^^

Reduction along an axis with sum operation.

Note:
    This is deprecated. Use ``sum`` instead.

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - N-D array.
     - 

* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - N-D array
     - 

ReduceMean
^^^^^^^^^^

Reduction by mean along an axis.

Note:
    This is deprecated. Use ``mean`` instead.

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - N-D array
     - 

* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - N-D array
     - 

Arithmetic
----------

Add2
^^^^

Element-wise addition.

.. math::
   y_i = x^{(0)}_i + x^{(1)}_i

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x0
     - N-D array
     - 
   * - x1
     - N-D array
     - 

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - inplace
     - bool
     - False
     - The output array is shared with the 1st input array if True.

* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - N-D array
     - 

BcAdd2
^^^^^^

Note: This shouldn't be called by users.


* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x0
     - N-D array
     - 
   * - x1
     - N-D array
     - 

* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - N-D array
     - 

Sub2
^^^^

Element-wise subtraction.

.. math::
   y_i = x^{(0)}_i - x^{(1)}_i

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x0
     - N-D array
     - 
   * - x1
     - N-D array
     - 

* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - N-D array
     - 

Mul2
^^^^

Element-wise multiplication.

.. math::
   y_i = x^{(0)}_i x^{(1)}_i

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x0
     - N-D array
     - 
   * - x1
     - N-D array
     - 

* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - N-D array
     - 

Div2
^^^^

Element-wise division.

.. math::
   y_i = \frac{x^{(0)}_i} {x^{(1)}_i}

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x0
     - N-D array
     - 
   * - x1
     - N-D array
     - 

* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - N-D array
     - 

Pow2
^^^^

Element-wise power function.

.. math::
   y_i = {(x^{(0)}_i)} ^ {x^{(1)}_i}

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x0
     - N-D array
     - 
   * - x1
     - N-D array
     - 

* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - N-D array
     - 

AddScalar
^^^^^^^^^

Element-wise scalar addition.

.. math::
   y_i = x_i + v


* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - Input variable
     - 

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - val
     - double
     - 1
     - Value of the scalar


* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - N-D array with the same shape as x
     - 

MulScalar
^^^^^^^^^

Element-wise scalar multiplication.

.. math::
   y_i = v x_i

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - Input variable
     - 

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - val
     - double
     - 1
     - Value of the scalar


* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - N-D array with the same shape as x
     - 

PowScalar
^^^^^^^^^

Element-wise scalar power function.

.. math::
   y_i = (x_i) ^ v

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - Input variable
     - 

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - val
     - double
     - 1
     - Value of the scalar


* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - N-D array with the same shape as x
     - 

RSubScalar
^^^^^^^^^^

Element-wise scalar subtraction.

.. math::
   y_i = v - x_i

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - Input variable
     - 

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - val
     - double
     - 1
     - Value of the scalar


* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - N-D array with the same shape as x
     - 

RDivScalar
^^^^^^^^^^

Element-wise scalar division.

.. math::
    y_i = \frac{v}{x_i}

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - Input variable
     - 

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - val
     - double
     - 1
     - Value of the scalar


* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - N-D array with the same shape as x
     - 

RPowScalar
^^^^^^^^^^

Element-wise scalar power function.

.. math::
    y_i = v ^ {x_i}

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - Input variable
     - 

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - val
     - double
     - 1
     - Value of the scalar


* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - N-D array with the same shape as x
     - 


Logical
-------

Sign
^^^^

Element-wise sign function.

In the forward pass, it is defined as

.. math::

    f(x) = \begin{cases}
        1  & (x > 0) \\
        -1 & (x < 0) \\
        \alpha & (x = 0)
    \end{cases}.

In the backward pass, it is defined as

.. math::
    \frac{\partial f(x)}{\partial x} = 1,

or in other words, it behaves as the identity function for the gradient in the backward pass.

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - Input
     - 

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - alpha
     - float
     - 0.0
     - Value in case of :math:`x = 0`.


* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - N-D array with the same shape as x
     - 

Minimum2
^^^^^^^^

Element-wise minimum.

.. math::
   y_i = \min(x^{(0)}_i, x^{(1)}_i)

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x0
     - N-D array
     - 
   * - x1
     - N-D array
     - 

* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - N-D array of min value
     - 

Maximum2
^^^^^^^^

Element-wise maximum.

.. math::
   y_i = \max(x^{(0)}_i, x^{(1)}_i)

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x0
     - N-D array
     - 
   * - x1
     - N-D array
     - 

* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - N-D array of max value
     - 

MinimumScalar
^^^^^^^^^^^^^

Element-wise scalar minimum.

.. math::
    y_i = \min(x_i, v)

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - Input variable
     - 

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - val
     - double
     - 1.0
     - Value of the scalar


* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - N-D array with the same shape as x
     - 

MaximumScalar
^^^^^^^^^^^^^

Element-wise scalar maximum.

.. math::
    y_i = \max (x_i, v)

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - Input variable
     - 

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - val
     - double
     - 1.0
     - Value of the scalar


* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - N-D array with the same shape as x
     - 

LogicalAnd
^^^^^^^^^^

Elementwise logical AND.

.. math::
    f(x^{(0)}_i,x^{(1)}_i) = \begin{cases}
        1 & (x^{(0)}_i \neq 0 \;\&\; x^{(1)}_i \neq 0) \\
        0 & otherwise
    \end{cases}.

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x0
     - N-D array
     - 
   * - x1
     - N-D array
     - 

* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - 
     - 

LogicalOr
^^^^^^^^^

Elementwise logical OR.

.. math::
    f(x^{(0)}_i,x^{(1)}_i) = \begin{cases}
        0 & (x^{(0)}_i = 0 \;\&\; x^{(1)}_i = 0) \\
        1 & otherwise
    \end{cases}.
* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x0
     - N-D array
     - 
   * - x1
     - N-D array
     - 

* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - 
     - 

LogicalXor
^^^^^^^^^^

Elementwise logical XOR.

.. math::
    f(x^{(0)}_i,x^{(1)}_i) = \begin{cases}
        1 & (x^{(0)}_i = 0 \;\&\; x^{(1)}_i = 0) \\
        1 & (x^{(0)}_i \neq 0 \;\&\; x^{(1)}_i \neq 0) \\
        0 & otherwise
    \end{cases}.

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x0
     - N-D array
     - 
   * - x1
     - N-D array
     - 

* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - 
     - 

Equal
^^^^^

Element wise 'equal'

.. math::
    f(x^{(0)}_i,x^{(1)}_i) = \begin{cases}
        1 & (x^{(0)}_i = x^{(1)}_i) \\
        0 & otherwise
    \end{cases}.
    
* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x0
     - N-D array
     - 
   * - x1
     - N-D array
     - 

* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - 
     - 

NotEqual
^^^^^^^^


Element wise 'not equal'

.. math::
    f(x^{(0)}_i,x^{(1)}_i) = \begin{cases}
        0 & (x^{(0)}_i = x^{(1)}_i) \\
        1 & otherwise
    \end{cases}.

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x0
     - N-D array
     - 
   * - x1
     - N-D array
     - 

* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - 
     - 

GreaterEqual
^^^^^^^^^^^^

Element wise comparison. The :math:`i^{th}` element of the output is:

.. math::

    f(x^{(0)}_i,x^{(1)}_i) = \begin{cases}
        1  & (x^{(0)}_i \geq x^{(1)}_i) \\
        0 & (x^{(0)}_i < x^{(1)}_i)
    \end{cases}.

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x0
     - N-D array
     - 
   * - x1
     - N-D array
     - 

* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - 
     - 

Greater
^^^^^^^

Element wise comparison. The :math:`i^{th}` element of the output is:

.. math::

    f(x^{(0)}_i,x^{(1)}_i) = \begin{cases}
        1  & (x^{(0)}_i > x^{(1)}_i) \\
        0 & (x^{(0)}_i \leq x^{(1)}_i)
    \end{cases}.

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x0
     - N-D array
     - 
   * - x1
     - N-D array
     - 

* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - 
     - 

LessEqual
^^^^^^^^^

Element wise comparison. The :math:`i^{th}` element of the output is:

.. math::

    f(x^{(0)}_i,x^{(1)}_i) = \begin{cases}
        1  & (x^{(0)}_i \leq x^{(1)}_i) \\
        0 & (x^{(0)}_i > x^{(1)}_i)
    \end{cases}.


* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x0
     - N-D array
     - 
   * - x1
     - N-D array
     - 

* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - 
     - 

Less
^^^^     

Element wise comparison. The :math:`i^{th}` element of the output is:

.. math::

    f(x^{(0)}_i,x^{(1)}_i) = \begin{cases}
        1  & (x^{(0)}_i < x^{(1)}_i) \\
        0 & (x^{(0)}_i \geq x^{(1)}_i)
    \end{cases}.


* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x0
     - N-D array
     - 
   * - x1
     - N-D array
     - 

* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - 
     - 

LogicalAndScalar
^^^^^^^^^^^^^^^^

Elementwise logical AND with scalar.

.. math::
    f(x_i,v) = \begin{cases}
        1 & (x_i \neq 0 \;\&\; v \neq 0) \\
        0 & otherwise
    \end{cases}.
    
* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x0
     - Input variable
     - 

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - val
     - bool
     - 
     - 


* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - N-D array with the same shape as x
     - 

LogicalOrScalar
^^^^^^^^^^^^^^^

Elementwise logical OR with scalar.

.. math::
    f(x_i,v) = \begin{cases}
        0 & (x_i = 0 \;\&\; v = 0) \\
        1 & otherwise
    \end{cases}.     
* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x0
     - Input variable
     - 

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - val
     - bool
     - 
     - 


* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - N-D array with the same shape as x
     - 

LogicalXorScalar
^^^^^^^^^^^^^^^^

Elementwise logical XOR with scalar.

.. math::
    f(x_i,v) = \begin{cases}
        1 & (x_i = 0 \;\&\; v = 0) \\
        1 & (x_i \neq 0 \;\&\; v \neq 0) \\
        0 & otherwise
    \end{cases}.
    
* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x0
     - Input variable
     - 

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - val
     - bool
     - 
     - 


* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - N-D array with the same shape as x
     - 

EqualScalar
^^^^^^^^^^^

Element wise 'equal' with a scalar

.. math::
    f(x_i,v) = \begin{cases}
        1 & (x_i = v) \\
        0 & otherwise
    \end{cases}.
     
* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x0
     - Input variable
     - 

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - val
     - double
     - 1
     - Value of the scalar


* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - N-D array with the same shape as x
     - 

* inputs
  * Variable x0
* outputs
  * Variable output
* params
  * double val

NotEqualScalar
^^^^^^^^^^^^^^

Element wise 'not equal' with a scalar

.. math::
    f(x_i,v) = \begin{cases}
        0 & (x_i = v) \\
        1 & otherwise
    \end{cases}.

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x0
     - Input variable
     - 

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - val
     - double
     - 1
     - Value of the scalar


* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - N-D array with the same shape as x
     - 

* inputs
  * Variable x0
* outputs
  * Variable output
* params
  * double val

GreaterEqualScalar
^^^^^^^^^^^^^^^^^^

Element wise comparison with a scalar. The :math:`i^{th}` element of the output is:

.. math::

    f(x^{(0)}_i,v) = \begin{cases}
        1  & (x^{(0)}_i \geq v \\
        0 & (x^{(0)}_i < v
    \end{cases}.
     
* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x0
     - Input variable
     - 

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - val
     - double
     - 1
     - Value of the scalar


* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - N-D array with the same shape as x
     - 

GreaterScalar
^^^^^^^^^^^^^

Element wise comparison with a scalar. The :math:`i^{th}` element of the output is:

.. math::

    f(x^{(0)}_i,v) = \begin{cases}
        1  & (x^{(0)}_i > v \\
        0 & (x^{(0)}_i \leq v
    \end{cases}.
     
* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x0
     - Input variable
     - 

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - val
     - double
     - 1
     - Value of the scalar


* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - N-D array with the same shape as x
     - 

LessEqualScalar
^^^^^^^^^^^^^^^

Element wise comparison with a scalar. The :math:`i^{th}` element of the output is:

.. math::

    f(x^{(0)}_i,v) = \begin{cases}
        1  & (x^{(0)}_i \leq v) \\
        0 & (x^{(0)}_i > v)
    \end{cases}.

     
* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x0
     - Input variable
     - 

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - val
     - double
     - 1
     - Value of the scalar


* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - N-D array with the same shape as x
     - 

LessScalar
^^^^^^^^^^

Element wise comparison with a scalar. The :math:`i^{th}` element of the output is:

.. math::

    f(x^{(0)}_i,v) = \begin{cases}
        1  & (x^{(0)}_i < v) \\
        0 & (x^{(0)}_i \geq v)
    \end{cases}.
     
* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x0
     - Input variable
     - 

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - val
     - double
     - 1
     - Value of the scalar


* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - N-D array with the same shape as x
     - 

LogicalNot
^^^^^^^^^^

Element-wise logical NOT operation

.. math::
    f(x_i) = \begin{cases}
        1 & (x_i = 0) \\
        0 & otherwise
    \end{cases}.
     
* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x0
     - Input variable
     - 

* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - N-D array with the same shape as x
     - 

Math
----

Constant
^^^^^^^^

Generate a constant-valued array.

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - val
     - float
     - 0
     - Constant value.
   * - shape
     - Shape
     - []
     - Shape of the output array.

* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - N-D array where all values are the specified constant.
     -

Abs
^^^

Element-wise absolute value function.

.. math::
   y_i = |x_i|

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - Input variable
     - 

* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - Element-wise absolute variable
     - 

Exp
^^^

Element-wise natural exponential function.

.. math::
   y_i = \exp(x_i).

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - Input variable
     - 

* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - Element-wise exp variable
     - 

Log
^^^

Element-wise natural logarithm function.

.. math::
   y_i = \ln(x_i).

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - Input variable
     - 

* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - Element-wise log variable
     - 

Identity
^^^^^^^^

Identity function.

.. math::
    y = x

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - N-D array.
     - 

* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - N-D array
     - 


BatchMatmul
^^^^^^^^^^^

Batch matrix multiplication.

Two of batches of matrices are multiplied for each sample in a batch. A batch of matrices is composed as [..., P, Q] where the last two dimensions compose matrix dimensions, and the first dimensions up to the third last dimension are considered as batch samples.

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - a
     - N-D array with >= 2-dim. The last two dimensions will be treated as a matrix.
     -
   * - b
     - N-D array with >= 2-dim. The last two dimensions will be treated as a matrix. The product of the size of 0-th dimension through the size of the third last dimension must be same as that of the input ``a``.
     -

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - transpose_a
     - bool
     - False
     - Transpose the last two axes of ``a`` in matrix multiplication.
   * - transpose_b
     - bool
     - False
     - Transpose the last two axes of ``b`` in matrix multiplication.


* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - Output of sample-wise matrix multiplication in a batch. When ``a`` is of a shape of [N, P, Q], ``b`` is of a shape of [N, Q, R], and transpose options are all False, the output will be a shape of [N, P, R].
     - 

Round
^^^^^^^^^^

Element-wise round function.

In the forward pass, this function simply computes `round` to the nearest integer value.

.. math::
    y_i = round(x_i).

In the backward pass, the simple Straight-Through Estimator (STE) is applied, 

.. math::
    \frac{\partial y_i}{\partial x_i} = 1.
 

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - Input variable
     - 

* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - N-D array with the same shape as x
     - 

Array Manipulation
------------------

Concatenate
^^^^^^^^^^^

Concatenate a variable number of input arrays along the specified axis.

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - N-D arrays.
     - Variadic Parameter

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - axis
     - int64
     - len(x[0].shape) - 1
     - Axis


* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - Concatenate variable
     - 

Split
^^^^^

Split arrays at the specified axis.

note:
    This function should not be called directly when constructing models.
    Instead, use :meth:`nnabla.functions.split` which
    automatically sets `n_output` from the input's shape and axis.
  
* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - N-D array
     - 

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - axis
     - int64
     - 0
     - Axis


* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - list of N-D arrays
     - Variadic Parameter

Stack
^^^^^

Joins two or more arrays on a new axis.

Note:
    Unlike :meth:`nnabla.functions.concatenate` , which joins arrays on an existing axis,
    Stack joins arrays on a new axis.

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - N-D arrays. The sizes of all the arrays to be stacked must be the same.
     - Variadic Parameter

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - axis
     - int64
     - 0
     - The axis on which to concatenate arrays. Axis indices take on values 0, 1, 2, and so on from the left. For example, to stack four (3,28,28) inputs on the second axis, specify 1. In this case, the output size will be (3,4,28,28).


* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - Output
     - 

Slice
^^^^^

Slice arrays along specified axis.

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - N-D array
     - 

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - start
     - repeated int64
     - (0,) * len(x.shape)
     - Start indices for each axis
   * - stop
     - repeated int64
     - tuple(x.shape)
     - Stop indices for each axis
   * - step
     - repeated int64
     - (1,) * len(x.shape)
     - Step indices for each axis


* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - Sliced N-D array
     - 

Transpose
^^^^^^^^^

Transposes tensor dimensions.

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - N-D array
     - 

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - axes
     - repeated int64
     - 
     - Source axis indices for each axis.


* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - Transposed N-D array.
     - 

Broadcast
^^^^^^^^^

Broadcasting ND-array to the specified shape.

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - N-D array
     - 

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - shape
     - Shape
     - 
     - Shape broadcasted to. The size must be the same in axis where ``x``'s shape is not 1.


* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - Broadcasted N-D array
     - 

BroadcastTo
^^^^^^^^^^^

Broadcasting ND-array to the specified buffer

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - N-D array
     - 
   * - y
     - N-D array
     - 

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - axis
     - int64
     - -1
     - Target axis to start broadcasting. If this is not set, broadcast will try to fit y to x starting from the last dimension


* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - z
     - Broadcasted N-D array
     - 

OneHot
^^^^^^

OneHot creates one-hot vector based on input indices.

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - N-D array
     - Integer

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - shape
     - Shape
     - 
     - 

* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - output
     - N-D array
     - 

Flip
^^^^

Reverses the order of elements of the specified dimension of an array.

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - N-D array
     - 

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - axes
     - repeated int64
     - [len(x.shape) - 1]
     - The index of the dimension to reverse the order of the elements. Axis indices take on values 0, 1, 2, and so on from the left. For example, to flip a 32 (W) by 24 (H) 100 RGB image (100,3,24,32) vertically and horizontally, specify (2,3).


* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - N-D array
     - 

Shift
^^^^^

Shifts the array elements by the specified amount.

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - N-D array.
     - 

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - shifts
     - repeated int64
     - (0,) * len(x.shape)
     - The amount to shift elements. For example, to shift image data to the right by 2 pixels and up 3 pixels, specify (-3,2).
   * - border_mode
     - string ("nearest" or "reflect")
     - "nearest"
     - Specify how to process the ends of arrays whose values will be undetermined as a result of shifting. nearest: The data at the ends of the original      array is copied and used. reflect: Original data reflected      at the ends of the original array is used.


* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - N-D array.
     - 

Reshape
^^^^^^^

Reshapes the input variable in-place. It does not create a copy of the variable.
The output variable (y) has a new shape but points to the same data as the input variable (x).
This means that if the data in the output variable (y) is modified, the data in the input
variable (x) also gets modified since the reshape was done in-place.

Note:
    This function has the same behavior as the :meth:`nnabla.Variable.reshape` method.

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - N-D array.
     - 

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - shape
     - Shape
     - 
     - Dimensions for each axis


* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - Reshaped N-D array
     - 

MatrixDiag
^^^^^^^^^^

Returns an array where the last two dimensions consist of the diagonal matrix. 

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - N-D array with shape (:math:`M_0 \times \ldots \times M_N`). 
     - 

* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - N-D array with shape (:math:`M_0 \times \ldots \times M_N \times M_N`). 
     The last two axes has the same dimension. 
     - 

MatrixDiagPart
^^^^^^^^^^^^^^

Returns an array in which the values of the last dimension consist of the diagonal 
elements of the last two dimensions of an input array.

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - N-D array with shape (:math:`M_0 \times \ldots \times M_N \times M_N`).
     The last two axes has the same dimension.   
     - 

* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - N-D array with shape (:math:`M_0 \times \ldots \times M_N`). 
     - 

Stochasticity
-------------

Dropout
^^^^^^^

Dropout. 
Samples a number :math:`u` from a uniform distribution in :math:`[0, 1]` ,
and ignores the input if :math:`u > p`.

.. math::
    y = \left\{
    \begin{array}{ll}
      \frac{x}{1 - p} & (u > p) \\
      0 & ({\rm otherwise})
    \end{array} \right.

Note:
    Usually dropout only applied during training as below
    (except `Bayesian dropout`_).

    .. code-block:: python

        h = PF.affine(x, num_hidden)
        if train:
            h = F.dropout(h, 0.5)

.. _Bayesian dropout: https://arxiv.org/abs/1506.02142

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - N-D array
     - 

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - p
     - double
     - 0.5
     - :math:`p` in definition.
   * - seed
     - int64
     - -1
     - Random seed. When -1, seed is sampled from global random number generator.


* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - N-D array with the same shape as x
     - 

TopKData
^^^^^^^^

Select the `k` largest values from each sample in `x` to propagate
unmodified and set all other values to 0. If `abs` is True, the `k`
largest values are selected by magnitude. If `reduce` is True (the
default), all feature dimensions are reduced to a single dimension of
size `k` that propagates only the `k` largest values. Otherwise, if
`reduce` is False, input and output dimensions are identical.
Dimensions before `base_axis` are treated as number of sample
dimensions and `k` values get selected from all elements of a
sample (dimensions from `base_axis`) regardless of shape.

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - N-D array
     -

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - k
     - int64
     - 
     - Number of largest data values to propagate.
   * - abs
     - bool
     - False
     - Determine largest data values by magnitude.
   * - reduce
     - bool
     - True
     - Reduce feature size to one dimension of size `k`.
   * - base_axis
     - int64
     - 1
     - First dimension of the sample shape.

* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - N-D array.
     -

TopKGrad
^^^^^^^^

Select the `k` largest gradients for each sample in `x` to
back-propagate unmodified and set all other gradients to 0.
If `abs` is True, the `k` largest gradients are selected by
magnitude. Dimensions before `base_axis` are treated as number
of sample dimensions and `k` gradients get selected from all
gradients of a sample (dimensions from `base_axis`)
regardless of shape.

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - N-D array
     -

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - k
     - int64
     - 
     - Number of largest data values to propagate.
   * - abs
     - bool
     - False
     - Determine largest data values by magnitude.
   * - base_axis
     - int64
     - 1
     - First dimension of the sample shape.

* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - N-D array with same shape and data as `x`.
     -

Rand
^^^^

Samples numbers from a uniform distribution :math:`x \sim U(low, high)`
given lowest value :math:`low`, upper bound :math:`high`,
and shape of the returned Variable.

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - low
     - float
     - 0
     - :math:`low` in definition.
   * - high
     - float
     - 1
     - :math:`high` in definition.
   * - shape
     - Shape
     - []
     - Shape of returned variable.
   * - seed
     - int64
     - -1
     - Random seed. When -1, seed is sampled from global random number generator.

* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - Variable with the shape specified in the argument.
     - 


Randint
^^^^^^^

Samples integer numbers from a uniform distribution :math:`x \sim U(low, high)`
given lowest value :math:`low`, upper bound :math:`high`,
and shape of the returned Variable.

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - low
     - int64
     - 0
     - :math:`low` in definition.
   * - high
     - int64
     - 1
     - :math:`high` in definition.
   * - shape
     - Shape
     - []
     - Shape of returned variable.
   * - seed
     - int64
     - -1
     - Random seed. When -1, seed is sampled from global random number generator.

* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - Variable with the shape specified in the argument. The dtype is int32.
     - Integer


Randn
^^^^^

Samples numbers from a normal distribution :math:`x \sim N(\mu, \sigma)`
given mean :math:`\mu`, standard deviation :math:`\sigma`,
and shape of the returned Variable.

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - mu
     - float
     - 0
     - :math:`\mu` in definition.
   * - sigma
     - float
     - 1
     - :math:`\sigm` in definition.
   * - shape
     - Shape
     - []
     - Shape of returned variable.
   * - seed
     - int64
     - -1
     - Random seed. When -1, seed is sampled from global random number generator.


* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - Variable with the shape specified in the argument.
     - 

RandomCrop
^^^^^^^^^^

RandomCrop randomly extracts a portion of an array.

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - N-D array
     - 

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - shape
     - Shape
     - x.shape
     - The data size to extract. For example, to randomly extract a portion of the image (3,48,48) from a 3,64,64 image, specify (3,48,48).
   * - base_axis
     - int64
     - 1
     - 
   * - seed
     - int64
     - -1
     - Random seed. When -1, seed is sampled from global random number generator.


* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - N-D array
     - 

RandomFlip
^^^^^^^^^^

Reverses the order of elements of the specified dimension of an array at 50% probability.

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - N-D array
     - 

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - axes
     - repeated int64
     - [len(x.shape) - 1]
     - The index of the axis to reverse the order of the elements. Axis indices take on values 0, 1, 2, and so on from the left. For example, to flip a 32 (W) by 24 (H) 100 RGB images (100, 3,24,32) vertically and horizontally at random, specify (2,3).
   * - base_axis
     - int64
     - 1
     - 
   * - seed
     - int64
     - -1
     - Random seed. When -1, seed is sampled from global random number generator.


* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - N-D array
     - 

RandomShift
^^^^^^^^^^^^

Randomly shifts the array elements within the specified range.

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - N-D array.
     - 

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - shifts
     - repeated int64
     - (0,) * len(x.shape)
     - Max absolute amount to shift elements. For example, to shift image data horizontally by :math:`\pm 2` pixels and vertically by :math:`\pm 3` pixels, specify (3,2).
   * - border_mode
     - string ("nearest" or "reflect")
     - "nearest"
     - Specify how to process the ends of arrays whose values will be undetermined as a result of shifting. nearest: The data at the ends of the   original array is copied and used. reflect: Original data reflected at   the ends of the original array is used.
   * - base_axis
     - int64
     - 1
     - 
   * - seed
     - int64
     - -1
     - Random seed. When -1, seed is sampled from global random number generator.


* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - N-D array.
     - 

ImageAugmentation
^^^^^^^^^^^^^^^^^

ImageAugmentation randomly alters the input image.

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - N-D array.
     - 

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - shape
     - Shape
     - x.shape
     - The output image data size.
   * - pad
     - Shape
     - (0, 0)
     - Border padding values for each spatial axis. Padding will be added both sides of the dimension.
   * - min_scale
     - float
     - 1.0
     - The minimum scale ratio when randomly scaling the image. For example, to scale down to 0.8 times the size of the original image, specify "0.8". To not apply random scaling, set both min_scale and max_scale to "1.0".
   * - max_scale
     - float
     - 1.0
     - The maximum scale ratio when randomly scaling the image. For example, to scale down to 2 times the size of the original image, specify "2.0".
   * - angle
     - float
     - 0.0
     - The rotation angle range in radians when randomly rotating the image. The image is randomly rotated in the -Angle to +Angle range. For example, to rotate in a +-15 degree range, specify "0.26" (15 degrees/360 degrees * 2PI). To not apply random rotation, specify "0.0".
   * - aspect_ratio
     - float
     - 1.0
     - The aspect ratio range when randomly deforming the image. For example, to deform aspect ratio of image from 1:1.3 to 1.3:1, specify "1.3". To not apply random deforming, specify "1.0".
   * - distortion
     - float
     - 0.0
     - The distortion range when randomly distorting the image. To not apply distortion, specify "0.0".
   * - flip_lr
     - bool
     - False
     - Whether to randomly flip the image horizontally at 50% probability.
   * - flip_ud
     - bool
     - False
     - Whether to randomly flip the image vertically at 50% probability.
   * - brightness
     - float
     - 0.0
     - The absolute range of values to randomly add to the brightness. A random value in the -Brightness to +Brightness range is added to the brightness. For example, to vary the brightness in the -0.05 to +0.05 range, specify "0.05". To not apply random addition to brightness, specify "0.0".
   * - brightness_each
     - bool
     - False
     - Whether to apply the random addition to brightness (as specified by brightness) to each color channel. True: brightness is added based on a different random number for each channel. False: brightness is added based on a random number common to all channels.
   * - contrast
     - float
     - 1.0
     - The range in which to randomly vary the image contrast. The contrast is varied in the 1/Contrast times to Contrast times range. The output brightness is equal to (input - contrast_center) * contrast + contrast_center. For example, to vary the contrast in the 0.91 times to 1.1 times range, specify "1.1". To not apply random contrast variation, specify "1.0".
   * - contrast_center
     - float
     - 0.0
     - Intensity center used for applying contrast.
   * - contrast_each
     - bool
     - False
     - Whether to apply the random contrast variation (as specified by contrast) to each color channel. True: contrast is varied based on a different random number for each channel. False: contrast is varied based on a random number common to all channels.
   * - noise
     - float
     - 0.0
     - Sigma of normal random number to be added.
   * - seed 
     - int64
     - -1
     - Random seed. When -1, seed is sampled from global random number generator.

* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - N-D array.
     - 

     
Loss Functions
--------------

SigmoidCrossEntropy
^^^^^^^^^^^^^^^^^^^

Element-wise cross entropy between `x` and the target variables, passed to a sigmoid function.

.. math::
    y_i = - \left(x^{(1)}_i \ln \left(\sigma \left(x^{(0)}_i \right)\right) + \
    \left(1 - x^{(1)}_i\right) \ln \left(1 - \sigma \left(x^{(0)}_i \
    \right)\right)\right)

where :math:`\sigma(s)=\frac{1}{1+\exp(-s)}`.

Note:
    SigmoidCrossEntropy is equivalent to Sigmoid+BinaryCrossEntropy, but computing them at once has the effect of reducing computational error.

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - N-D array. Typically indicates a score. The value lies in :math:`[-\infty, \infty]`
     - Parameter
   * - target
     - N-D array of labels. Only 0 or 1 value is allowed.
     - Integer Parameter

* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - N-D array of element-wise losses.
     - 

BinaryCrossEntropy
^^^^^^^^^^^^^^^^^^

Element-wise cross entropy between `x` and the target variables.

.. math::
    y_i = - \left(x^{(1)}_i * \ln \left(x^{(0)}_i\right) + \left(1 - \
    x^{(1)}_i\right) * \ln \left(1 - x^{(0)}_i\right)\right).

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - Probabilities N-D array. :math:`-\infty` to :math:`\infty`.
     - 
   * - target
     - N-D array of labels. Usually set as 0 or 1, but, unlike SigmoidCrossEntropy, it allows probability (0 to 1) as inputs and backpropagation can be done.
     - 

* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - N-D array of element-wise losses.
     - 

SoftmaxCrossEntropy
^^^^^^^^^^^^^^^^^^^

Element-wise cross entropy between the variables and the variables of a label given by a category index with Softmax normalization.

.. math::
    y_{j} = -\ln \left(\frac{\exp(x_{j,t_j})}{\sum_{i'} \exp(x_{j,i'})}\right)

along dimension specified by axis (:math:`i` is the axis where normalization is performed on).

Note:
    SoftmaxCrossEntropy is equivalent to Softmax+CategoricalCrossEntropy, but computing them at once has the effect of reducing computational error.

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - N-D array. Typically indicates a score. :math:`(D_1 \times ... \times D_i \times ... \times D_N)`
     - Parameter
   * - target
     - N-D array of labels. :math:`(D_1 \times ... \times 1 \times ... \times D_N)`
     - Integer Parameter

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - axis
     - int64
     - len(x.shape) - 1
     - Axis normalization is taken.


* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - N-D array of element-wise losses. :math:`(D_1 \times ... \times 1 \times ... \times D_N)`
     - 

CategoricalCrossEntropy
^^^^^^^^^^^^^^^^^^^^^^^

Element-wise cross entropy between `x` and the target `t` where targets are given by a category index.

.. math::
    y_{j} = -\ln \left( x_{j, t_j} \right)

along dimension specified by axis (:math:`i` is the axis where normalization is performed on).

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - N-D array. Typically indicates a score. :math:`(D_1 \times ... \times D_i \times ... \times D_N)`
     - Parameter
   * - target
     - N-D array of labels. :math:`(D_1 \times ... \times 1 \times ... \times D_N)`
     - Integer Parameter

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - axis
     - int64
     - len(x.shape) - 1
     - Axis normalization is taken.


* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - N-D array of element-wise losses. :math:`(D_1 \times ... \times 1 \times ... \times D_N)`
     - 

SquaredError
^^^^^^^^^^^^

Element-wise squared error

.. math::
    y_i = \left(x^{(0)}_i - x^{(1)}_i\right)^2.

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x0
     - N-D array.
     - 
   * - x1
     - N-D array.
     - 

* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - N-D array.
     - 

AbsoluteError
^^^^^^^^^^^^^

Element-wise absolute error

.. math::
    y_i = | x^{(0)}_i - x^{(1)}_i |.

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x0
     - N-D array.
     - 
   * - x1
     - N-D array.
     - 

* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - N-D array.
     - 

HuberLoss
^^^^^^^^^

Element-wise Huber loss

.. math::
    y_i= \left\{
    \begin{array}{ll}
      d^2 & (|d| < \delta)\\
      \delta (2 |d| - \delta) & ({\rm otherwise})
    \end{array} \right.

where :math:`d = x^{(0)}_i - x^{(1)}_i`

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x0
     - N-D array.
     - 
   * - x1
     - N-D array.
     - 

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - delta
     - float
     - 1.0
     - Delta


* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - N-D array of element-wise losses.
     - 

EpsilonInsensitiveLoss
^^^^^^^^^^^^^^^^^^^^^^

Element-wise Epsilon Insensitive Loss

.. math::
    y_i= \left\{
    \begin{array}{ll}
      | x^{(0)}_i - x^{(1)}_i | - \epsilon & if \ \ | x^{(0)}_i - x^{(1)}_i | > \epsilon \\
			0 & otherwise       
    \end{array} \right.

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x0
     - N-D array.
     - 
   * - x1
     - N-D array.
     - 

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - epsilon
     - float
     - 
     - Insensitive parameter.


* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - N-D array of element-wise losses.
     - 

KLMultinomial
^^^^^^^^^^^^^

The Kullback Leibler Divergence for multinomial distributions.

.. math::
    D = \sum_i p_i \log \left( \frac{p_i}{q_i} \right)

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - p
     - N-D array of the source categorical probabilities
     - 
   * - q
     - N-D array of the target categorical probabilities
     - 

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - base_axis
     - int64
     - 1
     - Dimensions up to base_axis is treated as sample dimension.


* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - D
     - Kullback Leibler divergence :math:`KL(p \parallel q)`.
     - 

Quantization Neural Network Layers
----------------------------

BinarySigmoid
^^^^^^^^^^^^^

Element-wise binary sigmoid function. In the forward pass, it computes

.. math::
    f(x) = \begin{cases}
        1 & (x > 0) \\
        0 & ({\rm otherwise})\end{cases},

but in the backward pass, a straight-through approximation of the gradient
is used, i.e.,

.. math::
    \frac{\partial f(x)}{\partial x} =
    \begin{cases}
        0 & (|x| \geq 1) \\
        \frac{1}{2} & ({\rm otherwise})
    \end{cases}.

References:

    * `Courbariaux, Matthieu, and Yoshua Bengio. Binarynet: Training deep
      neural networks with weights and activations constrained to+ 1 or-1.
      <https://arxiv.org/abs/1602.02830>`_


* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - Input .
     - 

* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - Output.
     - 

BinaryTanh
^^^^^^^^^^^

Element-wise binary tanh function. In the forward pass, it computes

.. math::
    f(x) = \begin{cases}
        1 & (x > 0) \\ 
        -1 & ({\rm otherwise})
    \end{cases},

but in the backward pass, a straight-through approximation of the gradient
is used, i.e.,

.. math::
    \frac{\partial f(x)}{\partial x} =
    \begin{cases}
        0 & (|x| \geq 1) \\
        1 & ({\rm otherwise}) \end{cases}.

References:

    * `Courbariaux, Matthieu, and Yoshua Bengio. Binarynet: Training deep
      neural networks with weights and activations constrained to+ 1 or-1.
      <https://arxiv.org/abs/1602.02830>`_

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - Input .
     - 

* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - Output.
     - 

BinaryConnectAffine
^^^^^^^^^^^^^^^^^^^

This function provides a BinaryConnect affine layer. It computes in
the forward pass

.. math::

    y_j = \sum_{i} sign(w_{j,i}) x_i,

i.e., the weights :math:`w_{j,i}` are binarized to :math:`sign(w_{j,i})` and,
hence, each weight is in :math:`\{-1,\,1\}`. By this weight binarization, the
inner product computations do not require any multiplications anymore as
they turn into additions/subtractions.

This function should be used together with
:meth:`~nnabla.functions.batch_normalization`.

.. note::

    1) If you would like to share the binary weights between other
    layers, please use the standard, floating value weights (`weight`)
    and not the binary weights (`binary_weight`).

    2) The weights and the binary weights become in sync only after a call to
    :meth:`~nnabla.Variable.forward`, and not after a call to
    :meth:`~nnabla.Variable.backward`. If you wish to store the parameters of
    the network, remember to call :meth:`~nnabla.Variable.forward`, once before
    doing so, otherwise the weights and the binary weights will not be in sync.

    3) CPU and GPU implementations now use floating values for `binary_weight`,
    since this function is for simulation purposes.

References:

    * `M. Courbariaux, Y. Bengio, and J.-P. David. BinaryConnect:
      Training Deep Neural Networks with binary weights during propagations.
      <https://arxiv.org/abs/1511.00363>`_

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - Input .
     - 
   * - weight
     - Weight .
     - Parameter
   * - binary_weight
     - Binarized weight .
     - Parameter
   * - bias
     - Bias.
     - Optional Parameter

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - base_axis
     - int64
     - 1
     - Dimensions up to base_axis is treated as sample dimension.


* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - Output.
     - 

BinaryConnectConvolution
^^^^^^^^^^^^^^^^^^^^^^^^

This function provides a BinaryConnect convolution layer. It computes in
the forward pass

.. math::

    y_{n, a, b} = \sum_{m} \sum_{i} \sum_{j} sign(w_{n, m, i, j}) x_{m, a + i, b + j},

i.e., the weights :math:`w_{n, m, i, j}` are binarized to
:math:`sign(w_{n, m, i, j})` and, hence,
each weight is in :math:`\{-1,\,1\}`. By this weight binarization, the
inner product computations do not require any multiplications anymore as
they turn into additions/subtractions.

This function should be used together with :meth:`~nnabla.functions.batch_normalization`.

Reference

    * `M. Courbariaux, Y. Bengio, and J.-P. David. BinaryConnect:
      Training Deep Neural Networks with binary weights during propagations.
      <https://arxiv.org/abs/1511.00363>`_


.. note::

    1) If you would like to share the binary weights between other
    layers, please use the standard, floating value weights (`weight`)
    and not the binary weights (`binary_weight`).

    2) The weights and the binary weights become in sync only after a call to
    :meth:`~nnabla.Variable.forward`, and not after a call to
    :meth:`~nnabla.Variable.backward`. If you wish to store the parameters of
    the network, remember to call :meth:`~nnabla.Variable.forward`, once before
    doing so, otherwise the weights and the binary weights will not be in sync.

    3) CPU and GPU implementations now use floating values for `binary_weight`,
    since this function is for simulation purposes.

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - Input.
     - 
   * - weight
     - Weight.
     - Parameter
   * - binary_weight
     - Binarized weight.
     - Parameter
   * - bias
     - Bias.
     - Optional Parameter

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - base_axis
     - int64
     - 1
     - Dimensions up to base_axis is treated as sample dimension.
   * - pad
     - Shape
     - (0,) * (len(x.shape) - (base_axis+1))
     - Padding sizes for dimensions.
   * - stride
     - Shape
     - (1,) * (len(x.shape) - (base_axis+1))
     - Stride sizes for dimensions.
   * - dilation
     - Shape
     - (1,) * (len(x.shape) - (base_axis+1))
     - Dilation sizes for dimensions.
   * - group
     - int64
     - 1
     - Number of groups of channels. This makes the connection across channels sparser, by grouping connections along the mapping direction.


* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - Output
     - 

BinaryWeightAffine
^^^^^^^^^^^^^^^^^^

This function provides a Binary Weight Network affine layer. It computes in
the forward pass

.. math::

    y_j = \frac{1}{\|\mathbf{w}_j\|_{\ell_1}} \sum_{i} sign(w_{j,i}) x_i

i.e., the weights :math:`w_{j,i}` are binarized to :math:`sign(w_{j,i})` and,
hence, each weight is in :math:`\{-1,\,1\}`. By this weight binarization, the
inner product computations turn into additions/subtractions which are followed
by multiplication with the scaling factor
:math:`\alpha_j = \frac{1}{\|\mathbf{w}_j\|_{\ell_1}}`.

Reference

    * `Rastegari, Mohammad, et al. XNOR-Net: ImageNet Classification Using
      Binary Convolutional Neural Networks.
      <https://arxiv.org/abs/1603.05279>`_

.. note::

    1) If you would like to share the binary weights with other layers, please
    use the standard, floating value weights (`weight`) and not the binary
    weights (`binary_weight`).

    2) The weights and the binary weights become in sync only after a call to
    :meth:`~nnabla.Variable.forward`, and not after a call to
    :meth:`~nnabla.Variable.backward`. If you wish to store the parameters of
    the network, remember to call :meth:`~nnabla.Variable.forward`, once before
    doing so, otherwise the weights and the binary weights will not be in sync.

    3) CPU and GPU implementations now use floating values for `binary_weight`,
    since this function is for simulation purposes.

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - Input .
     - 
   * - weight
     - Weight.
     - Parameter
   * - binary_weight
     - Binarized weight.
     - Parameter
   * - alpha
     - Alpha.
     - Parameter
   * - bias
     - Bias.
     - Optional Parameter

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - base_axis
     - int64
     - 1
     - Dimensions up to base_axis is treated as sample dimension.


* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - Output.
     - 

BinaryWeightConvolution
^^^^^^^^^^^^^^^^^^^^^^^

This function provides a Binary Weight Network convolution layer. It computes in
the forward pass

.. math::

    y_{n, a, b} = \frac{1}{\|\mathbf{w}_n\|_{\ell_1}} \sum_{m} \sum_{i} \sum_{j} sign(w_{n, m, i, j}) x_{m, a + i, b + j}.

i.e., the weights :math:`w_{n, m, i, j}` are binarized to
:math:`sign(w_{n, m, i, j})` and, hence, each weight is in :math:`\{-1,\,1\}`.
By this weight binarization, the inner product computations turn into
additions/subtractions which are followed by multiplication with the scaling
factor :math:`\alpha_n = \frac{1}{\|\mathbf{w}_n\|_{\ell_1}}`.

Reference

    * `Rastegari, Mohammad, et al. XNOR-Net: ImageNet Classification Using
      Binary Convolutional Neural Networks.
      <https://arxiv.org/abs/1603.05279>`_

.. note::

    1) If you would like to share the binary weights between other standard layers, please
    use the standard, floating value weights (`weight`)
    and not the binary weights (`binary_weight`).

    2) The weights and the binary weights become in sync only after a call to
    :meth:`~nnabla.Variable.forward`, and not after a call to
    :meth:`~nnabla.Variable.backward`. If you wish to store the parameters of
    the network, remember to call :meth:`~nnabla.Variable.forward`, once
    before doing so, otherwise the weights and the binary weights will not be
    in sync.

    3) CPU and GPU implementations now use floating values for `binary_weight`,
    since this function is for simulation purposes.

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - Input.
     - 
   * - weight
     - Weight.
     - Parameter
   * - binary_weight
     - Binarized weight.
     - Parameter
   * - alpha
     - Alpha.
     - Parameter
   * - bias
     - Bias.
     - Optional Parameter

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - base_axis
     - int64
     - 1
     - Dimensions up to base_axis is treated as sample dimension.
   * - pad
     - Shape
     - (0,) * (len(x.shape) - (base_axis+1))
     - Padding sizes for dimensions.
   * - stride
     - Shape
     - (1,) * (len(x.shape) - (base_axis+1))
     - Stride sizes for dimensions.
   * - dilation
     - Shape
     - (1,) * (len(x.shape) - (base_axis+1))
     - Dilation sizes for dimensions.
   * - group
     - int64
     - 1
     - Number of groups of channels. This makes the connection across channels sparser, by grouping connections along the mapping direction.


* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - Output
     - 

INQAffine
^^^^^^^^^

This function provides a INQ affine layer. It computes in
the forward pass

.. math::

    y_j = \sum_{i} w_{j,i} x_i,

where the weights :math:`w_{j,i}` are quantized sequentially during
training to power-of-two numbers. In the backward pass, only the non-fixed
(i.e., learnable) weights are updated.

References:

    * `Zhou A, Yao A, Guo Y, Xu L, Chen Y. Incremental network quantization:
      Towards lossless CNNs with low-precision weights.
      <https://arxiv.org/abs/1702.03044>`_

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - Input .
     -
   * - weight
     - Weight .
     - Parameter
   * - indicator_fixedweights
     - Indicates which weights are already fixed (0 = not fixed, 1 = fixed) .
     - Integer Parameter
   * - bias
     - Bias.
     - Optional Parameter

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - base_axis
     - int64
     - 1
     - Dimensions up to base_axis is treated as sample dimension.
   * - num_bits
     - int64
     - 4
     - Number of bits per weight. Needs to be >= 2 as two bits are used to code `zero` and sign of weight.
   * - inq_iterations
     - repeated int64
     - ()
     - List which specifies after how many forward passes we fix 50% of the learnable weights. If we have done as many iterations as specified in the last element of `inq_iterations`, then all weights are fixed.
   * - selection_algorithm
     - string ("largest_abs" or "random")
     - "largest_abs"
     - Chooses algorithm that we use for selecting the weights to fix ("largest_abs" ... fix weights with largest absolute value, "random" ... fix weights randomly)
   * - seed
     - int64
     - -1
     - Random seed. When -1, seed is sampled from global random number generator.

* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - Output.
     -

INQConvolution
^^^^^^^^^^^^^^

This function provides a INQ convolution layer. It computes in
the forward pass

.. math::

    y_{n, a, b} = \sum_{m} \sum_{i} \sum_{j} w_{n, m, i, j} x_{m, a + i, b + j},

where the weights :math:`w_{j,i}` are quantized sequentially during
training to power-of-two numbers. In the backward pass, only the non-fixed
(i.e., learnable) weights are updated.

Reference

    * `Zhou A, Yao A, Guo Y, Xu L, Chen Y. Incremental network quantization:
      Towards lossless CNNs with low-precision weights.
      <https://arxiv.org/abs/1702.03044>`_

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - Input.
     -
   * - weight
     - Weight.
     - Parameter
   * - indicator_fixedweights
     - Indicates which weights are already fixed (0 = not fixed, 1 = fixed) .
     - Integer Parameter
   * - bias
     - Bias.
     - Optional Parameter

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - base_axis
     - int64
     - 1
     - Dimensions up to base_axis is treated as sample dimension.
   * - pad
     - Shape
     - (0,) * (len(x.shape) - (base_axis+1))
     - Padding sizes for dimensions.
   * - stride
     - Shape
     - (1,) * (len(x.shape) - (base_axis+1))
     - Stride sizes for dimensions.
   * - dilation
     - Shape
     - (1,) * (len(x.shape) - (base_axis+1))
     - Dilation sizes for dimensions.
   * - group
     - int64
     - 1
     - Number of groups of channels. This makes the connection across channels sparser, by grouping connections along the mapping direction.
   * - num_bits
     - int64
     - 4
     - Number of bits per weight. Needs to be >= 2 as two bits are used to code `zero` and sign of weight.
   * - inq_iterations
     - repeated int64
     - ()
     - List which specifies after how many forward passes we fix 50% of the learnable weights. If we have done as many iterations as specified in the last element of `inq_iterations`, then all weights are fixed.
   * - selection_algorithm
     - string ("largest_abs" or "random")
     - "largest_abs"
     - Chooses algorithm that we use for selecting the weights to fix ("largest_abs" ... fix weights with largest absolute value, "random" ... fix weights randomly)
   * - seed
     - int64
     - -1
     - Random seed. When -1, seed is sampled from global random number generator.

* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - Output
     -

FixedPointQuantize
^^^^^^^^^^^^^^^^^^
This function uniformly quantizes values in fixed-point number representation.

In the forward pass, 

.. math::

   q_i= \left\{
	   \begin{array}{ll}
			max & if \ \ \ x_i > max \\
		  sign(x_i) \times floor(|x_i| \delta^{-1} + 2^{-1}) \times \delta & if \ \ min \le x_i \le max \\
	  	min & if \ \ x_i < min \\
	   \end{array} \right.,

where :math:`\delta` is the step size, 
:math:`(min, max) :=(- (2^{n-1} - 1)\delta, (2^{n-1} - 1)\delta)` if :math:`sign` is true, 
:math:`(min, max) := (0, (2^n - 1) \delta)` otherwise, and  
:math:`n` is the total bit-width used.

In the backward pass when using `ste_fine_grained` as false,  

.. math::

   \frac{\partial q_i}{\partial x_i} = 1.

In the backward pass when using `ste_fine_grained` as true,  

.. math::

   \frac{\partial q_i}{\partial x_i}= \left\{
	   \begin{array}{ll}
			0 & if \ \ \ x_i > max \\
		  1 & if \ \ min \le x_i \le max \\
	  	0 & if \ \ x_i < min \\
	   \end{array} \right..
   
.. note::


	Quantized values are stored as floating point number, since this function is for simulation purposes.
   

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - N-D array
     - 

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - sign
     - bool
     - True
     - Indicate the signed number or the unsigned number. Default is true.
   * - n
     - int64
     - 8
     - Bit width used. Note that `sign` consumes one bit. :math:`n-1` is used for number representation in `signed` case.   
   * - delta
     - float
     - 2**-4
     - Step size.
   * - ste_fine_grained
     - bool
     - True
     - Straight Through Estimator is fine-grained or not.
			 
* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - N-D array.
     -

Pow2Quantize
^^^^^^^^^^^^

This function quantizes values in the power of 2 number representation, 
in other words, it is linear (uniform) quantization in :math:`log_2` domain. 

In the forward pass of `signed` case,  

.. math::

   q_i= \left\{
	   \begin{array}{ll}
			max_{+} & if \ \ \overline{q_i} > max_{+} \\
			\overline{q_i} & if \ \ min_{+} \le \overline{q_i} \le max_{+} \\
		  min_{+} & if \ \ 0 \le \overline{q_i} < min_{+} \\
		  min_{-} & if \ \ min_{-} < \overline{q_i} < 0 \\
		  \overline{q_i} & if \ \ max_{-} \le \overline{q_i} \le min_{-}\\
	  	max_{-} & if \ \ \overline{q_i} < max_{-} \\
	   \end{array} \right.,

where 

.. math::
   
   && max_{+} = 2^{m}, min_{+} = 2^{m - (2^{n-1} - 1)},\\  
   && max_{-} = -2^{m}, min_{-} = -2^{m - (2^{n-1} - 1)},\\
   && \overline{q_i} = sign(x_i) \times 2^{round(\log_2 |x_i|)}.
   
This quantization uses the geometric mean between two power-of-two numbers 
as quantization threshold.   

In the forward pass of `unsigned` case,  

.. math::

   q_i= \left\{
	   \begin{array}{ll}
			max & if \ \ \overline{q_i} > max \\
			\overline{q_i} & if \ \ min \le \overline{q_i} \le max \\
		  min & if \ \ 0 < \overline{q_i} < min \\
	   \end{array} \right.,

where 

.. math::
   
   && max = 2^{m}, min = 2^{m - (2^{n} - 1)},\\  
   && \overline{q_i} = 2^{int(\log_2 |x_i|)}.
   
   
When using `with_zero` as true, a pruning threshold is used to round an input to 
0 or :math:`min`. The pruning threshold is defined in this function as the following, 

.. math::
   
   pruning\ threshold = min \times 2^{-\frac{1}{2}}.
   
If an absolute value of the input is lesser than this value, the input is rounded to 0, otherwise :math:`min`. 

In the backward pass when using ste_fine_grained as false,

.. math::

   \frac{\partial q_i}{\partial x_i} = 1.

In the backward pass when using ste_fine_grained as true,

.. math::

   \frac{\partial q_i}{\partial x_i}= \left\{
	   \begin{array}{ll}
			0 & if \ \ \overline{q_i} > max_{+} \\
			1 & if \ \ otherwise \\
	  	0 & if \ \ \overline{q_i} < max_{-} \\
	   \end{array} \right..

   
There are some literatures using pow2 quantization in their proposed methods. 
   
References:

  * `Miyashita Daisuke, Lee H. Edward, Murmann Boris. 
    Convolutional Neural Networks using Logarithmic Data Representation. 
    <https://arxiv.org/abs/1603.01025>`_

  * `Aojun Zhou, Anbang Yao, Yiwen Guo, Lin Xu, Yurong Chen.
    Incremental Network Quantization: Towards Lossless CNNs with Low-precision Weights.
    <https://arxiv.org/abs/1702.03044>`_
    
.. note::


	Quantized values are stored as floating point number, since this function is for simulation purposes.
   

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - N-D array
     - 

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - sign
     - bool
     - True
     - Indicate the signed number or the unsigned number. Default is true.
   * - with_zero
     - bool
     - True
     - Indicate using zero as a quantized value. Default is true. Note that `zero` consumes one bit.     
   * - n
     - int64
     - 8
     - Bit width used, Note that `sign` consumes one bit. :math:`n-1` is used for number representation in `signed` case. Default is 8.
   * - m
     - int64
     - 1
     - :math:`2^m` is the upper bound of the dynamic range and :math:`-2^m` is the lower bound, :math:`m \in \mathcal{Z}`. Default is 1.
   * - ste_fine_grained
     - bool
     - True
     - Straight Through Estimator is fine-grained or not.
			 
* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - N-D array.
     - 

Validation
----------

TopNError
^^^^^^^^^

Top N error along the dimension specified by the axis, the element of outputs is 

.. math::

    y_i = \left \{
    \begin{array}{l}
    1 \ (x_i \ is \ not \ within \ N-th \ place) \\
    0 \ (x_i \ is \ within \ N-th \ place)
    \end{array}
    \right.

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - Probabilities N-D array. :math:`D_1 \times ... \times D_i \times ... \times D_N`
     - 
   * - target
     - N-D array of labels. :math:`D_1 \times ... \times 1 \times ... \times D_N`
     - Integer

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - axis
     - int64
     - len(x.shape) - 1
     - Axis on which the top N error is calculated.
   * - n
     - int64
     - 1
     - top N

* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - output
     - Element-wise error N-D array. (:math:`D_1 \times ... \times 1 \times ... \times D_N`)
     - 

BinaryError
^^^^^^^^^

Elementwise binary error.

.. math::
    y_i = \left \{
    \begin{array}{l}
    0 ((x^{(0)} \geq 0.5) = (x^{(1)} \geq 0.5)) \\
    1 ((x^{(0)} \geq 0.5) \neq (x^{(1)} \geq 0.5))
    \end{array}
    \right.


* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - Probabilities N-D array. \f$-\infty\f$ to \f$\infty\f$.
     - 
   * - target
     - Labels N-D array. Usually set as 0 or 1, but, it allows probability (0 to 1) as inputs.
     - 

* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - output
     - Element-wise errors N-D array.
     - 

ConfusionMatrix
^^^^^^^^^

Confusion matrix.
The return value is already summed over samples.

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - Probabilities N-D array. (\f$D_1 \times ... \times D_i \times ... \times D_N\f$)
     - 
   * - target
     - Labels N-D array. (\f$D_1 \times ... \times 1 \times ... \times D_N\f$)
     - Integer

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - axis
     - int64
     - len(x.shape) - 1
     - Axis on which the confusion matrix is calculated.

* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - output
     - Confusion matrix 2-D array. Col index is estimated class. Row index is label class.
     - 

Unsupported, Special Use
------------------------

VATNoise
^^^^^^^^

Noise for virtual adversarial training.

This layer is a special layer for GUI network designing, specialized for getting
the noise of virtual adversarial training.

In the backward process, the weight parameter will be replaced with the gradient.

Forward

.. math::
    y_i = \frac{\epsilon x_i}{\sqrt{\sum_k x_k^2 + c}}

Backward

.. math::
    \delta x_i = 0

.. math::
    w_i = \epsilon \delta y_i

Note:
    This layer is a special layer for GUI network designing.

References:
    * `Miyato et.al, Distributional Smoothing with Virtual Adversarial Training.
      <https://arxiv.org/abs/1507.00677>`_

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - N-D array of noise input. Noise is standard Gaussian noise initially, but the next step, fed back gradient variable.
     - 
   * - w
     - N-D array for keep gradient values.
     - 

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - base_axis
     - int64
     - 1
     - Dimensions up to base_axis is treated as sample dimension.
   * - eps
     - float
     - 1.0
     - Noise norm (l2) factor.


* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - N-D array
     - 

Unlink
^^^^^^

This function behaves as an identity function on the forward pass,
and deletes the gradient for the background pass.

This layer is a special layer for GUI network designing, used for getting
zero backward operation by adding this layer.

Forward

.. math::
    y_i = x_i

Backward

.. math::
    \delta x_i = 0

Note:
    This layer is a special layer for GUI network designing.

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - N-D array.
     - 

* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - N-D array.
     - 

Sink
^^^^

Creates a dummy variable used to call forward or backward function
of multiple variables at one place.

This takes any numbers of input variables with any shape,
and creates a single 0-shape outputs.
The forward pass does nothing. The backward pass set ones
to the input grads if one_input_grad is set as true.

* Input(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - x
     - Any number of inputs with any shape.
     - Variadic

* Argument(s)

.. list-table::

   * - Name
     - Type
     - Default
     - Description
   * - one_input_grad
     - bool
     - True
     - Set grads of inputs as one during backward. It is useful to set false if you want to set external gradients to the input variables.

* Output(s)

.. list-table::

   * - Name
     - Description
     - Options
   * - y
     - Dummy variable.
     - 
