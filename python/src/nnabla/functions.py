# Copyright (c) 2017 Sony Corporation. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from .function_bases import *


def sum(x, axis=None, keepdims=False):
    """Reduction along axes with sum operation.

    Args:
        x (Variable): An input variable.
        axis (None, int or tuple of ints): Axis or axes along which the sum is
            calculated. Passing the default value `None` will reduce all dimensions.
        keepdims (bool): Flag whether the reduced axes are kept as a dimension with 1 element.

    Returns:
        ~nnabla.Variable: N-D array.
    """
    from .function_bases import sum as sum_base
    if axis is None:
        axis = range(x.ndim)
    elif not hasattr(axis, '__iter__'):
        axis = [axis]
    return sum_base(x, axis, keepdims)


def mean(x, axis=None, keepdims=False):
    """Reduction along axes with mean operation.

    Args:
        x (Variable): An input variable.
        axis (None, int or tuple of ints): Axis or axes along which mean is
            calculated. Passing the default value `None` will reduce all dimensions.
        keepdims (bool): Flag whether the reduced axes are kept as a dimension with 1 element.

    Returns:
        ~nnabla.Variable: N-D array.

    """
    from .function_bases import mean as mean_base
    if axis is None:
        axis = range(x.ndim)
    elif not hasattr(axis, '__iter__'):
        axis = [axis]
    return mean_base(x, axis, keepdims)


def max(x, axis=None, keepdims=False):
    """Reduction along axes with max operation.

    Args:
        x (Variable): An input variable.
        axis (None, int or tuple of ints): Axis or axes along which max is
            calculated. Passing the default value `None` will reduce all dimensions.
        keepdims (bool): Flag whether the reduced axes are kept as a dimension with 1 element.

    Returns:
        ~nnabla.Variable: N-D array.

    """
    from .function_bases import max as max_base
    if axis is None:
        axis = range(x.ndim)
    elif not hasattr(axis, '__iter__'):
        axis = [axis]
    return max_base(x, axis, keepdims)


def min(x, axis=None, keepdims=False):
    """Reduction along axes with min operation.

    Args:
        x (Variable): An input variable.
        axis (None, int or tuple of ints): Axis or axes along which min is
            calculated. Passing the default value `None` will reduce all dimensions.
        keepdims (bool): Flag whether the reduced axes are kept as a dimension with 1 element.

    Returns:
        ~nnabla.Variable: N-D array.

    """
    from .function_bases import min as min_base
    if axis is None:
        axis = range(x.ndim)
    elif not hasattr(axis, '__iter__'):
        axis = [axis]
    return min_base(x, axis, keepdims)


def prod(x, axis=None, keepdims=False):
    """Reduction along axes with product operation.

    Args:
        x (Variable): An input variable.
        axis (None, int or tuple of ints): Axis or axes along which product is
            calculated. Passing the default value `None` will reduce all dimensions.
        keepdims (bool): Flag whether the reduced axes are kept as a dimension with 1 element.

    Returns:
        ~nnabla.Variable: N-D array.

    Note:
        Backward computation is not accurate in a zero value input.

    """
    from .function_bases import prod as prod_base
    if axis is None:
        axis = range(x.ndim)
    elif not hasattr(axis, '__iter__'):
        axis = [axis]
    return prod_base(x, axis, keepdims)


def reduce(x, op='sum'):
    """Reduction function with given operation.

    Args:
        x (Variable): An input.
        op (str): 'sum' or 'mean'.

    Note:
        This is deprecated. Use ``mean`` or ``sum`` instead.

    """
    import warnings
    warnings.warn(
        "Deprecated API. Use ``sum`` or ``mean`` instead.", DeprecationWarning)
    from .function_bases import reduce_sum, reduce_mean
    if op == 'sum':
        return reduce_sum(x)
    elif op == 'mean':
        return reduce_mean(x)
    raise ValueError()


def split(x, axis=0):
    """
    Split arrays at the specified axis.

    It returns a number corresponding the size of the given
    axis (i.e ``x.shape[axis]``) of :obj:`~nnabla.Variable` s.

    Args:
        x(~nnabla.Variable): N-D array
        axis(int): Axis

    Returns: A :obj:`tuple` of :obj:`~nnabla.Variable` s

    See Also:
        :func:`nnabla.function_bases.split`.

    """
    from .function_bases import split as split_base
    return split_base(x, axis, x.shape[axis])


def batch_normalization(x, beta, gamma, mean, variance, axes=[1], decay_rate=0.9, eps=1e-05, batch_stat=True, output_stat=False, n_outputs=None):
    r"""
    Batch normalization.

    .. math::
        \begin{eqnarray}
          \mu &=& \frac{1}{M} \sum x_i \\
          \sigma^2 &=& \frac{1}{M} \left(\sum x_i - \mu\right)^2 \\
          \hat{x}_i &=& \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} \\
          y_i &=& \hat{x}_i \gamma + \beta.
        \end{eqnarray}


    At testing time, the mean and variance values used are those that were computed during training by moving average.

    References:

        * `Ioffe and Szegedy, Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift.
          <https://arxiv.org/abs/1502.03167>`_

    Args:
        x(~nnabla.Variable): N-D array of input.
        beta(~nnabla.Variable): N-D array of beta which is learned.
        gamma(~nnabla.Variable): N-D array of gamma which is learned.
        mean(~nnabla.Variable): N-D array of running mean (modified during forward execution).
        variance(~nnabla.Variable): N-D array of running variance (modified during forward execution).
        axes(repeated int64): Axes mean and variance are taken.
        decay_rate(float): Decay rate of running mean and variance.
        eps(float): Tiny value to avoid zero division by std.
        batch_stat(bool): Use mini-batch statistics rather than running ones.
        output_stat(bool): It true, the batch statistics of mean and variance,
            will be returned as Variables. They are also differentiable.

    Returns:
        Retruns batch normalization output as :obj:`~nnabla.Variable`.
        If ``output_stat=True``, it also returns the mean and variance
        of the mini-batch

        * :obj:`~nnabla.Variable`: Output of the batch normalization
        * :obj:`~nnabla.Variable`: Mean (if ``output_stat=True`)
        * :obj:`~nnabla.Variable`: Variance (if ``output_stat=True`)

    See Also:
        ``nnabla.function_bases.batch_normalization``.

    """
    from .function_bases import batch_normalization as batch_normalization_base
    n_outputs = 3 if output_stat else 1
    assert batch_stat or (not output_stat)
    return batch_normalization_base(x, beta, gamma, mean, variance,
                                    axes=axes,
                                    decay_rate=decay_rate,
                                    eps=eps,
                                    batch_stat=batch_stat,
                                    n_outputs=n_outputs)


def fixed_point_quantize(x, sign=True, n=8, delta=2**-4, quantize=True, ste_fine_grained=True, outputs=None):
    r"""Fixed Point Quantize

    Args:
        x (Variable): An input variable.
        sign (bool): Indicate the signed number or the unsigned number. Default is true.
        n (int): Bit width used. Note that `sign` consumes one bit. :math:`n-1` is used for number representation in `signed` case.   
        delta (float): Step size.
        quantize (bool): If true, quantize input, otherwise not.
        ste_fine_grained (bool): If true, STE is not 1.

    Returns:
        ~nnabla.Variable: N-D array.

    See Also:
        ``nnabla.function_bases.fixed_point_quantize``.

    In the forward pass, 
    
    .. math::
    
        \begin{equation}
            q_i= \left\{
               \begin{array}{ll}
                    max & if \ \ \ x_i > max \\
                    sign(x_i) \times floor(|x_i| \delta^{-1} + 2^{-1}) \times \delta & if \ \ min \le x_i \le max \\
                    min & if \ \ x_i < min \\
               \end{array} \right.,
        \end{equation}
    
    where :math:`\delta` is the step size, 
    :math:`(min, max) :=(- (2^{n-1} - 1)\delta, (2^{n-1} - 1)\delta)` if :math:`sign` is true, 
    :math:`(min, max) := (0, (2^n - 1) \delta)` otherwise, and  
    :math:`n` is the total bit-width used.
    
    In the backward pass when using `ste_fine_grained` as false,  
    
    .. math::
    
        \begin{equation}
            \frac{\partial q_i}{\partial x_i} = 1.
        \end{equation}
    
    In the backward pass when using `ste_fine_grained` as true,  
    
    .. math::
    
        \begin{equation}
            \frac{\partial q_i}{\partial x_i}= \left\{
        	    \begin{array}{ll}
        			0 & if \ \ \ x_i > max \\
        		    1 & if \ \ min \le x_i \le max \\
        	  	    0 & if \ \ x_i < min \\
        	    \end{array} \right..
        \end{equation}
       
    .. note::
        
        Quantized values are stored as floating point number, since this function is for simulation purposes.
   
    """
    from .function_bases import fixed_point_quantize as fixed_point_quantize_base
    if not quantize:
        return x
    return fixed_point_quantize_base(x, sign, n, delta, ste_fine_grained, outputs=outputs)

