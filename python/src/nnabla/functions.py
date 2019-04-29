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
from six.moves import reduce as rd
import numpy as np


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


def max(x, axis=None, keepdims=False, with_index=False, only_index=False):
    """Reduce the input N-D array `x` along the given `axis` using the max
    operation. The `axis` argument may be a single integer to reduce
    over one axis, a tuple of integers to reduce over multiple axes,
    or ``None`` to reduce over all axes. If `keepdims` is ``True``,
    the output will keep all reduced dimensions with size 1. If
    `with_index` is True, result is a tuple ``(sorted, indices)`` or
    only ``indices`` if `only_index` is True. Setting `only_index` to
    True implies that `with_index` is also True.

    .. code-block:: python

        import numpy as np
        import nnabla as nn
        import nnabla.functions as F

        nn.set_auto_forward(True)
        x = nn.Variable.from_numpy_array(np.random.rand(2, 3, 4))

        maxval = F.max(x, axis=1)
        assert np.allclose(maxval.d, np.max(x.d, axis=1))

        maxval, indices = F.max(x, axis=1, with_index=True)
        assert np.allclose(maxval.d, np.max(x.d, axis=1))
        assert np.all(indices.d == np.argmax(x.d, axis=1))

        indices = F.max(x, axis=1, only_index=True)
        assert np.all(indices.d == np.argmax(x.d, axis=1))

    Args:
        x (Variable): An input variable.
        axis (None, int or tuple of ints): Axis or axes along which max is
            calculated. The default value `None` will reduce all dimensions.
        keepdims(bool): Keep reduced axes as dimension with 1 element.
        with_index(bool): Return tuple of max values and index.
        only_index(bool): Return only the index of max values.

    Returns:
        ~nnabla.Variable: N-D array.

    """
    from .function_bases import max as max_base
    if axis is None:
        axis = range(x.ndim)
    elif not hasattr(axis, '__iter__'):
        axis = [axis]
    n_outputs = 2 if with_index and not only_index else 1
    return max_base(x, axis, keepdims, with_index, only_index, n_outputs)


def min(x, axis=None, keepdims=False, with_index=False, only_index=False):
    """Reduce the input N-D array `x` along the given `axis` using the min
    operation. The `axis` argument may be a single integer to reduce
    over one axis, a tuple of integers to reduce over multiple axes,
    or ``None`` to reduce over all axes. If `keepdims` is ``True``,
    the output will keep all reduced dimensions with size 1. If
    `with_index` is True, result is a tuple ``(sorted, indices)`` or
    only ``indices`` if `only_index` is True. Setting `only_index` to
    True implies that `with_index` is also True.

    .. code-block:: python

        import numpy as np
        import nnabla as nn
        import nnabla.functions as F

        nn.set_auto_forward(True)
        x = nn.Variable.from_numpy_array(np.random.rand(2, 3, 4))

        minval = F.min(x, axis=1)
        assert np.allclose(minval.d, np.min(x.d, axis=1))

        minval, indices = F.min(x, axis=1, with_index=True)
        assert np.allclose(minval.d, np.min(x.d, axis=1))
        assert np.all(indices.d == np.argmin(x.d, axis=1))

        indices = F.min(x, axis=1, only_index=True)
        assert np.all(indices.d == np.argmin(x.d, axis=1))

    Args:
        x (Variable): An input variable.
        axis (None, int or tuple of ints): Axis or axes along which min is
            calculated. The default value `None` will reduce all dimensions.
        keepdims(bool): Keep reduced axes as dimension with 1 element.
        with_index(bool): Return tuple of min values and index.
        only_index(bool): Return only the index of min values.

    Returns:
        ~nnabla.Variable: N-D array.

    """
    from .function_bases import min as min_base
    if axis is None:
        axis = range(x.ndim)
    elif not hasattr(axis, '__iter__'):
        axis = [axis]
    n_outputs = 2 if with_index and not only_index else 1
    return min_base(x, axis, keepdims, with_index, only_index, n_outputs)


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


@function_api
def slice(ctx, x, start=None, stop=None, step=None, n_outputs=-1, outputs=None):
    r"""
    Slice arrays along specified axis. This function 
    complies with python slice wherre `slice(None, None, -1)` and 
    `slice(-1, None, -1)` are the special case, which flips the 
    input array and results in the output array from the end to the beginning
    of the input array along the corresponding dimension.


    Args:
        x(~nnabla.Variable): N-D array
        start(repeated int64): Start indices for each axis
            [default=``(0,) * len(x.shape)``]
        stop(repeated int64): Stop indices for each axis
            [default=``tuple(x.shape)``]
        step(repeated int64): Step indices for each axis
            [default=``(1,) * len(x.shape)``]

    Returns:
        ~nnabla.Variable: Sliced N-D array
    """
    import copy
    start = copy.copy(start)
    stop = copy.copy(stop)
    step = copy.copy(step)

    from .function_bases import slice as slice_base
    if start is None:
        start = (0,) * len(x.shape)
    if stop is None:
        stop = tuple(x.shape)
    if step is None:
        step = (1,) * len(x.shape)

    shape = x.shape
    for i, sss in enumerate(zip(start, stop, step)):
        s0, s1, s2 = sss
        # SPECIAL CASE: slice(-1, None, <0) or slice(None, None, <0)
        SLICE_NONE = 0x7fffffff
        if s0 == None:
            start[i] = SLICE_NONE
        if s1 == None:
            stop[i] = SLICE_NONE
        if s2 == None:
            step[i] = SLICE_NONE
    return slice_base(x, start, stop, step, n_outputs, outputs)


def batch_normalization(x, beta, gamma, mean, variance, axes=[1], decay_rate=0.9, eps=1e-05, batch_stat=True, output_stat=False, n_outputs=None):
    r"""
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
        Returns batch normalization output as :obj:`~nnabla.Variable`.
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
    if batch_stat and (mean.parent or variance.parent) is not None:
        raise ValueError(
            "if batch_stat is True, mean and variable must not have a parent function")

    if len(axes) == 1:
        return batch_normalization_base(x, beta, gamma, mean, variance,
                                        axes=axes,
                                        decay_rate=decay_rate,
                                        eps=eps,
                                        batch_stat=batch_stat,
                                        n_outputs=n_outputs)

    def transpose_and_reshape(x, axes):
        transposed = transpose(x, transpose_axes)
        return reshape(transposed, [rd(lambda x, y: x * y, transposed.shape[:len(axes)])] + list(
            transposed.shape[len(axes):])), transposed.shape

    def inverse_transpose_and_reshape(x, axes, variable_shape):
        un_reshaped = reshape(
            x, list(variable_shape[:len(axes)] + variable_shape[len(axes):]))
        return transpose(un_reshaped, inv_transpose_axes)

    def get_tranpose_args(ndim, axes):
        transpose_axes = [i for i in list(
            axes)] + [i for i in range(ndim) if i not in list(axes)]
        inv_transpose_axes = np.argsort(transpose_axes).tolist()
        return transpose_axes, inv_transpose_axes

    transpose_axes, inv_transpose_axes = get_tranpose_args(len(x.shape), axes)
    inp, transposed_inp_shape = transpose_and_reshape(x, axes)
    beta, transposed_beta_shape = transpose_and_reshape(beta, axes)
    gamma, transposed_gamma_shape = transpose_and_reshape(gamma, axes)
    mean, transposed_mean_shape = transpose_and_reshape(mean, axes)
    variance, transposed_variance_shape = transpose_and_reshape(variance, axes)

    if n_outputs == 1:
        out = batch_normalization_base(inp, beta, gamma, mean, variance,
                                       axes=[0],
                                       decay_rate=decay_rate,
                                       eps=eps,
                                       batch_stat=batch_stat,
                                       n_outputs=n_outputs)
        return inverse_transpose_and_reshape(out, axes, transposed_inp_shape)
    out, mean, variance = batch_normalization_base(inp, beta, gamma, mean, variance,
                                                   axes=[0],
                                                   decay_rate=decay_rate,
                                                   eps=eps,
                                                   batch_stat=batch_stat,
                                                   n_outputs=n_outputs)
    out = inverse_transpose_and_reshape(out, axes, transposed_inp_shape)
    mean = inverse_transpose_and_reshape(mean, axes, transposed_mean_shape)
    variance = inverse_transpose_and_reshape(
        variance, axes, transposed_variance_shape)
    return out, mean, variance


def sync_batch_normalization(x, beta, gamma, mean, variance, comm, group="world", axes=[1], decay_rate=0.9, eps=1e-05, batch_stat=True, output_stat=False, n_outputs=None):
    r"""
    Synchronized batch normalization.

    For some tasks (e.g., semantic segmentation), batch size will be too small and BatchNormalization layer might not work well.
    SyncBatchNorlization layer solves these problems by synchronizing batch stats (mean and var) between multiple processes.

    .. math::
        \begin{eqnarray}
          \mu &=& \frac{1}{M} \sum x_i \\
          \sigma^2 &=& \frac{1}{M} \left(\sum x_i - \mu\right)^2 \\
          \hat{x}_i &=& \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} \\
          y_i &=& \hat{x}_i \gamma + \beta.
        \end{eqnarray}

    References:

        * Implementing Synchronized Multi-GPU Batch Normalization https://hangzhang.org/PyTorch-Encoding/notes/syncbn.html

    Args:
        x(~nnabla.Variable): N-D array of input.
        beta(~nnabla.Variable): N-D array of beta which is learned.
        gamma(~nnabla.Variable): N-D array of gamma which is learned.
        mean(~nnabla.Variable): N-D array of running mean (modified during forward execution).
        variance(~nnabla.Variable): N-D array of running variance (modified during forward execution).
        comm(~nnabla.communicators.Communicator): The communicator
        group(string): The name of the communicator group
        axes(repeated int64): Axes mean and variance are taken.
        decay_rate(float): Decay rate of running mean and variance.
        eps(float): Tiny value to avoid zero division by std.
        batch_stat(bool): Use mini-batch statistics rather than running ones.
        output_stat(bool): It true, the batch statistics of mean and variance,
            will be returned as Variables. They are also differentiable.

    Returns:
        Returns batch normalization output as :obj:`~nnabla.Variable`.
        If ``output_stat=True``, it also returns the mean and variance
        of the mini-batch

        * :obj:`~nnabla.Variable`: Output of the batch normalization
        * :obj:`~nnabla.Variable`: Mean (if ``output_stat=True`)
        * :obj:`~nnabla.Variable`: Variance (if ``output_stat=True`)

    See Also:
        ``nnabla.function_bases.batch_normalization``.

    """
    from .function_bases import sync_batch_normalization as batch_normalization_base
    n_outputs = 3 if output_stat else 1
    return batch_normalization_base(x, beta, gamma, mean, variance,
                                    comm, group=group,
                                    axes=axes,
                                    decay_rate=decay_rate,
                                    eps=eps,
                                    batch_stat=batch_stat,
                                    n_outputs=n_outputs)


def mean_subtraction(x, mean, t, base_axis=1, update_running_mean=True):
    r"""
    It subtracts the mean of the elements of the input array,
    and normalizes it to :math:`0`. Preprocessing arrays with this function has the effect of improving accuracy
    in various tasks such as image classification.

    At training time, this function is defined as

    .. math::
        \begin{eqnarray}
          \mu &=& \frac{1}{M} \sum x_i \\
          y_i &=& x_i - \mu
        \end{eqnarray}

    At testing time, the mean values used are those that were computed during training by moving average.

    Note:
        The backward performs an approximated differentiation that takes into account only the latest mini-batch.

    Args:
        x(~nnabla.Variable): N-D array of input.
        mean(~nnabla.Variable): N-D array of running mean (modified during forward execution).
        t(~nnabla.Variable): Scalar of num of iteration of running mean (modified during forward execution).
        base_axis(int): Base axis of Mean Subtraction operation. Dimensions up to base_axis is treated as sample dimension.
            [default=``1``]
        update_running_mean(bool): Update running mean during forward execution.
            [default=``True``]

    Returns:
        ~nnabla.Variable: N-D array.

    See Also:
        ``nnabla.function_bases.mean_subtraction``.

    """
    from .function_bases import mean_subtraction as mean_subtraction_base
    return mean_subtraction_base(x, mean, t,
                                 base_axis=base_axis,
                                 update_running_mean=update_running_mean)


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


def pow2_quantize(x, sign=True, with_zero=True, n=8, m=1, quantize=True, ste_fine_grained=True, outputs=None):
    r"""Pow2 Quantize

    Args:
        x (Variable): An input variable.
        sign (bool): Indicate the signed number or the unsigned number. Default is true.
        with_zero (bool): Indicate using zero as a quantized value. Default is true. Note that `zero` consumes one bit.
        n (int): Bit width used. Note that `sign` consumes one bit. :math:`n-1` is used for number representation in `signed` case. Default is 8.   
        m (int): :math:`2^m` is the upper bound of the dynamic range and :math:`-2^m` is the lower bound, :math:`m \in \mathcal{Z}`. Default is 1.
        quantize (bool): If true, quantize input, otherwise not.
        ste_fine_grained (bool): If true, STE is not 1.

    Returns:
        ~nnabla.Variable: N-D array.

    See Also:
        ``nnabla.function_bases.pow2_quantize``.

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

    """

    from .function_bases import pow2_quantize as pow2_quantize_base
    if not quantize:
        return x
    return pow2_quantize_base(x, sign, with_zero, n, m, ste_fine_grained, outputs=outputs)


def clip_by_value(x, min, max):
    r"""Clip inputs by values.

    .. math::

        y = \begin{cases}
                max & (x > max) \\
                x & (otherwise) \\
                min & (x < min)
            \end{cases}.

    Args:
        x (Variable): An input variable.
        min (Variable): A min variable by which `x` is clipped. Note that the shape of `min` must be the same as `x`'s.
        max (Variable): A max variable by which `x` is clipped. Note that the shape of `max` must be the same as `x`'s

    Returns:
        ~nnabla.Variable: N-D array.

    """
    from .function_bases import maximum2 as maximum2_base
    from .function_bases import minimum2 as minimum2_base
    return minimum2_base(maximum2_base(x, min), max)


def clip_by_norm(x, clip_norm, axis=None):
    r"""
    Clip inputs by its L2 norm when the L2 norm is larger than the threshold value (defined by clip_norm).
    If it is less than the threshold, inputs are not modified. If it is applied, the operation is represented as

    .. math::
      y = N \times \frac{x}{\|x\|_2}.

    where :math:`x` is the input, :math:`y` is the output,
    and :math:`N` is `clip_norm`. this is the case that `axes` is not set.
    When `axes` is set, the norm is computed over `axes`.

    Args:
        x (Variable): An input variable.
        clip_norm (`Variable` or `float`): An input scalar variable or float value. Must be positive.
        axis (None, int or tuple of ints): Axis or axes along which the reduction is performed. Passing the default value `None` will reduce all dimensions.

    Returns:
        ~nnabla.Variable: N-D array.

    """
    from .function_bases import pow_scalar as pow_scalar_base
    from .function_bases import maximum2 as maximum2_base
    from .function_bases import maximum_scalar as maximum_scalar_base
    from .function_bases import sum as sum_base
    from ._variable import Variable as Variable_base
    from ._nd_array import NdArray as NdArray_base

    if axis is None:
        axis = range(x.ndim)
    elif not hasattr(axis, '__iter__'):
        axis = [axis]
    x_norm = pow_scalar_base(sum_base(x**2.0, axis, True), 0.5)
    if isinstance(clip_norm, (Variable_base, NdArray_base)):
        y = x * clip_norm / maximum2_base(x_norm, clip_norm)
    else:
        if clip_norm <= 0:
            raise ValueError("clip_norm must be positive.")
        y = x * clip_norm / maximum_scalar_base(x_norm, clip_norm)
    return y


def interpolate(x, scale=None, output_size=None, mode='linear', align_corners=None):
    '''
    Resize an ND array with interpolation.

    Scaling factors for spatial dimensions are determined by either
    ``scale`` or ``output_size``.

    ``nd = len(scale)`` or ``nd = len(output_size)`` determines the number of
    spatial dimensions, and the last ``nd`` dimensions of the input ``x`` are    
    considered as the spatial dimensions to be resized.


    If ``scale`` is given, the ``output_size`` is calculated by
    ``output_size[i] = floor(scale[i] * x.shape[i - len(scale)])``.

    Example:

    .. code-block:: python

        import numpy as np
        import nnabla as nn
        import nnabla.functions as F

        x_data = np.random.rand(64, 3, 224, 224)
        x = nn.Variable.from_numpy_array(x_data)

        # Resize by scales
        y = F.interpolate(x, scale=(2, 2), mode='linear')
        print(y.shape)  # (64, 3, 448, 448)
        y.forward()
        print(y.d)  # Print output

        # Resize to a size
        y2 = F.interpolate(x, output_size=(320, 257), mode='linear')
        print(y2.shape)  # (64, 3, 320, 257)
        y2.forward()
        print(y2.d)  # Print output

    Args:
        x(~nnabla.Variable): N-D array with an arbitrary number of dimensions.
        scale(tuple of ints): Scale factors along axes. The default is
            ``None``, and if this is omitted, ``output_size`` must be specified.
        output_size(tuple of ints): The output sizes for axes. If this is
            given, the scale factors are determined by the output sizes and the
            input sizes. The default is ``None``, and if this is omitted,
            ``scale`` must be specified.
        mode(str): Interpolation mode chosen from ('linear'|'nearest').
            The default is 'linear'.
        align_corners(bool): If true, the corner pixels of input and output
            arrays are aligned, such that the output corner pixels have the
            same values with the input corner pixels.
            The default is ``None``, and it becomes ``True`` if mode is
            'linear', otherwise ``False``.

    Returns:
        ~nnabla.Variable: N-D array.

    '''
    from .function_bases import interpolate as interpolate_base
    import math
    if scale is None and output_size is None:
        raise ValueError('Either scale or output_size must be given')
    elif output_size is None:
        output_size = [int(math.floor(s * d))
                       for d, s in zip(x.shape[-len(scale):], scale)]
    if align_corners is None:
        if mode == 'linear':
            align_corners = True
        else:
            align_corners = False
    return interpolate_base(x, output_size, mode, align_corners)


def sort(x, axis=-1, reverse=False, with_index=False, only_index=False):
    """Sorts the elements of `x` along a given `axis` in ascending order
    by value. A negative `axis` counts from the last dimension of `x`,
    so the default of -1 sorts along the last dimension. If `reverse`
    is True, then the elements are soreted in descending order.

    If `with_index` is True, result is a tuple ``(sorted, indices)``
    or only ``indices`` if `only_index` is True. Setting `only_index`
    to True implies that `with_index` is also True.

    .. code-block:: python

        import numpy as np
        import nnabla as nn
        import nnabla.functions as F

        nn.set_auto_forward(True)
        x = nn.Variable.from_numpy_array(np.random.rand(2, 3, 4))

        sorted = F.sort(x)
        assert np.allclose(sorted.d, np.sort(x.d))

        sorted, indices = F.sort(x, with_index=True)
        assert np.allclose(sorted.d, np.sort(x.d))
        assert np.all(indices.d == np.argsort(x.d))

        indices = F.sort(x, only_index=True)
        assert np.all(indices.d == np.argsort(x.d))

    Args:
        x(~nnabla.Variable): N-D array
        axis(int): Axis along which to sort.
        reverse(bool): Sort in descending order.
        with_index(bool): Return sorted values and index.
        only_index(bool): Return only the sort index.

    Returns: :obj:`~nnabla.Variable` `sorted` or :obj:`~nnabla.Variable` `indices` or (:obj:`~nnabla.Variable` `sorted`, :obj:`~nnabla.Variable` `indices`)

    """
    from .function_bases import sort as sort_base
    n_outputs = 2 if with_index and not only_index else 1
    return sort_base(x, axis, reverse, with_index, only_index, n_outputs)


def tile(x, reps):
    """Forward `x` repeated the number of times given by `reps`. If `reps` is
    a sequence, the output has dimension of ``d = max(len(reps), x.ndim)`` and
    either `x` is promoted to be d-dimensional by prepending new axes or `reps`
    is promoted to x.ndim by prepending 1's.

    Args:
        x(~nnabla.Variable): Input N-D array.
        reps(int or sequence of int): Repetitions of `x` along each axis.

    Returns:
        ~nnabla.Variable: N-D array.

    >>> import numpy as np, nnabla as nn, nnabla.functions as F
    >>> F.tile(nn.Variable([2, 3], 3).shape    # reps is promoted to [1, 3]
    (2, 9)
    >>> F.tile(nn.Variable([3], [2, 3]).shape  # x is promoted to shape (1, 3)
    (2, 9)
    >>> nn.set_auto_forward(True)
    >>> x = nn.Variable.from_numpy_array(np.array([1, 2, 3]))
    >>> print(F.tile(x, 3).d)
    [1. 2. 3. 1. 2. 3. 1. 2. 3.]
    >>> print(F.tile(x, [2, 3]).d)
    [[1. 2. 3. 1. 2. 3. 1. 2. 3.]
     [1. 2. 3. 1. 2. 3. 1. 2. 3.]]
    >>> x = nn.Variable.from_numpy_array(np.array([[1, 3], [2, 4]]))
    >>> print(F.tile(x, 3).d)
    [[1. 3. 1. 3. 1. 3.]
     [2. 4. 2. 4. 2. 4.]]
    >>> print(F.tile(x, [2, 3]).d)
    [[1. 3. 1. 3. 1. 3.]
     [2. 4. 2. 4. 2. 4.]
     [1. 3. 1. 3. 1. 3.]
     [2. 4. 2. 4. 2. 4.]]

    """
    from .function_bases import tile as tile_base
    reps = [reps] if isinstance(reps, int) else reps
    return tile_base(x, reps)
