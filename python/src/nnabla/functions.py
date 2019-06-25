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

import nnabla as nn
import numpy as np
from .normalization_functions import *


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
    start = list(start[:]) if start is not None else len(x.shape) * (0,)
    stop = list(stop[:]) if stop is not None else tuple(x.shape)
    step = list(step[:]) if step is not None else len(x.shape) * (1,)

    for i, (s0, s1, s2) in enumerate(zip(start, stop, step)):
        # SPECIAL CASE: slice(-1, None, <0) or slice(None, None, <0)
        SLICE_NONE = 0x7fffffff
        if s0 == None:
            start[i] = SLICE_NONE
        if s1 == None:
            stop[i] = SLICE_NONE
        if s2 == None:
            step[i] = SLICE_NONE

    from .function_bases import slice as slice_base
    return slice_base(x, start, stop, step, n_outputs, outputs)


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

    Returns: ~nnabla.Variable `sorted` or ~nnabla.Variable `indices` or (~nnabla.Variable `sorted`, ~nnabla.Variable `indices`)

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


def stft(x, window_size, stride, fft_size, window_type='hanning', center=True, pad_mode='reflect'):
    """Computes the short-time Fourier transform

    Args:
        x (~nnabla.Variable): Time domain sequence of size `batch_size x sample_size`.
        window_size (int): Size of STFT analysis window.
        stride (int): Number of samples that we shift the window, also called `hop size`.
        fft_size (int): Size of the FFT, the output will have `fft_size // 2+ 1` frequency bins.
        window_type (str): Analysis window, can be either `hanning` or `hamming`.
        center (bool): If `True`, then the signal `x` is padded by half the FFT size using reflection padding.
        pad_mode (str): Padding mode, which can be `'constant'` or `'reflect'`. `'constant'` pads with `0`.

    Returns:
        Returns real and imaginary parts of STFT result.

        * :obj:`~nnabla.Variable`: Real part of STFT of size `batch_size x fft_size//2 + 1 x frame_size`.
        * :obj:`~nnabla.Variable`: Imaginary part of STFT of size `batch x fft_size//2 + 1 x frame_size`.
    """
    from nnabla.parameter import get_parameter, get_parameter_or_create
    conv_r = get_parameter('conv_r')
    conv_i = get_parameter('conv_i')

    if conv_r is None or conv_i is None:
        if window_type == 'hanning':
            window_func = np.hanning(window_size + 1)[:-1]
        elif window_type == 'hamming':
            window_func = np.hamming(window_size + 1)[:-1]
        else:
            raise ValueError("Unknown window type {}.".format(window_type))

        # pad window if `fft_size > window_size`
        if fft_size > window_size:
            diff = fft_size - window_size
            window_func = np.pad(
                window_func, (diff//2, diff - diff//2), mode='constant')
        elif fft_size < window_size:
            raise ValueError(
                "FFT size has to be as least as large as window size.")

        # compute STFT filter coefficients
        mat_r = np.zeros((fft_size//2 + 1, 1, fft_size))
        mat_i = np.zeros((fft_size//2 + 1, 1, fft_size))

        for w in range(fft_size//2+1):
            for t in range(fft_size):
                mat_r[w, 0, t] = np.cos(2. * np.pi * w * t / fft_size)
                mat_i[w, 0, t] = -np.sin(2. * np.pi * w * t / fft_size)
        mat_r = mat_r * window_func
        mat_i = mat_i * window_func

        conv_r = get_parameter_or_create(
            'conv_r', initializer=mat_r, need_grad=False)
        conv_i = get_parameter_or_create(
            'conv_i', initializer=mat_i, need_grad=False)

    if center:
        # pad at begin/end (per default this is a reflection padding)
        x = pad(x, (fft_size // 2, fft_size // 2), mode=pad_mode)

    # add channel dimension
    x = reshape(x, (x.shape[0], 1, x.shape[1]), inplace=False)

    # compute STFT
    y_r = convolution(x, conv_r, stride=(stride,))
    y_i = convolution(x, conv_i, stride=(stride,))

    return y_r, y_i


def istft(y_r, y_i, window_size, stride, fft_size, window_type='hanning', center=True):
    """Computes the inverse shoft-time Fourier transform

    Note: We use a constant square inverse window for the reconstruction
    of the time-domain signal, therefore, the first and last
    `window_size - stride` are not perfectly reconstructed.

    Args:
        y_r (~nnabla.Variable): Real part of STFT of size `batch_size x fft_size//2 + 1 x frame_size`.
        y_i (~nnabla.Variable): Imaginary part of STFT of size `batch_size x fft_size//2 + 1 x frame_size`.
        window_size (int): Size of STFT analysis window.
        stride (int): Number of samples that we shift the window, also called `hop size`.
        fft_size (int): Size of the FFT, (STFT has `fft_size // 2 + 1` frequency bins).
        window_type (str): Analysis window, can be either `hanning` or `hamming`.
        center (bool): If `True`, then it is assumed that the time-domain signal has centered frames.

    Returns:
        ~nnabla.Variable: Time domain sequence of size `batch_size x sample_size`.
    """
    from nnabla.parameter import get_parameter, get_parameter_or_create
    conv_cos = get_parameter('conv_cos')
    conv_sin = get_parameter('conv_sin')

    if conv_cos is None or conv_sin is None:
        if window_type == 'hanning':
            window_func = np.hanning(window_size + 1)[:-1]
        elif window_type == 'hamming':
            window_func = np.hamming(window_size + 1)[:-1]
        else:
            raise ValueError("Unknown window type {}.".format(window_type))

        # pad window if `fft_size > window_size`
        if fft_size > window_size:
            diff = fft_size - window_size
            window_func = np.pad(
                window_func, (diff//2, diff - diff//2), mode='constant')
        elif fft_size < window_size:
            raise ValueError(
                "FFT size has to be as least as large as window size.")

        # compute inverse STFT filter coefficients
        if fft_size % stride != 0:
            raise ValueError("FFT size needs to be a multiple of stride.")

        inv_window_func = np.zeros_like(window_func)
        for s in range(0, fft_size, stride):
            inv_window_func += np.roll(np.square(window_func), s)

        mat_cos = np.zeros((fft_size//2 + 1, 1, fft_size))
        mat_sin = np.zeros((fft_size//2 + 1, 1, fft_size))

        for w in range(fft_size//2+1):
            alpha = 1.0 if w == 0 or w == fft_size//2 else 2.0
            alpha /= fft_size
            for t in range(fft_size):
                mat_cos[w, 0, t] = alpha * \
                    np.cos(2. * np.pi * w * t / fft_size)
                mat_sin[w, 0, t] = alpha * \
                    np.sin(2. * np.pi * w * t / fft_size)
        mat_cos = mat_cos * window_func / inv_window_func
        mat_sin = mat_sin * window_func / inv_window_func

        conv_cos = get_parameter_or_create(
            'conv_sin', initializer=mat_cos, need_grad=False)
        conv_sin = get_parameter_or_create(
            'conv_cos', initializer=mat_sin, need_grad=False)

    # compute inverse STFT
    x_cos = deconvolution(y_r, conv_cos, stride=(stride,))
    x_sin = deconvolution(y_i, conv_sin, stride=(stride,))

    x = reshape(x_cos - x_sin, (x_cos.shape[0], x_cos.shape[2]))

    if center:
        x = x[:, fft_size//2:-fft_size//2]

    return x


def gather_nd(data, indices):
    """Gather elements or slices from `data` according to `indices`, which must
    be at least two-dimensional with the first dimension :math:`M` being less or
    equal to the :math:`N` dimensions of `data`. Given `data` with shape
    :math:`(X_0, X_1, ..., X_{N-1})` and indices with shape :math:`(M, Y_0, ...,
    Y_{K-1})` output has shape :math:`(Y_0, ..., Y_{K-1}, X_M, ..., X_{N-1})`.
    If :math:`M == N`, output shape is simply :math:`(Y_0, ..., Y_{K-1})`.

    The forward of :func:`~nnabla.functions.gather_nd` is equivalent to:

    .. code-block:: python

      def gather_nd(data, index):
          import numpy as np
          tmp_index = index.reshape(index.shape[0], -1)
          tmp_index = (idx + (Ellipsis,) for idx in zip(*new_index))
          out_shape = index.shape[1:] + data.shape[index.shape[0]:]
          return np.vstack(data[idx] for idx in tmp_index).reshape(*out_shape)

    Examples:

    >>> import numpy as np, nnabla as nn, nnabla.functions as F
    >>> nn.set_auto_forward(True)
    >>> data = F.arange(1, 11).reshape([2, 5])
    >>> print(data.d)
    [[ 1.  2.  3.  4.  5.]
     [ 6.  7.  8.  9. 10.]]
    >>> F.gather_nd(data, [[1, 1, 0]]).shape
    (3, 5)
    >>> F.gather_nd(data, [[1, 1, 0], [0, 1, 0]]).shape
    (3,)
    >>> print(F.gather_nd(data, [[1, 1, 0], [0, 1, 0]]).d)
    [6. 7. 1.]
    >>> print(F.gather_nd(data, [[1, 1, 0]]).d)
    [[ 6.  7.  8.  9. 10.]
     [ 6.  7.  8.  9. 10.]
     [ 1.  2.  3.  4.  5.]]

    When `indices` is provided as a :obj:`~nnabla.Variable` it will be possible
    to change the actual index values after function creation. It is important
    to note that out-of-bound indices raise errors when running on CPU but are
    ignored when using an accelerated computation context.

    >>> indices = nn.Variable((2, 1))
    >>> indices.d = [[0], [0]]
    >>> y = F.gather_nd(data, indices)
    >>> print(y.d)
    [1.]
    >>> indices.d = [[1], [4]]
    >>> y.forward()
    >>> print(y.d)
    [10.]

    Args:
        data(~nnabla.Variable, ~nnabla.NdArray): input data
        indices(list, numpy.ndarray, ~nnabla.Variable, ~nnabla.NdArray): gather indices

    Returns: ~nnabla.Variable or ~nnabla.NdArray of gathered elements.
    """
    from .function_bases import gather_nd as gather_nd_base
    if not isinstance(indices, (nn.Variable, nn.NdArray)):
        if not isinstance(indices, np.ndarray):
            indices = np.asarray(indices, dtype=np.int)
        indices = nn.Variable.from_numpy_array(indices)
    return gather_nd_base(data, indices)


def scatter_nd(data, indices, shape=None, out=None):
    """Scatter `data` according to `indices` into a new array of given `shape`
    or an existing array provided as `out`. Exactly one of the `shape` or `out`
    argument must be given. Given output `shape`, or shape of `out` array,
    :math:`(X_0,X_1,\ldots,X_{N-1})` and `indices` shape
    :math:`(M,Y_0,\ldots,Y_{K-1})` the input `data` shape is
    :math:`(Y_0,\ldots,Y_{K-1},X_M,\ldots,X_{N-1})`, where :math:`M<=N`. If
    :math:`M==N` the `data` shape is simply :math:`(Y_0,\ldots,Y_{K-1})`.
    Note that `indices` are treated as integers and potentially converted.

    The forward of :func:`~nnabla.functions.scatter_nd` is equivalent to:

    .. code-block:: python

      def scatter_nd(data, indices, shape=None, out=None):
          assert (shape and not out) or (out and not shape)
          if isinstance(indices, numpy.ndarray)
              indices = indices.tolist()
          result = out if out else numpy.zeros(shape)
          result[indices] = data
          return result

    Examples:

    >>> import numpy as np, nnabla as nn, nnabla.functions as F
    >>> nn.set_auto_forward(True)
    >>> data = nn.Variable.from_numpy_array(np.array([9, 10, 11, 12]))
    >>> indices = nn.Variable.from_numpy_array(np.array([[4, 3, 1, 7]]))
    >>> scattered = F.scatter_nd(data, indices, shape=(8,))
    >>> print(scatterd.d)
    [ 0. 11.  0. 10.  9.  0.  0. 12.]
    >>> print(F.gather_nd(scattered, indices).d)
    [ 9. 10. 11. 12.]

    Args:
        data(~nnabla.Variable, ~nnabla.NdArray): input data
        indices(list, numpy.ndarray, ~nnabla.Variable, ~nnabla.NdArray): scatter indices
        shape(tuple, list): shape of new output array
        out(~nnabla.Variable, ~nnabla.NdArray): existing output array

    Returns: ~nnabla.Variable or ~nnabla.NdArray of given `shape`.

    """
    from .function_bases import scatter_nd as scatter_nd_base
    if not isinstance(indices, (nn.Variable, nn.NdArray)):
        if not isinstance(indices, np.ndarray):
            indices = np.asarray(indices, dtype=np.int)
        indices = nn.Variable.from_numpy_array(indices)
    if shape is None and out is None:
        raise TypeError("One of `shape` or `out` argument must be supplied.")
    if shape and out:
        raise TypeError("Only one of `shape` or `out` argument may be used.")
    if out:
        if isinstance(out, nn.Variable):
            out = out.data
        if not isinstance(out, nn.NdArray):
            raise TypeError("`out` argument must be NdArray or Variable type.")
        shape = out.shape
        outputs = [out]
    else:
        if isinstance(shape, np.ndarray):
            shape = shape.tolist()
        outputs = None
    return scatter_nd_base(data, indices, shape, outputs=outputs)
