# Copyright 2017,2018,2019,2020,2021 Sony Corporation.
# Copyright 2021 Sony Group Corporation.
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
from . import dtypes
import nnabla as nn
import numpy as np
from .normalization_functions import *
from .numpy_compat_functions import *


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


def norm(x, p=None, axis=None, keepdims=False):
    r"""
    Reduction along axes with norm operation.

    .. math::
        y = \|x\|_p = \left( \sum_i |x_i|^p \right)^{\frac{1}{p}}

    Args:
        x (Variable): An input variable.
        p (float): Order of the norm.
        axis (None, int or tuple of ints): Axis or axes along which product is
            calculated. Passing the default value `None` will reduce all dimensions.
        keepdims (bool): Flag whether the reduced axes are kept as a dimension with 1 element.

    Returns:
        ~nnabla.Variable: N-D array.

    """
    from .function_bases import norm as norm_base
    if axis is None:
        axis = range(x.ndim)
    elif not hasattr(axis, '__iter__'):
        axis = [axis]
    return norm_base(x, p, axis, keepdims)


def norm_normalization(x, p=None, axes=None, eps=1e-12):
    r"""
    Norm normalization.

    .. math::
        y = \frac{x_i}{\|x\|_p}

    Args:
        x(~nnabla.Variable): N-D array.
        p(float): Order of the norm.
            [default= `2` ]
        axes(repeated int64): Axes to be reduced. If empty list is given, all dimensions are reduced.
            [default= `range(x.ndim)` ]
        eps(float): Epsilon for the normalization. This `eps` is added before taking the p-th root in the norm computation.
            [default= `1e-12` ]

    Returns:
        ~nnabla.Variable: N-D array
    """
    from .function_bases import norm_normalization as norm_normalization_base
    if axes is None:
        axes = range(x.ndim)
    elif not hasattr(axes, '__iter__'):
        axes = [axes]
    return norm_normalization_base(x, p, axes, eps)


def spectral_norm(w, u, dim=0, itr=1, eps=1e-12, test=False, output_u=False):
    r"""
    Spectral Normalization.

    .. math::

        W_{sn} = \frac{W}{\sigma(W)}

    where :math:`W` is the input matrix, and the :math:`\sigma(W)` is the spectral norm of :math:`W`. The spectral norm is approximately computed by the power iteration.

    References:

        Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida, 
        "Spectral Normalization for Generative Adversarial Networks", 
        International Conference on Learning Representations. 2018.

    Args:
        w(~nnabla.Variable): N-D array of learnable weights. This is normally network parameter.
        u(~nnabla.Variable): 1-D array of singular vector. When `test == False`, the data region of `u` will be updated during forward calculation.
        dim(int): Output dimension. Default is 0. If the dimension is not 0, then the specified dimension becomes the most-left dimension by transposing.
            [default= `0` ]
        itr(int): Number of power iterations. Default is 1.
            [default= `1` ]
        eps(float): Epsilon for the normalization. This `eps` is added before taking the sqrt in the norm computation.
            [default= `1e-12` ]
        test(bool): When in `True`, `u` will not be updated. Default is `False`.
            [default= `False` ]
        output_u(bool): Output original `u` or not. `u` is updated when `test == True` but you can get original `u` as output with this option. Default is `False`.
            [default= `False` ]

    Returns:
        ~nnabla.Variable: Spectrally normalized :math:`W_{sn}` with the same shape as :math:`W`.
    """
    from .function_bases import spectral_norm as spectral_norm_base
    n_outputs = 2 if output_u else 1
    return spectral_norm_base(w, u, dim, itr, eps, test, output_u, n_outputs)


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


def meshgrid(*x, ij_indexing=False):

    from .function_bases import meshgrid as meshgrid_base
    return meshgrid_base(*x, ij_indexing=ij_indexing, n_outputs=len(x))


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
            [default= `(0,) * len(x.shape)` ]
        stop(repeated int64): Stop indices for each axis
            [default= `tuple(x.shape)` ]
        step(repeated int64): Step indices for each axis
            [default= `(1,) * len(x.shape)` ]

    Returns:
        ~nnabla.Variable: Sliced N-D array
    """
    start = list(start[:]) if start is not None else len(x.shape) * (0,)
    stop = list(stop[:]) if stop is not None else tuple(x.shape)
    step = list(step[:]) if step is not None else len(x.shape) * (1,)

    for i, (s0, s1, s2) in enumerate(zip(start, stop, step)):
        # Passing all logic to c++ side breaks the slice.args which is used in some cases; nn.grad and loading a nnp file, so handle special cases in python side.
        if s0 is None:
            if s2 is not None and s2 < 0:
                start[i] = -1
            else:
                start[i] = 0
        if s1 is None:
            if s2 is not None and s2 < 0:
                stop[i] = -x.shape[i] - 1
            else:
                stop[i] = x.shape[i]
        if s2 is None:
            step[i] = 1

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
            [default= `1` ]
        update_running_mean(bool): Update running mean during forward execution.
            [default= `True` ]

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
    r"""Fixed Point Quantize.

    This function simulates to uniformly quantize values in fixed-point number representation.

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
    r"""Pow2 Quantize.

    This function simulates to uniformly quantize values in fixed-point number representation.

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


def min_max_quantize(x, qr_min, qr_max, ql_min, ql_max, decay=0.999, x_min_max=False, ema=False,
                     ste_fine_grained=True, eps=0.01, quantize=True, outputs=None):
    r"""Min-max quantization.

    This function simulates to uniformly quantize values in fixed-point number representation.

    Min-max quantization is defined as the following equation

    .. math::

        y = round \left(\frac{\min(\max(x, m), M) - m}{scale} \right) \times scale + m, 

    where the :math:`scale` is defined as 

    .. math::

        scale = \frac{M - m}{M_q - m_q}, 

    and 

    .. math::

        m_q = ql_{min}, \\
        M_q = ql_{max}, \\
        m = qr_{min}, \\
        M = qr_{max}.

    In the backward pass when using `ste_fine_grained` as false,

        .. math::

          \frac{\partial q_i}{\partial x_i} = 1.


    In the backward pass when using `ste_fine_grained` as true,

        .. math::

           \frac{\partial q_i}{\partial x_i}= \left\{
         \begin{array}{ll}
           0 & if \ \ \ x_i > M \\
           1 & if \ \ m \le x_i \le M \\
           0 & if \ \ x_i < m \\
         \end{array} \right..

    :math:`qr_{min}` and :math:`qr_{max}` are treaded as follows.

        * `x_min_max` is `True` and `ema` is `True`: 
          Exponential moving average are computed for each :math:`min(x)` and :math:`max(x)` 
          then stored in :math:`qr_{min}` and :math:`qr_{max}`.
        * `x_min_max` is `True` and `ema` is `False`:
          :math:`min(x)` and :math:`max(x)` are computed then stored in :math:`qr_{min}` and :math:`qr_{max}`.
        * `x_min_max` is `False` and `ema` is `True`:
          Exponential moving average stored in :math:`qr_{min}` and :math:`qr_{max}` are used.
        * `x_min_max` is `False` and `ema` is `False`
          Gradients of :math:`qr_{min}` and :math:`qr_{max}` are computed in the backward pass.

    More precisely, in inference of the min-max quantization, one has to consider *zero-point (zp)*
    which corresponds
    to the real value 0, and its data type is an integer. *zero-point* is defined as 

        .. math::

           && zp_f = ql_{min} -\frac{qr_{min}}{scale}, \\
           && zp = \left\{
         \begin{array}{ll}
           ql_{max} & if \ \ \ zp_f >= ql_{max} \\
           round(zp_f) & if \ \ otherwise \\
           ql_{min}  & if \ \ zp_f <= ql_{min} \\
         \end{array} \right..

    Accordingly, in order to simulate quantization effect of *zero-point*, 
    during both forward and backward pass, :math:`qr_{min}` and :math:`qr_{max}` are adjusted as follows,

        .. math::

           qr_{min}^{adj} = ql_{min} - zp * scale, \\
           qr_{max}^{adj} = ql_{max} - zp * scale.

    These operations are often called *nudge*. 

    Finally, in the formulas of the min-max quantization, :math:`m` and :math:`M` are replaced by
    :math:`qr_{min}^{adj}` and :math:`qr_{max}^{adj}` respectively.

    Args:
        x (~nnabla.Variable): Input N-D array.
        qr_min (~nnabla.Variable): Minimum quantization range (modified during forward execution).
        qr_max (~nnabla.Variable): Maximum quantization range (modified during forward execution).
        ql_min (~nnabla.Variable): Minimum quantization level, typically 0.
        ql_max (~nnabla.Variable): Maximum quantization level, typically 255.
        decay (float): The decay rate for the exponential moving average.
        x_min_max (bool): Use the min and max of x to compute quantization ranges. Default is `False`.
        ema (bool): Use the exponential moving average for the min and max quantization ranges.
                    Default is `False`.
        ste_fine_grained (bool): If `True`, STE is not 1, the {0, 1}-mask computed from the min-max is
                                 applied to the gradient in the backward; otherwise, STE is 1.
        eps (float): Epsilon, or small value to ensure :math:`qr_{max} - qr_{min}` must be greater
                     than the epsilon.
        quantize (bool): Apply quantization or not.

    References:
        Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew Howard, Hartwig Adam, and Dmitry Kalenichenko, "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference", https://arxiv.org/abs/1712.05877

    """

    from .function_bases import min_max_quantize as min_max_quantize_base
    if not quantize:
        return x
    return min_max_quantize_base(x, qr_min, qr_max, ql_min, ql_max, decay, x_min_max, ema,
                                 ste_fine_grained, eps, quantize, outputs=outputs)


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
        min (Variable or float): A min variable or float value by which `x` is clipped. Note that if Variable is given, its shape must be the same as `x`'s.
        max (Variable or float): A max variable or float value by which `x` is clipped. Note that if Variable is given, its shape must be the same as `x`'s

    Returns:
        ~nnabla.Variable: N-D array.

    """
    if np.isscalar(min):
        maximum_base = maximum_scalar
    elif isinstance(min, (nn.Variable, nn.NdArray)):
        maximum_base = maximum2
    else:
        raise TypeError("min must be Variable, NdArray, or scalar.")

    if np.isscalar(max):
        minimum_base = minimum_scalar
    elif isinstance(max, (nn.Variable, nn.NdArray)):
        minimum_base = minimum2
    else:
        raise TypeError("max must be Variable, NdArray, or scalar.")

    return minimum_base(maximum_base(x, min), max)


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
        clip_norm (Variable or float): An input scalar variable or float value. Must be positive.
        axis (None, int or tuple of ints): Axis or axes along which the reduction is performed. Passing the default value `None` will reduce all dimensions.

    Returns:
        ~nnabla.Variable: N-D array.

    """
    from .function_bases import sum as sum_base

    if axis is None:
        axis = range(x.ndim)
    elif not hasattr(axis, '__iter__'):
        axis = [axis]
    x_norm = pow_scalar(sum_base(x**2.0, axis, True), 0.5)
    if isinstance(clip_norm, (nn.Variable, nn.NdArray)):
        y = x * clip_norm / maximum2(x_norm, clip_norm)
    else:
        if clip_norm <= 0:
            raise ValueError("clip_norm must be positive.")
        y = x * clip_norm / maximum_scalar(x_norm, clip_norm)
    return y


def interpolate(x, scale=None, output_size=None, mode='linear',
                align_corners=False, half_pixel=False, half_pixel_for_nn=False,
                channel_last=False):
    '''
    Resize an ND array with interpolation.

    Scaling factors for spatial dimensions are determined by either
    ``scale`` or ``output_size``.

    ``nd = len(scale)`` or ``nd = len(output_size)`` determines the number of
    spatial dimensions, and the last ``nd`` dimensions of the input ``x`` are    
    considered as the spatial dimensions to be resized.


    If ``scale`` is given, the ``output_size`` is calculated by

    .. code-block:: python

      output_size[i] = floor(scale[i] * x.shape[i - len(scale)]).

    Calculation of the coordinate transformation are as follows.

    The input coordinate i_input is computed by the output coordinate i_output,
    the input size size_input, and the output size size_output as

    .. table:: 
        :align: center
        :widths: auto

        ================= ============== =================================================================
         align_corners     half_pixel                            i_input                                  
        ================= ============== =================================================================
              True             True       Not supported.
        ----------------- -------------- -----------------------------------------------------------------
              True             False      i_output * (size_input - 1) / (size_output - 1)                 
        ----------------- -------------- -----------------------------------------------------------------
              False            True       (i_output + 0.5) * size_input / size_output - 0.5               
        ----------------- -------------- -----------------------------------------------------------------
              False            False      i_output * size_input / size_output                             
        ================= ============== =================================================================


    In the case of the `nearest` mode and ``half_pixel_for_nn`` is ``True``, 
    the input coordinate i_input is computed by the output coordinate i_output as

    .. code-block::

      i_input = (i_output + 0.5) * size_input / size_output.


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
            same values with the input corner pixels. Default is ``False``.
        half_pixel:
            If true, in the coordinate transformation, 0.5 is added to the output coordinate
            and 0.5 is subtracted from the input coordinate after scaling. Default is ``False``.
        half_pixel_for_nn: 
            This is a special argument to support the backward-compatibility of the nearest neighbor interpolation.
            Default is ``False``. When in ``True``, the implementation of nearest neighbor interpolation
            is the old one.
        channel_last: Last dimension is the channel (NHWC order) if True.

    Returns:
        ~nnabla.Variable: N-D array.

    .. warning::
        Up to the version 1.8.0, the default of `align_corners` was ``None``, and it becomes ``True``
        if `mode` is linear, otherwise ``False``.

    .. warning::
        Up to the version 1.8.0, the nearest `mode` interpolation corresponds to
        the nearest `mode` and `half_pixel_for_nn` = ``True`` after the version 1.8.0.

    '''
    from .function_bases import interpolate as interpolate_base
    import math
    if scale is None and output_size is None:
        raise ValueError('Either scale or output_size must be given')
    elif output_size is None:
        input_size = x.shape[-len(scale)-1:-1] if channel_last \
            else x.shape[-len(scale):]
        output_size = [int(math.floor(s * d))
                       for d, s in zip(input_size, scale)]
    return interpolate_base(x, output_size, mode, align_corners, half_pixel, half_pixel_for_nn, channel_last)


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
    >>> F.tile(nn.Variable([2, 3]), 3).shape    # reps is promoted to [1, 3]
    (2, 9)
    >>> F.tile(nn.Variable([3]), [2, 3]).shape  # x is promoted to shape (1, 3)
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


def stft(x, window_size, stride, fft_size, window_type='hanning', center=True, pad_mode='reflect', as_istft_backward=False):
    """Computes the short-time Fourier transform

    Args:
        x (~nnabla.Variable): Time domain sequence of size `batch_size x sample_size`.
        window_size (int): Size of STFT analysis window.
        stride (int): Number of samples that we shift the window, also called `hop size`.
        fft_size (int): Size of the FFT, the output will have `fft_size // 2+ 1` frequency bins.
        window_type (str): Analysis window, can be either `hanning`, `hamming` or `rectangular`.
            For convenience, also `window_type=None` is supported which is equivalent to `window_type='rectangular'`.
        center (bool): If `True`, then the signal `x` is padded by half the FFT size using reflection padding.
        pad_mode (str): Padding mode, which can be `'constant'` or `'reflect'`. `'constant'` pads with `0`.
        as_istft_backward: If `True`, then forward execution behaves as backward execution of ISTFT, 
            treating input `x` as output gradient of ISTFT and outputs `y_r` and `y_i` as inputs gradient of ISTFT. 
            This option is only used in nn.grad operator.

    Returns:
        Returns real and imaginary parts of STFT result.

        * :obj:`~nnabla.Variable`: Real part of STFT of size `batch_size x fft_size//2 + 1 x frame_size`.
        * :obj:`~nnabla.Variable`: Imaginary part of STFT of size `batch x fft_size//2 + 1 x frame_size`.
    """
    from .function_bases import stft as stft_base
    if window_type is None:
        window_type = "rectangular"
    return stft_base(x, window_size, stride, fft_size, window_type, center, pad_mode, as_istft_backward)


def _stft_v1(x, window_size, stride, fft_size, window_type='hanning', center=True, pad_mode='reflect'):
    """Computes the short-time Fourier transform

    Args:
        x (~nnabla.Variable): Time domain sequence of size `batch_size x sample_size`.
        window_size (int): Size of STFT analysis window.
        stride (int): Number of samples that we shift the window, also called `hop size`.
        fft_size (int): Size of the FFT, the output will have `fft_size // 2+ 1` frequency bins.
        window_type (str): Analysis window, can be either `hanning`, `hamming` or `rectangular`.
            For convenience, also `window_type=None` is supported which is equivalent to `window_type='rectangular'`.
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
        elif window_type == 'rectangular' or window_type is None:
            window_func = np.ones(window_size)
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
    x = reshape(x, (x.shape[0], 1, x.shape[1]))

    # compute STFT
    y_r = convolution(x, conv_r, stride=(stride,))
    y_i = convolution(x, conv_i, stride=(stride,))

    return y_r, y_i


def istft(y_r, y_i, window_size, stride, fft_size, window_type='hanning', center=True, pad_mode='reflect', as_stft_backward=False):
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
        window_type (str): Analysis window, can be either `hanning`, `hamming` or `rectangular`.
            For convenience, also `window_type=None` is supported which is equivalent to `window_type='rectangular'`.
        center (bool): If `True`, then it is assumed that the time-domain signal has centered frames.
        pad_mode (str): Padding mode corresponding to STFT `pad_mode`, which can be `'constant'` or `'reflect'`. `'constant'` pads with `0`.
            This option is ignored for the normal use of ISTFT. You need to set the same `pad_mode` only when `as_stft_backward == True`.
        as_stft_backward (bool): If `True`, then forward execution behaves as backward execution of STFT,
            treating inputs `y_r` and `y_i` as outputs gradient of STFT and output `x` as input gradient of STFT.
            This option is only used in nn.grad operator.

    Returns:
        ~nnabla.Variable: Time domain sequence of size `batch_size x sample_size`.
    """
    from .function_bases import istft as istft_base
    if window_type is None:
        window_type = "rectangular"
    return istft_base(y_r, y_i, window_size, stride, fft_size, window_type, center, pad_mode, as_stft_backward)


def _istft_v1(y_r, y_i, window_size, stride, fft_size, window_type='hanning', center=True):
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
        window_type (str): Analysis window, can be either `hanning`, `hamming` or `rectangular`.
            For convenience, also `window_type=None` is supported which is equivalent to `window_type='rectangular'`.
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
        elif window_type == 'rectangular' or window_type is None:
            window_func = np.ones(window_size)
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
            'conv_cos', initializer=mat_cos, need_grad=False)
        conv_sin = get_parameter_or_create(
            'conv_sin', initializer=mat_sin, need_grad=False)

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
            indices = np.asarray(indices, dtype=int)
        indices = nn.Variable.from_numpy_array(indices)
    return gather_nd_base(data, indices)


def scatter_nd(data, indices, shape=None, out=None, add=False):
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
        add(tool): Add the input data to the same destination specified by the indices.

    Returns: ~nnabla.Variable or ~nnabla.NdArray of given `shape`.

    """
    from .function_bases import scatter_nd as scatter_nd_base
    if not isinstance(indices, (nn.Variable, nn.NdArray)):
        if not isinstance(indices, np.ndarray):
            indices = np.asarray(indices, dtype=int)
        indices = nn.Variable.from_numpy_array(indices)
    if shape is None and out is None:
        raise TypeError("One of `shape` or `out` argument must be supplied.")
    if shape and out:
        raise TypeError("Only one of `shape` or `out` argument may be used.")
    if out:
        if not isinstance(out, (nn.Variable, nn.NdArray)):
            raise TypeError("`out` argument must be NdArray or Variable type.")
        shape = out.shape
    elif isinstance(shape, np.ndarray):
        shape = shape.tolist()
    return scatter_nd_base(data, indices, out, shape, add)


def scatter_add(x0, indices, x1, axis=None):
    '''Add all values from `x1` into the `x0` according to index specified by `indices`.
    This function adds `x1` into the copy of `x0` and outputs the copy.
    The original `x0` will not be changed.
    `x0`, `indices` and `x1` must have same number of dimensions.

    The forward of :func:`~nnabla.functions.scatter_add` is equivalent to:

    .. code-block:: python

      def scatter_add(x0, indices, x1, axis):
          # Assuming each input is 3 dimensional
          import numpy as np
          output = np.copy(x0)
          for i in range(indices.shape[0]):
              for j in range(indices.shape[1]):
                  for k in range(indices.shape[2]):
                      if axis == 0:
                          output[indices[i][j][k]][j][k] += x1[i][j][k]
                      elif axis == 1:
                          output[i][indices[i][j][k]][k] += x1[i][j][k]
                      elif axis == 2:
                          output[i][j][indices[i][j][k]] += x1[i][j][k]
          return output

    Args:
        x0(~nnabla.Variable): N-D array which the data is added to its copy.
        indices(~nnabla.Variable): N-D array scatter indices. 
          The size of each dimension must be equal or smaller than that of x0 except for the specified axis. 
          The value of indices must be smaller than the size of specified axis' dimension of x0. 
          The size of each dimension must be equal or smaller than that of x1. 
          Indices must not be negative.
        x1(~nnabla.Variable): N-D array which is scattered and added to x0.
        axis(int): Axis along which to index. The axis must not exceed the inputs' dimension.
            [default= `0` ]

    Returns:
        ~nnabla.Variable: N-D array which contains the result of scatter addition. The shape is same as x0.
    '''
    from .function_bases import scatter_add as scatter_add_base
    return scatter_add_base(x0, indices, x1, axis)


def multi_head_attention(query, key, value, num_heads, q_weight, k_weight, v_weight, out_weight, q_bias=None, k_bias=None, v_bias=None, out_bias=None, attn_bias_k=None, attn_bias_v=None, dropout=0.0, additive_mask=None, key_padding_mask=None):
    '''MultiHeadAttention.

    Computes multi-headed attention with query, key, and value.
    We use the following notations to describe the inputs and outputs below.
    :math:`L_T`: target sequence length, :math:`L_S`: source sequence length, :math:`B`: batch size, :math:`D`: input dimension, :math:`E`: embedding dimension, :math:`H`: number of attention heads.

    References:

        A. Vaswani et al. "Attention is All You Need."
        NIPS. 2017.
        <https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf>

    Args:
        query (~nnabla.Variable): Input N-D array with shape :math:`(L_T, B, D_q)`.
        key (~nnabla.Variable): Input N-D array with shape :math:`(L_S, B, D_k)`.
        value (~nnabla.Variable): Input N-D array with shape :math:`(L_S, B, D_v)`.
        num_heads (int): Number of attention heads. Note that embedding dimensoin E must be divisible by the number of heads. Default is 12 which is conventional.
        q_weight (~nnabla.Variable): Input N-D array with shape :math:`(D_q, E)`.
        k_weight (~nnabla.Variable): Input N-D array with shape :math:`(D_k, E)`.
        v_weight (~nnabla.Variable): Input N-D array with shape :math:`(D_v, E_v)`.
        out_weight (~nnabla.Variable): Input N-D array with shape :math:`(D_v, E_{out})`.
        q_bias (~nnabla.Variable, optional): Input N-D array with shape :math:`(E, )`.
        k_bias (~nnabla.Variable, optional): Input N-D array with shape :math:`(E, )`.
        v_bias (~nnabla.Variable, optional): Input N-D array with shape :math:`(E_v, )`.
        out_bias (~nnabla.Variable, optional): Input N-D array with shape :math:`(E_{out}, )`.
        attn_bias_k (~nnabla.Variable, optional): Input N-D array with shape :math:`(E, )`.
        attn_bias_v (~nnabla.Variable, optional): Input N-D array with shape :math:`(E_v, )`.
        dropout (float, optional): Dropout ratio applied to parameters. Default is 0.
        additive_mask (~nnabla.Variable, optional): Input N-D array with shape :math:`(L_T, L_S)`. Values will be added to the attention layer to prevent attention to certain positions.
        key_padding_mask (~nnabla.Variable, optional): Input N-D array with shape :math:`(B, L_S)`. Specified padding elements will be ignored by the attention layer. Values must be either 1 or 0.

    Returns:
        ~nnabla.Variable: Output :math:`y` with shape :math:`(L_T, B, E_{out})`
        ~nnabla.Variable: Output :math:`h_n` with shape :math:`(B, L_T, L_S)`
    '''

    from . import functions as F

    tgt_len, batch_size, _ = query.shape
    src_len, batch_size, _ = key.shape
    q_embed_dim = q_weight.shape[1]
    k_embed_dim = k_weight.shape[1]
    v_embed_dim = v_weight.shape[1]
    out_dim = out_weight.shape[1]
    assert src_len == value.shape[0]
    head_dim = q_embed_dim // num_heads
    head_vdim = v_embed_dim // num_heads
    assert q_embed_dim == k_embed_dim, "embedding dimensions must be the same for query and key."
    assert head_dim * num_heads == q_embed_dim, "embedding dimension must be divisibile by num_heads %d" % num_heads
    assert head_vdim * \
        num_heads == v_embed_dim, "v_embed_dim must be divisibile by num_heads %d." % num_heads

    if key_padding_mask is not None:
        assert key_padding_mask.shape[0] == batch_size
        assert key_padding_mask.shape[1] == src_len

    # query:(L_T, B, E) --> q:(L_T, B, E)
    q = F.affine(query, q_weight, q_bias, base_axis=2)
    # key:(L_S, B, D_k) --> k:(L_S, B, E_k)
    k = F.affine(key, k_weight, k_bias, base_axis=2)
    # value:(L_S, B, D_v) --> v:(L_S, B, E_v)
    v = F.affine(value, v_weight, v_bias, base_axis=2)

    q *= float(head_dim) ** -0.5

    if attn_bias_k is not None:
        attn_bias_k = F.reshape(attn_bias_k, (1, 1, k_embed_dim))
        attn_bias_v = F.reshape(attn_bias_v, (1, 1, v_embed_dim))
        src_len += 1
        assert attn_bias_k is not None
        attn_bias_k = F.broadcast(
            attn_bias_k, (1, batch_size, attn_bias_k.shape[2]))
        attn_bias_v = F.broadcast(
            attn_bias_v, (1, batch_size, attn_bias_v.shape[2]))
        k = F.concatenate(k, attn_bias_k, axis=0)
        v = F.concatenate(v, attn_bias_v, axis=0)
        if additive_mask is not None:
            # additive_mask: (L_T, L_S) --> (L_T, L_S + 1)
            additive_mask = F.pad(additive_mask, (0, 1))
        if key_padding_mask is not None:
            # key_padding_mask: (B, L_S) --> (B, L_S + 1)
            key_padding_mask = F.pad(key_padding_mask, (0, 1))

    q = F.transpose(
        F.reshape(q, (tgt_len, batch_size * num_heads, head_dim)), (1, 0, 2))  # q:(B*H, L_T, head_dim)
    k = F.transpose(
        F.reshape(k, (-1, batch_size * num_heads, head_dim)), (1, 0, 2))  # k:(B*H, L_S, head_dim)
    v = F.transpose(
        F.reshape(v, (-1, batch_size * num_heads, head_vdim)), (1, 0, 2))  # v:(B*H, L_S, head_vdim)

    # attn_output_weights: (B*H, L_T, L_S)
    attn_output_weights = F.batch_matmul(q, k, transpose_b=True)
    assert list(attn_output_weights.shape) == [
        batch_size * num_heads, tgt_len, src_len]

    if additive_mask is not None:
        additive_mask = F.reshape(additive_mask, ((1,) + additive_mask.shape))
        attn_output_weights += additive_mask

    if key_padding_mask is not None:
        attn_output_weights = F.reshape(
            attn_output_weights, (batch_size, num_heads, tgt_len, src_len))
        attn_output_weights = F.where(
            F.broadcast(
                F.reshape(key_padding_mask, (batch_size, 1, 1, src_len)),
                attn_output_weights.shape),  # Condition
            F.constant(val=float('-inf'),
                       shape=attn_output_weights.shape),  # If true
            attn_output_weights)  # If false
        attn_output_weights = F.reshape(
            attn_output_weights, (batch_size*num_heads, tgt_len, src_len))

    attn_output_weights = F.softmax(
        attn_output_weights, axis=len(attn_output_weights.shape)-1)
    if dropout > 0:
        attn_output_weights = F.dropout(
            attn_output_weights, p=dropout)

    # (B*H, L_T, L_S) x (B*H, L_S, head_vdim) --> (B*H, L_T, head_vdim)
    attn_output = F.batch_matmul(attn_output_weights, v)
    assert list(attn_output.shape) == [
        batch_size * num_heads, tgt_len, head_vdim]
    attn_output = F.reshape(F.transpose(
        attn_output, (1, 0, 2)), (tgt_len, batch_size, v_embed_dim))  # attn_output: (L_T, B, E_v)

    attn_output = F.affine(attn_output, out_weight, out_bias, base_axis=2)

    return attn_output, attn_output_weights


def patch_correlation(x1, x2, patch=(1, 1), shift=(0, 0), patch_step=(1, 1),
                      shift_step=(1, 1), padding=(0, 0, 0, 0),
                      channel_last=False):
    r"""
    Multiplicative patch-wise comparison between inputs `x1` and `x2`, which
    must both be 4-dimensional NCHW (with `channel_last=False`) or NHWC (with
    `channel_last=True`) arrays (where *N* is the number of samples, *H* and
    *W* are the sample height and width and *C* is the number of channels).
    The function returns a 5-D array with shape :math:`(N, C_y, C_x, H_o, W_o)`
    where :math:`H_o, W_o` are determined by the possible patch locations within
    the, optionally padded, input image sizeand :math:`C_y, C_x` are determined
    by the optionally shifted patch positions.

    Mathematically, the patch correlation is formulated as

    .. math::

       O(s_y, s_x, h_0, w_0) =
       \sum_{c} \sum_{k_h} \sum_{k_w} I_1(c, h + k_h, w + k_w) \times I_2(c, h + k_h + s_h, w + k_w + s_w), 

    where :math:`I_1(c, h, w)` and :math:`I_2(c, h, w)` are the inputs at :math:`c`-th channel, 
    :math:`h`-th height, and :math:`w`-th width, :math:`k_h, k_w` indices for the patch size 
    and :math:`s_h, s_w` indices for the shifts.

    A single correlation value (per sample) is produced if the patch extends
    to the image dimensions and all other parameters use the default values.

    >>> import numpy as np, nnabla as nn, nnabla.functions as F
    >>> N, C, H, W = (1, 2, 3, 4)
    >>> x = nn.Variable.from_numpy_array(np.ones([N, C, H, W]))
    >>> F.patch_correlation(x, x, patch=(H, W)).d
    array([[[[[24.]]]]], dtype=float32)

    A patch that is smaller than the image size moves horizontal and vertical
    producing a value per position. The `patch_step` argument may be used to
    control the position increments.

    >>> F.patch_correlation(x, x, patch=(H-1, W-1)).d
    array([[[[[12., 12.],
              [12., 12.]]]]], dtype=float32)
    >>> F.patch_correlation(x, x, patch=(H-1, W-1), patch_step=(2, 1)).d
    array([[[[[12., 12.]]]]], dtype=float32)

    Multiple correlations may be performed at each position between the patch
    from `x1` and patches from `x2` at relative offsets striding the maximum
    vertical and horizontal distance given by the `shift` values at increments
    of `shift_step`. The shifted correlation values can be obtained for the
    from the second and third output dimension for the vertical and horizontal
    shifts.

    >>> F.patch_correlation(x, x, (H, 1), shift=(0, 1)).shape
    (1, 1, 3, 1, 4)
    >>> F.patch_correlation(x, x, (H, 1), shift=(0, 1)).d
    array([[[[[0., 6., 6., 6.]],
             [[6., 6., 6., 6.]],
             [[6., 6., 6., 0.]]]]], dtype=float32)
    >>> F.patch_correlation(x, x, (H, 1), shift=(0, 1), shift_step=(1, 2)).d
    array([[[[[0., 6., 6., 6.]],
             [[6., 6., 6., 0.]]]]], dtype=float32)

    Padding with zero values may be applied individually to the top, bottom,
    left and right side of the input image.

    >>> F.patch_correlation(x, x, patch=(H, W), padding=(0, 1, W, W)).d
    array([[[[[ 0.,  6., 12., 18., 24., 18., 12.,  6.,  0.],
              [ 0.,  4.,  8., 12., 16., 12.,  8.,  4.,  0.]]]]], dtype=float32)

    This function may be used to implement the FlowNetC correlation layer.

    >>> N, C, H, W = (1, 256, 44, 60)
    >>> x1, x2 = nn.Variable((N, C, H, W)), nn.Variable((N, C, H, W))
    >>> F.patch_correlation(x1, x2, shift=20, shift_step=2).shape
    (1, 21, 21, 44, 60)

    References:

        * `Fischer et al., FlowNet: Learning Optical Flow with Convolutional
          Networks. <https://arxiv.org/abs/1504.06852>`_

    Args:
        x1(~nnabla.Variable): Input N-D array with shape :math:`(N, C, H, W)`
            or :math:`(N, H, W, C)`.
        x2(~nnabla.Variable): Input N-D array with shape :math:`(N, C, H, W)`
            or :math:`(N, H, W, C)`.
        patch: A tuple with height and width of the correlation patch. A single
            integer expands to identical height and width.
        shift: A tuple of maximum vertical and horizontal displacement of
            patches from `x2` that are correlated with a single patch from `x1`.
            A single integer expands to identical vertical and horizontal
            displacement.
        patch_step: A tuple of vertical and horizontal increments for advancing
            the position of the correlation patch within the input image shape.
            A single integer expands to identical vertical and horizontal
            increments.
        shift_step: A tuple of vertical and horizontal increments for advancing
            the relative offset position within the shift range. A single
            integer expands to identical vertical and horizontal increments.
        padding: A tuple of top, bottom, left and right padding extent. A tuple
            of two values yields identical top/bottom and left/right padding
            from the first and second tuple value. A single integer expands to
            indential padding extent for all sides.
        channel_last: Last dimension is the channel (NHWC order) if True.

    Returns:
        ~nnabla.Variable: N-D array with shape :math:`(N, C_y, C_x, H_o, W_o)` or :math:`(N, H, W, C_y, C_x)` if `channel_last=True`.

          A spatial size of the output is calculated as

          .. math:: 

            H_o = \frac{H + (top\_pad + bottom\_pad) - patch_v}{patch\_step_v} + 1.

          A channel size of the ouptut is calculated as

          .. math::

            C_y = \frac{2 \times shift_v}{shift\_step_v} + 1.

          :math:`W_o` and :math:`C_x` are the same calculation with differenct components.
    """
    from .function_bases import patch_correlation as patch_correlation_base

    if not len(x1.shape) == len(x2.shape) == 4:
        raise ValueError("Both inputs x1 and x2 must have 4 dimensions.")

    if not x1.shape == x2.shape:
        raise ValueError("Both inputs x1 and x2 must have equal shape.")

    if isinstance(patch, int):
        patch = 2 * (patch,)
    if isinstance(shift, int):
        shift = 2 * (shift,)
    if isinstance(patch_step, int):
        patch_step = 2 * (patch_step,)
    if isinstance(shift_step, int):
        shift_step = 2 * (shift_step,)
    if isinstance(padding, int):
        padding = 4 * (padding,)
    if len(padding) == 2 and all(isinstance(p, int) for p in padding):
        padding = 2 * (padding[0],) + 2 * (padding[1],)

    if not channel_last:
        if x1.shape[1] == 1:
            x1 = reshape(x1, (x1.shape[0], *x1.shape[2:4], 1))
            x2 = reshape(x2, (x2.shape[0], *x2.shape[2:4], 1))
        else:
            x1 = transpose(x1, (0, 2, 3, 1))
            x2 = transpose(x2, (0, 2, 3, 1))

    y = patch_correlation_base(x1, x2, patch, shift, patch_step, shift_step,
                               padding)

    if not channel_last:
        y = transpose(y, (0, 3, 4, 1, 2))

    return y


def quantize_linear(x, scale, zero_point,
                    round_mode="HALF_AWAY_FROM_ZERO", narrow_range=False, dtype=np.int8):
    r"""

    Quantize linearly inputs with the scale and zero point.

      .. math::

          y = saturate(round(x / scale) + zero_point).

    :math:`saturate` rage is determined by `dtype` and :math:`round` mode is selected
    by `round_mode`. :math:`zero_point` is constrained by the `dtype` range and its values are
    rounded by `round_mode`.

    This function normally aligns with ONNX QuantizeLinear.


    Args:
        x (Variable): An input variable.
        scale (Variable): Scale variable.
        zero_point (Variable): Zero point variable.
        round_mode (str): Rounding mode. HALF_AWAY_FROM_ZERO or HALF_TO_EVEN.
        narrow_range (bool): If true, this function does not use the minimum quantized value. For
            example, if `dtype` is int8 (the range is in [-128, 127]), the output range
            is corrected in [-127, 127].
        dtype (numpy.dtype): Data type for the output. Currently np.int8 or np.uint8 are supported.
    """
    from .function_bases import quantize_linear as quantize_linear_base
    int_dtype = dtypes.np_dtpye_to_int[dtype]
    y = quantize_linear_base(x, scale, zero_point,
                             round_mode, narrow_range, int_dtype)
    return y


def linspace(start, stop, num):
    r"""
    Generate a one-dimensional vector/tensor of size `num` whose values are evenly spaced from `start` to `end`, inclusive.

    Args:
        start(float): Start value.
        stop(float): End value.
        num(int): Size of the constructed vector/tensor.

    Returns:
        ~nnabla.Variable: 1-D array with the generated values.
    """
    from .function_bases import linspace as linspace_base

    if not isinstance(num, int):
        raise TypeError(
            "'{}' object cannot be interpreted as an integer".format(type(num).__name__))
    if num < 0:
        raise ValueError(
            "Number of samples, {}, must be non-negative.".format(num))

    return linspace_base(start, stop, num)
