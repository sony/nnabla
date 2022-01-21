# Copyright 2019,2020,2021 Sony Corporation.
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
from six.moves import reduce as rd
import numpy as np

import nnabla as nn


def _check_axis(ndim, axis):
    if axis < 0 or axis >= ndim:
        raise ValueError(
            "axis must be in the range of [0, ndim). axis : {}, ndim: {}.".format(
                axis, ndim))


def _force_list(axis):
    if hasattr(axis, "__iter__"):
        return axis

    return [axis]


def _check_batch_axis_and_force_list(ndim, batch_axis):
    batch_axis = _force_list(batch_axis)

    for _axis in batch_axis:
        _check_axis(ndim, _axis)

    return batch_axis


def _get_axes_excluding(ndim, axes):
    axes = _force_list(axes)
    axes = [a+ndim*(a < 0) for a in axes]

    return [i for i in range(ndim) if i not in axes]


def _create_bn_dummy_vars(x, axes, beta, gamma, mean, variance):
    in_shape = x.shape
    axes = _force_list(axes)
    adaptive_shape = tuple(
        in_shape[i] if i in axes else 1 for i in range(len(in_shape)))

    # create dummy adaptive variables
    assert_flag = False
    if beta is not None:
        assert_flag |= beta.shape != adaptive_shape
    else:
        beta = constant(val=0, shape=adaptive_shape)

    if gamma is not None:
        assert_flag |= gamma.shape != adaptive_shape
    else:
        gamma = constant(val=1, shape=adaptive_shape)

    if assert_flag:
        raise ValueError(
            "The shapes of beta and gamma must be {}."
            " If you want to use a tensor with other shape, use arithmetic operators like"
            " `tensor_normalization(x, axes, beta=None, gamma=None) * gamma + beta`.".format(adaptive_shape))

    if mean is None:
        # _mean is never used and there is no need to initialize.
        mean = nn.Variable(adaptive_shape, need_grad=False)

    if variance is None:
        variance = nn.Variable(adaptive_shape, need_grad=False)  # same above.

    return beta, gamma, mean, variance


def _init_beta_gamma(shape, fix_parameters, param_init, no_bias, no_scale):
    from nnabla.parameter import get_parameter_or_create
    from nnabla.initializer import ConstantInitializer

    if no_bias:
        beta = None
    else:
        beta_init = param_init.get('beta', ConstantInitializer(0))
        beta = get_parameter_or_create(
            "beta", shape, beta_init, True, not fix_parameters)

    if no_scale:
        gamma = None
    else:
        gamma_init = param_init.get('gamma', ConstantInitializer(1))
        gamma = get_parameter_or_create(
            "gamma", shape, gamma_init, True, not fix_parameters)

    return beta, gamma


def _apply_affine(x, scale=None, bias=None):
    if scale is not None:
        x *= scale

    if bias is not None:
        x += bias

    return x


class BatchNormalizationInOutAdapter(object):
    def __init__(self, ndim, axes):
        assert len(axes) != 1
        outer_axes = sorted(set(range(ndim)).difference(axes))
        inner_axes = sorted(axes)
        self.ndim = ndim
        self.axes = axes
        self.outer_axes = outer_axes
        self.transpose_axes = outer_axes + inner_axes
        self.transposed_shape = None
        self.inv_transpose_axes = np.argsort(self.transpose_axes).tolist()

    def __call__(self, x):
        transposed = transpose(x, self.transpose_axes)
        if self.transposed_shape is not None:
            assert self.transposed_shape == transposed.shape, "Wrong shape input given."
        self.transposed_shape = transposed.shape
        reduced_inner_size = np.prod(transposed.shape[len(self.outer_axes):])
        outer_shape = list(transposed.shape[:len(self.outer_axes)])
        return reshape(transposed, outer_shape + [reduced_inner_size])

    def inv(self, y):
        transposed = reshape(y, self.transposed_shape)
        return transpose(transposed, self.inv_transpose_axes)


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
        beta(~nnabla.Variable or None): N-D array of beta which is learned. If None, the bias term is omitted.
        gamma(~nnabla.Variable or None): N-D array of gamma which is learned. If None, the scale term is omitted.
        mean(~nnabla.Variable or None):
            N-D array of running mean (modified during forward execution).
            If None, dummy variable is created and running mean is not updated.
            mean=None with batch_stat=False is prohibited.
        variance(~nnabla.Variable or None):
            N-D array of running variance (modified during forward execution).
            If None, dummy variable is created and running variance is not updated.
            variance=None with batch_stat=False is prohibited.
        axes(list of int or int): Mean and variance are calculated along these axes.
        decay_rate(float): Decay rate of running mean and variance.
        eps(float): Tiny value to avoid zero division by std.
        batch_stat(bool):
            Use mini-batch statistics rather than running ones.
            If False, mean and variance must be `~nnabla.Variable`. (None is prohibited.)
        output_stat(bool): It true, the batch statistics of mean and variance,
            will be returned as Variables. They are also differentiable.

    Returns:
        Returns batch normalization output as :obj:`~nnabla.Variable`.
        If ``output_stat=True``, it also returns the mean and variance
        of the mini-batch

        * :obj:`~nnabla.Variable`: Output of the batch normalization
        * :obj:`~nnabla.Variable`: Mean (if `output_stat=True`)
        * :obj:`~nnabla.Variable`: Variance (if `output_stat=True`)

    See Also:
        ``nnabla.function_bases.batch_normalization``.

    """
    from .function_bases import batch_normalization as batch_normalization_base
    n_outputs = 3 if output_stat else 1
    axes = _force_list(axes)
    axes = [a+len(x.shape)*(a < 0) for a in axes]

    assert batch_stat or (not output_stat)

    if not batch_stat and (mean is None or variance is None):
        raise ValueError(
            "If batch_stat is False, mean and variable must not be None.")

    _, _, mean, variance = _create_bn_dummy_vars(
        x, axes, beta, gamma, mean, variance)

    if batch_stat and (mean.parent or variance.parent) is not None:
        raise ValueError(
            "if batch_stat is True, mean and variable must not have a parent function.")

    no_scale = gamma is None
    no_bias = beta is None

    if len(axes) == 1:
        return batch_normalization_base(x, beta, gamma, mean, variance,
                                        axes=axes,
                                        decay_rate=decay_rate,
                                        eps=eps,
                                        batch_stat=batch_stat,
                                        no_scale=no_scale,
                                        no_bias=no_bias,
                                        n_outputs=n_outputs)

    in_adapter = BatchNormalizationInOutAdapter(x.ndim, axes)
    param_adapter = BatchNormalizationInOutAdapter(x.ndim, axes)
    inp = in_adapter(x)
    if beta is not None:
        beta = param_adapter(beta)
    if gamma is not None:
        gamma = param_adapter(gamma)
    mean = param_adapter(mean)
    variance = param_adapter(variance)
    axis = x.ndim - len(axes)

    if n_outputs == 1:
        out = batch_normalization_base(inp, beta, gamma, mean, variance,
                                       axes=[axis],
                                       decay_rate=decay_rate,
                                       eps=eps,
                                       batch_stat=batch_stat,
                                       no_scale=no_scale,
                                       no_bias=no_bias,
                                       n_outputs=n_outputs)
        return in_adapter.inv(out)
    out, mean, variance = batch_normalization_base(inp, beta, gamma, mean, variance,
                                                   axes=[axis],
                                                   decay_rate=decay_rate,
                                                   eps=eps,
                                                   batch_stat=batch_stat,
                                                   no_scale=no_scale,
                                                   no_bias=no_bias,
                                                   n_outputs=n_outputs)
    out = in_adapter.inv(out)
    mean = param_adapter.inv(mean)
    variance = param_adapter.inv(variance)
    return out, mean, variance


def fused_batch_normalization(x, beta, gamma, mean, variance, z=None, axes=[1], decay_rate=0.9, eps=1e-05, batch_stat=True, nonlinearity='relu', output_stat=False, n_outputs=None):
    r"""
    Batch normalization fused with an add operation and an activation.

    References:

        * `Ioffe and Szegedy, Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift.
          <https://arxiv.org/abs/1502.03167>`_

    Args:
        x(~nnabla.Variable): N-D array of input.
        beta(~nnabla.Variable or None): N-D array of beta which is learned. If None, the bias term is omitted.
        gamma(~nnabla.Variable or None): N-D array of gamma which is learned. If None, the scale term is omitted.
        mean(~nnabla.Variable or None):
            N-D array of running mean (modified during forward execution).
            If None, dummy variable is created and running mean is never updated.
            mean=None with batch_stat=False is prohibited.
        variance(~nnabla.Variable):
            N-D array of running variance (modified during forward execution).
            If None, dummy variable is created and running variance is not updated.
            variance=None with batch_stat=False is prohibited.
        z(~nnabla.Variable, optional): N-D array
        axes(list of int or int): Mean and variance are calculated along these axes.
        decay_rate(float): Decay rate of running mean and variance.
        eps(float): Tiny value to avoid zero division by std.
        batch_stat(bool):
            Use mini-batch statistics rather than running ones.
            If False, mean and variance must be `~nnabla.Variable`. (None is prohibited.)
        nonlinearity(str): Nonlinearity chosen from relu. Default is relu.
        output_stat(bool): It true, the batch statistics of mean and variance,
            will be returned as Variables. They are also differentiable.

    Returns:
        Returns batch normalization output as :obj:`~nnabla.Variable`.
        If ``output_stat=True``, it also returns the mean and variance
        of the mini-batch

        * :obj:`~nnabla.Variable`: Output of the batch normalization
        * :obj:`~nnabla.Variable`: Mean (if `output_stat=True`)
        * :obj:`~nnabla.Variable`: Variance (if `output_stat=True`)

    See Also:
        ``nnabla.function_bases.batch_normalization``.

    """
    from .function_bases import fused_batch_normalization as fused_batch_normalization_base
    n_outputs = 3 if output_stat else 1
    axes = _force_list(axes)

    if not batch_stat and (mean is None or variance is None):
        raise ValueError(
            "If batch_stat is False, mean and variable must not be None.")

    beta, gamma, mean, variance = _create_bn_dummy_vars(
        x, axes, beta, gamma, mean, variance)

    if batch_stat and (mean.parent or variance.parent) is not None:
        raise ValueError(
            "if batch_stat is True, mean and variable must not have a parent function.")

    if len(axes) == 1:
        return fused_batch_normalization_base(x, beta, gamma, mean, variance, z,
                                              axes=axes,
                                              decay_rate=decay_rate,
                                              eps=eps,
                                              batch_stat=batch_stat,
                                              nonlinearity=nonlinearity,
                                              n_outputs=n_outputs)

    in_adapter = BatchNormalizationInOutAdapter(x.ndim, axes)
    param_adapter = BatchNormalizationInOutAdapter(x.ndim, axes)
    inp = in_adapter(x)
    z = in_adapter(z)
    beta = param_adapter(beta)
    gamma = param_adapter(gamma)
    mean = param_adapter(mean)
    variance = param_adapter(variance)
    axis = x.ndim - len(axes)
    if not output_stat:
        out = fused_batch_normalization_base(x, beta, gamma, mean, variance, z,
                                             axes=[axis],
                                             decay_rate=decay_rate,
                                             eps=eps,
                                             batch_stat=batch_stat,
                                             nonlinearity=nonlinearity,
                                             n_outputs=n_outputs)
        return in_adapter.inv(out)

    out, mean, variance = fused_batch_normalization_base(inp, beta, gamma, mean, variance, z,
                                                         axes=[axis],
                                                         decay_rate=decay_rate,
                                                         eps=eps,
                                                         batch_stat=batch_stat,
                                                         nonlinearity=nonlinearity,
                                                         n_outputs=n_outputs)
    out = in_adapter.inv(out)
    mean = param_adapter.inv(mean)
    variance = param_adapter.inv(variance)
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
        beta(~nnabla.Variable or None): N-D array of beta which is learned. If None, the bias term is omitted.
        gamma(~nnabla.Variable or None): N-D array of gamma which is learned. If None, the scale term is omitted.
        mean(~nnabla.Variable or None):
            N-D array of running mean (modified during forward execution).
            If None, dummy variable is created and running mean is never updated.
            mean=None with batch_stat=False is prohibited.
        variance(~nnabla.Variable or None):
            N-D array of running variance (modified during forward execution).
            If None, dummy variable is created and running variance is never updated.
            variance=None with batch_stat=False is prohibited.
        comm(~nnabla.communicators.Communicator): The communicator
        group(string): The name of the communicator group
        axes(list of int or int): Mean and variance are calculated along these axes.
        decay_rate(float): Decay rate of running mean and variance.
        eps(float): Tiny value to avoid zero division by std.
        batch_stat(bool):
            Use mini-batch statistics rather than running ones.
            If False, mean and variance must be `~nnabla.Variable`. (None is prohibited.)
        output_stat(bool): It true, the batch statistics of mean and variance,
            will be returned as Variables. They are also differentiable.

    Returns:
        Returns batch normalization output as :obj:`~nnabla.Variable`.
        If ``output_stat=True``, it also returns the mean and variance
        of the mini-batch

        * :obj:`~nnabla.Variable`: Output of the batch normalization
        * :obj:`~nnabla.Variable`: Mean (if `output_stat=True`)
        * :obj:`~nnabla.Variable`: Variance (if `output_stat=True`)

    See Also:
        ``nnabla.function_bases.batch_normalization``.

    """
    from .function_bases import sync_batch_normalization as batch_normalization_base
    n_outputs = 3 if output_stat else 1
    axes = _force_list(axes)

    if not batch_stat and (mean is None or variance is None):
        raise ValueError(
            "If batch_stat is False, mean and variable must not be None.")

    beta, gamma, mean, variance = _create_bn_dummy_vars(
        x, axes, beta, gamma, mean, variance)

    return batch_normalization_base(x, beta, gamma, mean, variance,
                                    comm, group=group,
                                    axes=axes,
                                    decay_rate=decay_rate,
                                    eps=eps,
                                    batch_stat=batch_stat,
                                    n_outputs=n_outputs)


def tensor_normalization(x, axes, beta=None, gamma=None, eps=1e-05, output_stat=False):
    r"""
    General tensor normalization.
    Input variable `x` is normalized by mean and std calculated by `x` itself.
    Mean and variance are calculated along `axes`.
    For example, if the input shape is (B, C, H, W) and axes is [0, 1],
     the shape of calculated mean and std are (B, C, 1 ,1).

    Note:
        Currently tensor_normalization is implemented not as cpp function
        but as wrapper function which just calls F.batch_normalization internally.
        That means F.reshape or F.transpose may be additionally called to satisfy the condition required by F.batch_normalization,
        and if you serialize graphs including tensor_normalization to nnp file,
        that nnp includes reshape, transpose and batch_normalization rather than tensor_normalization layer itself.

    Args:
        x (Variable): N-D array of input variable.
        axes (int or repeated int): Mean and variance are calculated along these axes.
        beta (Variable or None): An Adaptive biases. If None, the bias term is omitted.
        gamma (Variable or None): An Adaptive scales. If None, the scale term is omitted.
        eps (float): Tiny value to avoid zero division by std.
        output_stat(bool): It true, the batch statistics of mean and variance.
        will be returned as Variables. They are also differentiable.

    Returns:
        * :obj:`~nnabla.Variable`: Normalized output variable.
        * :obj:`~nnabla.Variable`: Mean (if ``output_stat=True`)
        * :obj:`~nnabla.Variable`: Std (if ``output_stat=True`)

    """
    # todo: should implement cpp function rather than just calling batch_normalization in python.

    return batch_normalization(x, beta, gamma, None, None,
                               axes=axes, decay_rate=0., eps=eps, batch_stat=True, output_stat=output_stat)


def weight_standardization(w, channel_axis=0, eps=1e-05, output_stat=False):
    r"""
    Applies Weight Standardization over an input weight, which is defined as:

    .. math::
      \begin{eqnarray}
        \mu_{W_i} &=& \frac{1}{I} \sum_{j=1}^{I} W_{ij} \\
        \sigma_{W_i} &=& \sqrt{\frac{1}{I} \sum_{i=1}^{I} \left(W_{ij} - \mu_{W_{i}}\right)^2 + \epsilon} \\
        \hat{W_{ij}} &=& \frac{W_{ij} - \mu_{W_i}}{\sigma_{W_i}} \\
        y &=& \hat{W} \ast x
      \end{eqnarray}

    Example:
        .. code-block:: python

            import numpy as np
            import nnabla as nn
            import nnabla.functions as F
            import nnabla.parametric_functions as PF

            rng = np.random.RandomState(313)
            x = nn.Variable.from_numpy_array(rng.randn(*(32, 16, 3, 3)))

            # For convolution:

            def ws_callback_conv(w):
                return F.weight_standardization(w, channel_axis=0)

            y = PF.convolution(x, 10, (2, 2), apply_w=ws_callback_conv)

            # For affine:

            def ws_callback_affine(w): 
                return F.weight_standardization(w, channel_axis=1)

            y = PF.affine(x, 10, apply_w=ws_callback_affine)


    References:

      * `Siyuan Qiao, Huiyu Wang, Chenxi Liu, Wei Shen, Alan Yuille, Weight Standardization
        <https://arxiv.org/pdf/1903.10520v1.pdf>`_

    Args:
        w (Variable): A weight variable.
        channel_axis (int): An axis for output channel. Default value is 0 which assumes the weights of convolution.
        eps (float): Tiny value to avoid zero division by std.
        output_stat(bool): If true, the batch statistics of mean and variance.

    Returns:
        * :obj:`~nnabla.Variable`: Standardized output weight.
        * :obj:`~nnabla.Variable`: Mean (if `output_stat=True`)
        * :obj:`~nnabla.Variable`: Std (if `output_stat=True`)

    """
    from .function_bases import weight_standardization as weight_standardization_base
    n_outputs = 3 if output_stat else 1

    return weight_standardization_base(w, channel_axis=channel_axis, eps=eps, n_outputs=n_outputs)


def _weight_standardization_v1(w, channel_axis=0, eps=1e-05, output_stat=False):
    r"""
    Applies Weight Standardization over an input weight, which is defined as:
    .. math::
      \begin{eqnarray}
        \mu_{W_i} &=& \frac{1}{I} \sum_{j=1}^{I} W_{ij} \\
        \sigma_{W_i} &=& \sqrt{\frac{1}{I} \sum_{i=1}^{I} \left(W_{ij} - \mu_{W_{i}}\right)^2 + \epsilon} \\
        \hat{W_{ij}} &=& \frac{W_{ij} - \mu_{W_i}}{\sigma_{W_i}} \\
        y &=& \hat{W} \ast x
      \end{eqnarray}
    Example:
        .. code-block:: python
            import numpy as np
            import nnabla as nn
            import nnabla.functions as F
            import nnabla.parametric_functions as PF
            rng = np.random.RandomState(313)
            x = nn.Variable.from_numpy_array(rng.randn(*(32, 16, 3, 3)))
            # For convolution:
            def ws_callback_conv(w):
                return F.weight_standardization(w, channel_axis=0)
            y = PF.convolution(x, 10, (2, 2), apply_w=ws_callback_conv)
            # For affine:
            def ws_callback_affine(w): 
                return F.weight_standardization(w, channel_axis=1)
            y = PF.affine(x, 10, apply_w=ws_callback_affine)
    References:
      * `Siyuan Qiao, Huiyu Wang, Chenxi Liu, Wei Shen, Alan Yuille, Weight Standardization
        <https://arxiv.org/pdf/1903.10520v1.pdf>`_
    Args:
        w (Variable): A weight variable.
        channel_axis (int): An axis for output channel. Default value is 0 which assumes the weights of convolution.
        eps (float): Tiny value to avoid zero division by std.
        output_stat(bool): If true, the batch statistics of mean and variance.
    Returns:
        * :obj:`~nnabla.Variable`: Standardized output weight.
        * :obj:`~nnabla.Variable`: Mean (if ``output_stat=True`)
        * :obj:`~nnabla.Variable`: Std (if ``output_stat=True`)
    """
    # check channel axis
    _check_axis(len(w.shape), channel_axis)

    return tensor_normalization(w, channel_axis, beta=None, gamma=None, eps=eps, output_stat=output_stat)


def layer_normalization(x, beta, gamma, batch_axis=0, eps=1e-05, output_stat=False):
    r"""
    Applies Layer Normalization over an input tensor, which is defined as:

    .. math::
      \begin{eqnarray}
        \mu^l &=& \frac{1}{H} \sum_{i=1}^{H} x_i^l \\
        \sigma^l &=& \sqrt{\frac{1}{H} \sum_{i=1}^{H} \left(x_i^l - \mu^l\right)^2 + \epsilon} \\
        y &=& \frac{x - \mu^l}{\sigma^l} \gamma + \beta
      \end{eqnarray}

    where :math:`x` and :math:`y` are input and output variable,
    :math:`\mu^l` and :math:`\sigma^l` are the mean and std of each layer which is separately calculated for each batch,
    and :math:`\beta` and :math:`\gamma` are adaptive biases and gains.

    If the input shape is [B, C, H, W] (= batch_axis=0), the shape of calculated mean and std are [B, 1, 1, 1]

    References:

        * `Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton, Layer Normalization.
          <https://arxiv.org/abs/1607.06450>`_

    Args:
        x (Variable): An input variable.
        beta (Variable or None): An Adaptive biases. If None, the bias term is omitted.
        gamma (Variable or None): An Adaptive gains. If None, the scale term is omitted.
        batch_axis (int or repeated int): Axes mean and variance are taken.
        eps (float): Tiny value to avoid zero division by std.
        output_stat(bool): If true, calculated mean and variance are also returned.

    Returns:
        * :obj:`~nnabla.Variable`: output variable which is normalized its statics and rescaled by alpha and beta.
        * :obj:`~nnabla.Variable`: Mean (if `output_stat=True`).
        * :obj:`~nnabla.Variable`: Std (if `output_stat=True`)
    """
    from .function_bases import layer_normalization as layer_normalization_base
    n_outputs = 3 if output_stat else 1

    batch_axis = _force_list(batch_axis)

    no_scale = gamma is None
    no_bias = beta is None

    return layer_normalization_base(x, beta=beta, gamma=gamma, batch_axis=batch_axis, eps=eps, no_scale=no_scale, no_bias=no_bias, n_outputs=n_outputs)


def _layer_normalization_v1(x, beta, gamma, batch_axis=0, eps=1e-05, output_stat=False):
    r"""
    Applies Layer Normalization over an input tensor, which is defined as:
    .. math::
      \begin{eqnarray}
        \mu^l &=& \frac{1}{H} \sum_{i=1}^{H} x_i^l \\
        \sigma^l &=& \sqrt{\frac{1}{H} \sum_{i=1}^{H} \left(x_i^l - \mu^l\right)^2 + \epsilon} \\
        y &=& \frac{x - \mu^l}{\sigma^l} \gamma + \beta
      \end{eqnarray}
    where :math:`x` and :math:`y` are input and output variable,
    :math:`\mu^l` and :math:`\sigma^l` are the mean and std of each layer which is separately calculated for each batch,
    and :math:`\beta` and :math:`\gamma` are adaptive biases and gains.
    If the input shape is [B, C, H, W] (= batch_axis=0), the shape of calculated mean and std are [B, 1, 1, 1]
    References:
        * `Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton, Layer Normalization.
          <https://arxiv.org/abs/1607.06450>`_
    Args:
        x (Variable): An input variable.
        beta (Variable or None): An Adaptive biases. If None, the bias term is omitted.
        gamma (Variable or None): An Adaptive gains. If None, the scale term is omitted.
        batch_axis (int or repeated int): Axes mean and variance are taken.
        eps (float): Tiny value to avoid zero division by std.
        output_stat(bool): If true, calculated mean and variance are also returned.
    Returns:
        * :obj:`~nnabla.Variable`: output variable which is normalized its statics and rescaled by alpha and beta.
        * :obj:`~nnabla.Variable`: Mean (if ``output_stat=True`).
        * :obj:`~nnabla.Variable`: Std (if ``output_stat=True`)
    """

    batch_axis = _check_batch_axis_and_force_list(len(x.shape), batch_axis)

    # cannot broadcast beta & gamma from [1, C, 1, 1] to [N, 1 ,1 ,1],
    # so these adaptation is applied after calling bn with dummy beta & gamma.
    # (Currently bn only accepts the case when reduction shape and adaptive parameter shape are the same.)
    out = tensor_normalization(
        x, batch_axis, beta=None, gamma=None, eps=eps, output_stat=output_stat)

    if not output_stat:
        return _apply_affine(out, scale=gamma, bias=beta)

    y, mean, var = out
    return _apply_affine(y, scale=gamma, bias=beta), mean, var


def instance_normalization(x, beta, gamma, channel_axis=1, batch_axis=0, eps=1e-05, output_stat=False):
    r"""
    Applies Instance Normalization over an input tensor, which is defined as:

    .. math::
      \begin{eqnarray}
        \mu^i &=& \frac{1}{H} \sum_{i=1}^{H} x_i^i \\
        \sigma^i &=& \sqrt{\frac{1}{H} \sum_{i=1}^{H} \left(x_i^i - \mu^i\right)^2 + \epsilon} \\
        y &=& \frac{x - \mu^i}{\sigma^i} \gamma + \beta
      \end{eqnarray}

    where :math:`x` and :math:`y` are input and output variable,
    :math:`\mu^i` and :math:`\sigma^i` are the mean and std of each instance which is separately calculated for each batch and channel,
    and :math:`\gamma` and :math:`\beta` are adaptive gains and biases.

    If the input shape is [B, C, H, W] (= channel_axis=1, batch_axis=0), the shape of calculated mean and std are [B, C, 1, 1]

    References:

        * `Dmitry Ulyanov, Andrea Vedaldi, Victor Lempitsky, Instance Normalization: The Missing Ingredient for Fast Stylization.
          <https://arxiv.org/abs/1607.08022>`_

    Args:
        x (Variable): An input variable.
        beta (Variable or None): An Adaptive biases. If None, the bias term is omitted.
        gamma (Variable or None): An Adaptive gains. If None, the scale term is omitted.
        channel_axis (int): Channel axis.
        batch_axis (int or repeated int): Batch axes.
        eps (float): Tiny value to avoid zero division by std.
        output_stat(bool): If true, the batch statistics of mean and variance.

    Returns:
        * :obj:`~nnabla.Variable`: Normalized output variable.
        * :obj:`~nnabla.Variable`: Mean (if `output_stat=True`)
        * :obj:`~nnabla.Variable`: Std (if `output_stat=True`)
    """
    from .function_bases import instance_normalization as instance_normalization_base
    n_outputs = 3 if output_stat else 1

    batch_axis = _force_list(batch_axis)

    no_scale = gamma is None
    no_bias = beta is None

    return instance_normalization_base(x, beta=beta, gamma=gamma, channel_axis=channel_axis, batch_axis=batch_axis, eps=eps, no_scale=no_scale, no_bias=no_bias, n_outputs=n_outputs)


def _instance_normalization_v1(x, beta, gamma, channel_axis=1, batch_axis=0, eps=1e-05, output_stat=False):
    r"""
    Applies Instance Normalization over an input tensor, which is defined as:

    .. math::
      \begin{eqnarray}
        \mu^i &=& \frac{1}{H} \sum_{i=1}^{H} x_i^i \\
        \sigma^i &=& \sqrt{\frac{1}{H} \sum_{i=1}^{H} \left(x_i^i - \mu^i\right)^2 + \epsilon} \\
        y &=& \frac{x - \mu^i}{\sigma^i} \gamma + \beta
      \end{eqnarray}

    where :math:`x` and :math:`y` are input and output variable,
    :math:`\mu^i` and :math:`\sigma^i` are the mean and std of each instance which is separately calculated for each batch and channel,
    and :math:`\gamma` and :math:`\beta` are adaptive gains and biases.

    If the input shape is [B, C, H, W] (= channel_axis=1, batch_axis=0), the shape of calculated mean and std are [B, C, 1, 1]

    References:

        * `Dmitry Ulyanov, Andrea Vedaldi, Victor Lempitsky, Instance Normalization: The Missing Ingredient for Fast Stylization.
          <https://arxiv.org/abs/1607.08022>`_

    Args:
        x (Variable): An input variable.
        beta (Variable): An Adaptive biases.
        gamma (Variable): An Adaptive gains.
        channel_axis (int): Channel axis.
        batch_axis (int or repeated int): Batch axes.
        eps (float): Tiny value to avoid zero division by std.
        output_stat(bool): If true, the batch statistics of mean and variance.

    Returns:
        * :obj:`~nnabla.Variable`: Normalized output variable.
        * :obj:`~nnabla.Variable`: Mean (if ``output_stat=True`)
        * :obj:`~nnabla.Variable`: Std (if ``output_stat=True`)
    """

    # check channel axis
    _check_axis(len(x.shape), channel_axis)

    # check batch axis
    batch_axis = _check_batch_axis_and_force_list(len(x.shape), batch_axis)

    # check whether broadcast is needed or not.
    # Unlike layer_norm and group_norm, only instance_norm can use bn scale bias & scale adaptation
    # by broadcasting channel axis to channel * batch axis. (like [1, C, 1, 1] -> [N, C, 1, 1])

    adapt_shape = [1 for _ in range(len(x.shape))]
    for baxis in batch_axis:
        adapt_shape[baxis] = x.shape[baxis]
    adapt_shape[channel_axis] = x.shape[channel_axis]
    adapt_shape = tuple(adapt_shape)

    if beta is not None and beta.shape != adapt_shape:
        assert beta.shape[channel_axis] == adapt_shape[channel_axis],\
            "channel size of beta: {} != channel size of x ({}).".format(beta.shape[channel_axis],
                                                                         adapt_shape[channel_axis])
        beta = broadcast(beta, shape=adapt_shape)

    if gamma is not None and gamma.shape != adapt_shape:
        assert gamma.shape[channel_axis] == adapt_shape[channel_axis], \
            "channel size of gamma: {} != channel size of x ({}).".format(gamma.shape[channel_axis],
                                                                          adapt_shape[channel_axis])
        gamma = broadcast(gamma, shape=adapt_shape)

    return tensor_normalization(x, batch_axis + [channel_axis, ], beta, gamma, eps, output_stat)


def group_normalization(x, beta, gamma, num_groups, channel_axis=1, batch_axis=0, eps=1e-05, output_stat=False):
    r"""
    Applies Group Normalization over an input tensor, which is defined as:

    .. math::
      \begin{eqnarray}
        \mu^g &=& \frac{1}{H} \sum_{i=1}^{H} x_i^g \\
        \sigma^g &=& \sqrt{\frac{1}{H} \sum_{i=1}^{H} \left(x_i^g - \mu^g\right)^2 + \epsilon} \\
        y &=& \frac{x - \mu^g}{\sigma^g} \gamma + \beta
      \end{eqnarray}

    where :math:`x` and :math:`y` are input and output variable,
    :math:`\mu^g` and :math:`\sigma^g` are the mean and std of each group which contains `num_channels / num_groups` channels,
    and :math:`\gamma` and :math:`\beta` are adaptive gains and biases.

    The input channels, specified by :attr:`channel_axis`, are separated into :attr:`num_groups` groups,
    and the mean and std are calculated over the each group.
    For example, if the input shape is [B, C, H, W] (= channel_axis=1, batch_axis=0),
    an input variable is once reshaped to [B, num_groups, C / num_groups, H, W]
    and standardize by its mean and std whose shapes are [B, num_groups, 1, 1, 1].
    Finally, an output variable is reshaped again to the original input shape (= [B, C, H, W] in the case above).

    References:

        * `Yuxin Wu, Kaiming He, Group Normalization.
          <https://arxiv.org/abs/1803.08494>`_

    Args:
        x (Variable): An input variable.
        beta (Variable or None): An Adaptive biases. If None, the bias term is omitted.
        gamma (Variable or None): An Adaptive gains. If None, the scale term is omitted.
        num_groups (int): A number of groups. The channel dim of 'x' must be integer multiple of `num_groups`.
        channel_axis (int): Channel axis.
        batch_axis (int or repeated int): Batch axes.
        eps (float): Tiny value to avoid zero division by std.
        output_stat(bool): If true, the batch statistics of mean and variance.

    Returns:
        * :obj:`~nnabla.Variable`: Normalized output variable.
        * :obj:`~nnabla.Variable`: Mean (if `output_stat=True`)
        * :obj:`~nnabla.Variable`: Std (if `output_stat=True`)
    """
    from .function_bases import group_normalization as group_normalization_base
    n_outputs = 3 if output_stat else 1

    batch_axis = _force_list(batch_axis)

    no_scale = gamma is None
    no_bias = beta is None

    return group_normalization_base(x, beta=beta, gamma=gamma, num_groups=num_groups, channel_axis=channel_axis, batch_axis=batch_axis, eps=eps, no_scale=no_scale, no_bias=no_bias, n_outputs=n_outputs)


def _group_normalization_v1(x, beta, gamma, num_groups, channel_axis=1, batch_axis=0, eps=1e-05, output_stat=False):
    r"""
    Applies Group Normalization over an input tensor, which is defined as:

    .. math::
      \begin{eqnarray}
        \mu^g &=& \frac{1}{H} \sum_{i=1}^{H} x_i^g \\
        \sigma^g &=& \sqrt{\frac{1}{H} \sum_{i=1}^{H} \left(x_i^g - \mu^g\right)^2 + \epsilon} \\
        y &=& \frac{x - \mu^g}{\sigma^g} \gamma + \beta
      \end{eqnarray}

    where :math:`x` and :math:`y` are input and output variable,
    :math:`\mu^g` and :math:`\sigma^g` are the mean and std of each group which contains `num_channels / num_groups` channels,
    and :math:`\gamma` and :math:`\beta` are adaptive gains and biases.

    The input channels, specified by :attr:`channel_axis`, are separated into :attr:`num_groups` groups,
    and the mean and std are calculated over the each group.
    For example, if the input shape is [B, C, H, W] (= channel_axis=1, batch_axis=0),
    an input variable is once reshaped to [B, num_groups, C / num_groups, H, W]
    and standardize by its mean and std whose shapes are [B, num_groups, 1, 1, 1].
    Finally, an output variable is reshaped again to the original input shape (= [B, C, H, W] in the case above).

    References:

        * `Yuxin Wu, Kaiming He, Group Normalization.
          <https://arxiv.org/abs/1803.08494>`_

    Args:
        x (Variable): An input variable.
        beta (Variable or None): An Adaptive biases. If None, the bias term is omitted.
        gamma (Variable or None): An Adaptive gains. If None, the scale term is omitted.
        num_groups (int): A number of groups. The channel dim of 'x' must be integer multiple of `num_groups`.
        channel_axis (int): Channel axis.
        batch_axis (int or repeated int): Batch axes.
        eps (float): Tiny value to avoid zero division by std.
        output_stat(bool): If true, the batch statistics of mean and variance.

    Returns:
        * :obj:`~nnabla.Variable`: Normalized output variable.
        * :obj:`~nnabla.Variable`: Mean (if ``output_stat=True`)
        * :obj:`~nnabla.Variable`: Std (if ``output_stat=True`)
    """
    _check_axis(len(x.shape), channel_axis)

    cdim = x.shape[channel_axis]

    if cdim % num_groups > 0:
        raise ValueError(
            "Channel dim ({}) must be integer multiple of num_groups ({}).".format(cdim, num_groups))

    shape = x.shape[:channel_axis] + (num_groups, int(cdim / num_groups))
    if channel_axis < len(x.shape) - 1:
        shape += x.shape[channel_axis + 1:]

    # create dummy adaptive constants and pass these to BN.
    # GN normalizes input along group axis but applies adaptive rescaling along the original channel axis,
    # so we have to apply adaptive scaling after reshaping the output from batch normalization.
    # (Currently bn only accepts the case when reduction shape and adaptive parameter shape are the same.)

    out = instance_normalization(x.reshape(shape), beta=None, gamma=None,
                                 channel_axis=channel_axis, batch_axis=batch_axis, eps=eps, output_stat=output_stat)

    if not output_stat:
        return _apply_affine(out.reshape(x.shape), scale=gamma, bias=beta)

    y, mean, var = out
    return _apply_affine(y.reshape(x.shape), scale=gamma, bias=beta), mean, var
