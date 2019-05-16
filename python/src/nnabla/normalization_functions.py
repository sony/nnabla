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

    return [i for i in range(ndim) if i not in axes]


def batch_normalization(x, beta, gamma, mean, variance, axes=[1], decay_rate=0.9, eps=1e-05, batch_stat=True,
                        output_stat=False, n_outputs=None):
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


def tensor_normalization(x, axes, eps=1e-05, output_stat=False):
    r"""
    General function for tensor normalization.
    Input variable `x` is normalized by mean and std calculated by `x` itself.
    Mean and std are taken by all `axes`.
    For example, if the input shape is (B, C, H, W) and axes is [1, 2, 3],
     the shape of calculated mean and std are (B, 1, 1 ,1).

    Args:
        x (Variable): N-D array of input variable.
        axes (repeated int): Axes mean and variance are taken.
        eps (float): Tiny value to avoid zero division by std.
        output_stat(bool): It true, the batch statistics of mean and variance.
        will be returned as Variables. They are also differentiable.

    Returns:
        * :obj:`~nnabla.Variable`: Normalized output variable.
        * :obj:`~nnabla.Variable`: Mean (if ``output_stat=True`)
        * :obj:`~nnabla.Variable`: Std (if ``output_stat=True`)

    """
    from .function_bases import mean as mean_base

    x_mean = mean_base(x, axes, keep_dims=True)
    subtracted = x - x_mean
    x_std = mean_base(subtracted ** 2, axes, keep_dims=True) ** 0.5

    if output_stat:
        return subtracted / (x_std + eps), x_mean, x_std

    return subtracted / (x_std + eps)


def weight_standardization(w, channel_axis=0, eps=1e-05, output_stat=False):
    r"""
    Applies Weight Standardization over an input weight, which is defined as:

    .. math::
      \begin{eqnarray}
        \mu_{W_i} &=& \frac{1}{I} \sum_{j=1}^{I} W_{ij} \\
        \sigma_{W_i} &=& \sqrt{\frac{1}{I} \sum_{i=1}^{I} \left(W_{ij} - \mu_{W_{i}}\right)^2} \\
        \hat{W_{ij}} &=& \frac{W_{ij} - \mu_{W_i}}{\sigma_{W_i} + \epsilon} \\
        y &=& \hat{W} \ast x
      \end{eqnarray}

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

    axes = _get_axes_excluding(len(w.shape), channel_axis)

    return tensor_normalization(w, axes, eps, output_stat)


def layer_normalization(x, beta, gamma, batch_axis=0, eps=1e-05, output_stat=False):
    r"""
    Applies Layer Normalization over an input tensor, which is defined as:

    .. math::
      \begin{eqnarray}
        \mu^l &=& \frac{1}{H} \sum_{i=1}^{H} x_i^l \\
        \sigma^l &=& \sqrt{\frac{1}{H} \sum_{i=1}^{H} \left(x_i^l - \mu^l\right)^2} \\
        y &=& \frac{x - \mu^l}{\sigma^l + \epsilon} \gamma + \beta
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
        beta (Variable): An Adaptive biases.
        gamma (Variable): An Adaptive gains.
        batch_axis (int or repeated int): Axes mean and variance are taken.
        eps (float): Tiny value to avoid zero division by std.
        output_stat(bool): If true, calculated mean and variance are also returned.

    Returns:
        * :obj:`~nnabla.Variable`: output variable which is normalized its statics and rescaled by alpha and beta.
        * :obj:`~nnabla.Variable`: Mean (if ``output_stat=True`).
        * :obj:`~nnabla.Variable`: Std (if ``output_stat=True`)
    """

    batch_axis = _check_batch_axis_and_force_list(len(x.shape), batch_axis)

    axes = _get_axes_excluding(len(x.shape), batch_axis)

    if output_stat:
        out, mean, std = tensor_normalization(x, axes, eps, output_stat)
        return out * gamma + beta, mean, std

    return tensor_normalization(x, axes, eps, output_stat) * gamma + beta


def instance_normalization(x, beta, gamma, channel_axis=1, batch_axis=0, eps=1e-05, output_stat=False):
    r"""
    Applies Instance Normalization over an input tensor, which is defined as:

    .. math::
      \begin{eqnarray}
        \mu^i &=& \frac{1}{H} \sum_{i=1}^{H} x_i^i \\
        \sigma^i &=& \sqrt{\frac{1}{H} \sum_{i=1}^{H} \left(x_i^i - \mu^i\right)^2} \\
        y &=& \frac{x - \mu^i}{\sigma^i + \epsilon} \gamma + \beta
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

    axes = _get_axes_excluding(len(x.shape), [channel_axis, ] + batch_axis)

    if output_stat:
        out, mean, std = tensor_normalization(x, axes, eps, output_stat)

        return out * gamma + beta, mean, std

    return tensor_normalization(x, axes, eps, output_stat) * gamma + beta


def group_normalization(x, beta, gamma, num_groups, channel_axis=1, batch_axis=0, eps=1e-05, output_stat=False):
    r"""
    Applies Group Normalization over an input tensor, which is defined as:

    .. math::
      \begin{eqnarray}
        \mu^g &=& \frac{1}{H} \sum_{i=1}^{H} x_i^g \\
        \sigma^g &=& \sqrt{\frac{1}{H} \sum_{i=1}^{H} \left(x_i^g - \mu^g\right)^2} \\
        y &=& \frac{x - \mu^g}{\sigma^g + \epsilon} \gamma + \beta
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
        beta (Variable): An Adaptive biases.
        gamma (Variable): An Adaptive gains.
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

    if output_stat:
        out, mu, sigma = instance_normalization(
            x.reshape(shape), beta, gamma, channel_axis, batch_axis, eps, output_stat)

        return out.reshape(x.shape), mu, sigma

    return instance_normalization(x.reshape(shape), beta, gamma, channel_axis, batch_axis, eps,
                                  output_stat).reshape(x.shape)
