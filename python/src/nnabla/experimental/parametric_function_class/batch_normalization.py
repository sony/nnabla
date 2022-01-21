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


import nnabla as nn
import nnabla.functions as F
from nnabla.initializer import (
    ConstantInitializer)

from .module import Module


class BatchNormalization(Module):
    """
    Batch normalization layer.

    .. math::

        \\begin{array}{lcl}
        \\mu &=& \\frac{1}{M} \\sum x_i\\\\
        \\sigma^2 &=& \\frac{1}{M} \\left(\\sum x_i - \\mu\\right)^2\\\\
        \\hat{x}_i &=& \\frac{x_i - \\mu}{\\sqrt{\\sigma^2 + \\epsilon }}\\\\
        y_i &= & \\hat{x}_i \\gamma + \\beta.
        \\end{array}

    where :math:`x_i, y_i` are the inputs.
    In testing, the mean and variance computed by moving average calculated during training are used.

    Args:
        inp (~nnabla.Variable): N-D array of input.
        axes (:obj:`tuple` of :obj:`int`):
            Mean and variance for each element in ``axes`` are calculated using
            elements on the rest axes. For example, if an input is 4 dimensions,
            and ``axes`` is ``[1]``,  batch mean is calculated as
            ``np.mean(inp.d, axis=(0, 2, 3), keepdims=True)``
            (using numpy expression as an example).
        decay_rate (float): Decay rate of running mean and variance.
        eps (float): Tiny value to avoid zero division by std.
        batch_stat (bool): Use mini-batch statistics rather than running ones.
        output_stat (bool): Output batch mean and variance.
        fix_parameters (bool): When set to `True`, the beta and gamma will not be updated.
        param_init (dict):
            Parameter initializers can be set with a dict. A key of the dict must
            be ``'beta'``, ``'gamma'``, ``'mean'`` or ``'var'``.
            A value of the dict must be an :obj:`~nnabla.initializer.Initializer`
            or a :obj:`numpy.ndarray`.
            E.g. ``{'beta': ConstantInitializer(0), 'gamma': np.ones(gamma_shape) * 2}``.

    Returns:
        :class:`~nnabla.Variable`: N-D array.

    References:

        - Ioffe and Szegedy, Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. https://arxiv.org/abs/1502.03167

    The shape of parameters has the same number of dimensions with the input
    data, and the shapes in ``axes`` has the same dimensions with the input, while the rest has ``1``.
    If an input is 4-dim and ``axes=[1]``, the parameter shape will be
    ``param_shape  = np.mean(inp.d, axis=(0, 2, 3), keepdims=True).shape``
    (using numpy expression as an example).

    """

    def __init__(self, n_features, n_dims, axes=[1], decay_rate=0.9, eps=1e-5,
                 batch_stat=True, output_stat=False, fix_parameters=False,
                 param_init=None):
        assert len(axes) == 1
        shape_stat = [1 for _ in range(n_dims)]
        shape_stat[axes[0]] = n_features

        if param_init is None:
            param_init = {}
        beta_init = param_init.get('beta', ConstantInitializer(0))
        gamma_init = param_init.get('gamma', ConstantInitializer(1))
        mean_init = param_init.get('mean', ConstantInitializer(0))
        var_init = param_init.get('var', ConstantInitializer(1))

        beta = nn.Variable.from_numpy_array(
            beta_init(shape_stat)).apply(need_grad=not fix_parameters)
        gamma = nn.Variable.from_numpy_array(gamma_init(
            shape_stat)).apply(need_grad=not fix_parameters)
        mean = nn.Variable.from_numpy_array(mean_init(shape_stat))
        var = nn.Variable.from_numpy_array(var_init(shape_stat))

        self.beta = beta
        self.gamma = gamma
        self.mean = mean
        self.var = var
        self.axes = axes
        self.decay_rate = decay_rate
        self.eps = eps
        self.output_stat = output_stat

    def __call__(self, inp, test=False):
        return F.batch_normalization(inp, self.beta, self.gamma, self.mean, self.var, self.axes,
                                     self.decay_rate, self.eps, not test, self.output_stat)


class BatchNorm1d(BatchNormalization):
    """
    Batch normalization layer for 3d-Array or 3d-Variable. This is typically used together with `Conv1d`.

    .. math::

        \\begin{array}{lcl}
        \\mu &=& \\frac{1}{M} \\sum x_i\\\\
        \\sigma^2 &=& \\frac{1}{M} \\left(\\sum x_i - \\mu\\right)^2\\\\
        \\hat{x}_i &=& \\frac{x_i - \\mu}{\\sqrt{\\sigma^2 + \\epsilon }}\\\\
        y_i &= & \\hat{x}_i \\gamma + \\beta.
        \\end{array}

    where :math:`x_i, y_i` are the inputs.
    In testing, the mean and variance computed by moving average calculated during training are used.

    Args:
        inp (~nnabla.Variable): N-D array of input.
        axes (:obj:`tuple` of :obj:`int`):
            Mean and variance for each element in ``axes`` are calculated using
            elements on the rest axes. For example, if an input is 4 dimensions,
            and ``axes`` is ``[1]``,  batch mean is calculated as
            ``np.mean(inp.d, axis=(0, 2, 3), keepdims=True)``
            (using numpy expression as an example).
        decay_rate (float): Decay rate of running mean and variance.
        eps (float): Tiny value to avoid zero division by std.
        batch_stat (bool): Use mini-batch statistics rather than running ones.
        output_stat (bool): Output batch mean and variance.
        fix_parameters (bool): When set to `True`, the beta and gamma will not be updated.
        param_init (dict):
            Parameter initializers can be set with a dict. A key of the dict must
            be ``'beta'``, ``'gamma'``, ``'mean'`` or ``'var'``.
            A value of the dict must be an :obj:`~nnabla.initializer.Initializer`
            or a :obj:`numpy.ndarray`.
            E.g. ``{'beta': ConstantInitializer(0), 'gamma': np.ones(gamma_shape) * 2}``.

    Returns:
        :class:`~nnabla.Variable`: N-D array.

    References:

        - Ioffe and Szegedy, Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. https://arxiv.org/abs/1502.03167

    The shape of parameters has the same number of dimensions with the input
    data, and the shapes in ``axes`` has the same dimensions with the input, while the rest has ``1``.
    If an input is 4-dim and ``axes=[1]``, the parameter shape will be
    ``param_shape  = np.mean(inp.d, axis=(0, 2, 3), keepdims=True).shape``
    (using numpy expression as an example).

    """

    def __init__(self, n_features, axes=[1], decay_rate=0.9, eps=1e-5,
                 batch_stat=True, output_stat=False, fix_parameters=False,
                 param_init=None):
        super(BatchNorm1d, self).__init__(n_features, 3, axes=axes, decay_rate=decay_rate,
                                          eps=1e-5, batch_stat=batch_stat, output_stat=output_stat,
                                          fix_parameters=fix_parameters,
                                          param_init=param_init)


class BatchNorm2d(BatchNormalization):
    """
    Batch normalization layer for 4d-Array or 4d-Variable. This is typically used together with `Conv2d`.

    .. math::

        \\begin{array}{lcl}
        \\mu &=& \\frac{1}{M} \\sum x_i\\\\
        \\sigma^2 &=& \\frac{1}{M} \\left(\\sum x_i - \\mu\\right)^2\\\\
        \\hat{x}_i &=& \\frac{x_i - \\mu}{\\sqrt{\\sigma^2 + \\epsilon }}\\\\
        y_i &= & \\hat{x}_i \\gamma + \\beta.
        \\end{array}

    where :math:`x_i, y_i` are the inputs.
    In testing, the mean and variance computed by moving average calculated during training are used.

    Args:
        inp (~nnabla.Variable): N-D array of input.
        axes (:obj:`tuple` of :obj:`int`):
            Mean and variance for each element in ``axes`` are calculated using
            elements on the rest axes. For example, if an input is 4 dimensions,
            and ``axes`` is ``[1]``,  batch mean is calculated as
            ``np.mean(inp.d, axis=(0, 2, 3), keepdims=True)``
            (using numpy expression as an example).
        decay_rate (float): Decay rate of running mean and variance.
        eps (float): Tiny value to avoid zero division by std.
        batch_stat (bool): Use mini-batch statistics rather than running ones.
        output_stat (bool): Output batch mean and variance.
        fix_parameters (bool): When set to `True`, the beta and gamma will not be updated.
        param_init (dict):
            Parameter initializers can be set with a dict. A key of the dict must
            be ``'beta'``, ``'gamma'``, ``'mean'`` or ``'var'``.
            A value of the dict must be an :obj:`~nnabla.initializer.Initializer`
            or a :obj:`numpy.ndarray`.
            E.g. ``{'beta': ConstantInitializer(0), 'gamma': np.ones(gamma_shape) * 2}``.

    Returns:
        :class:`~nnabla.Variable`: N-D array.

    References:

        - Ioffe and Szegedy, Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. https://arxiv.org/abs/1502.03167

    The shape of parameters has the same number of dimensions with the input
    data, and the shapes in ``axes`` has the same dimensions with the input, while the rest has ``1``.
    If an input is 4-dim and ``axes=[1]``, the parameter shape will be
    ``param_shape  = np.mean(inp.d, axis=(0, 2, 3), keepdims=True).shape``
    (using numpy expression as an example).

    """

    def __init__(self, n_features, axes=[1], decay_rate=0.9, eps=1e-5,
                 batch_stat=True, output_stat=False, fix_parameters=False,
                 param_init=None):
        super(BatchNorm2d, self).__init__(n_features, 4, axes=axes, decay_rate=decay_rate,
                                          eps=1e-5, batch_stat=batch_stat, output_stat=output_stat,
                                          fix_parameters=fix_parameters,
                                          param_init=param_init)


class BatchNorm3d(BatchNormalization):
    """
    Batch normalization layer for 5d-Array or 5d-Variable. This is typically used together with `Conv3d`.

    .. math::

        \\begin{array}{lcl}
        \\mu &=& \\frac{1}{M} \\sum x_i\\\\
        \\sigma^2 &=& \\frac{1}{M} \\left(\\sum x_i - \\mu\\right)^2\\\\
        \\hat{x}_i &=& \\frac{x_i - \\mu}{\\sqrt{\\sigma^2 + \\epsilon }}\\\\
        y_i &= & \\hat{x}_i \\gamma + \\beta.
        \\end{array}

    where :math:`x_i, y_i` are the inputs.
    In testing, the mean and variance computed by moving average calculated during training are used.

    Args:
        inp (~nnabla.Variable): N-D array of input.
        axes (:obj:`tuple` of :obj:`int`):
            Mean and variance for each element in ``axes`` are calculated using
            elements on the rest axes. For example, if an input is 4 dimensions,
            and ``axes`` is ``[1]``,  batch mean is calculated as
            ``np.mean(inp.d, axis=(0, 2, 3), keepdims=True)``
            (using numpy expression as an example).
        decay_rate (float): Decay rate of running mean and variance.
        eps (float): Tiny value to avoid zero division by std.
        batch_stat (bool): Use mini-batch statistics rather than running ones.
        output_stat (bool): Output batch mean and variance.
        fix_parameters (bool): When set to `True`, the beta and gamma will not be updated.
        param_init (dict):
            Parameter initializers can be set with a dict. A key of the dict must
            be ``'beta'``, ``'gamma'``, ``'mean'`` or ``'var'``.
            A value of the dict must be an :obj:`~nnabla.initializer.Initializer`
            or a :obj:`numpy.ndarray`.
            E.g. ``{'beta': ConstantInitializer(0), 'gamma': np.ones(gamma_shape) * 2}``.

    Returns:
        :class:`~nnabla.Variable`: N-D array.

    References:

        - Ioffe and Szegedy, Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. https://arxiv.org/abs/1502.03167

    The shape of parameters has the same number of dimensions with the input
    data, and the shapes in ``axes`` has the same dimensions with the input, while the rest has ``1``.
    If an input is 4-dim and ``axes=[1]``, the parameter shape will be
    ``param_shape  = np.mean(inp.d, axis=(0, 2, 3), keepdims=True).shape``
    (using numpy expression as an example).

    """

    def __init__(self, n_features, axes=[1], decay_rate=0.9, eps=1e-5,
                 batch_stat=True, output_stat=False, fix_parameters=False,
                 param_init=None):
        super(BatchNorm3d, self).__init__(n_features, 5, axes=axes, decay_rate=decay_rate,
                                          eps=1e-5, batch_stat=batch_stat, output_stat=output_stat,
                                          fix_parameters=fix_parameters,
                                          param_init=param_init)
