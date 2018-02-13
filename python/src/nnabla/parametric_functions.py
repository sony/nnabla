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

from six import exec_
import numpy as np

import nnabla as nn
import nnabla.functions as F
from nnabla.parameter import get_parameter_or_create, get_parameter
from nnabla.initializer import (
    calc_uniform_lim_glorot,
    ConstantInitializer, NormalInitializer, UniformInitializer)


def parametric_function_api(scope_name=None):
    """Decorator for parametric functions.

    The decorated function is always called under
    a parameter scope ``scope_name``.
    Also, the decorator adds an additional argument ``name`` (:obj:`str`,
    default is ``None``) at the end. If ``name`` is specified, the
    scope ``scope_name`` comes under a scope ``name``. This feature
    could reduce vertical space usage of the source code.
    Any parametric function should be decoreated by this.

    Args:
        scope_name (str, optional): The original function will be called
            under a parameter scope named by ``scope_name``.

    Returns:
        function: A decorated parametric function.

    """
    if scope_name is None:
        scope_name = name

    def parametric_function_api_inside(func):
        import inspect

        name = func.__name__
        doc = func.__doc__ + """
    Note:

        If the ``name`` option is passed, the parameters become wrapped inside the parameter scope
        with the specified name, yielding the same results as the following code.
        This can be used to simplify the code.

        .. code-block:: python

            with parametric_scope(name):
                output = {name}(<args>)

        """.format(name=name)

        spec = inspect.getargspec(func)
        defaults = spec.defaults
        if defaults is None:
            defaults = tuple()  # None will be appended later
        signature = inspect.formatargspec(
            spec.args + ['name'],
            spec.varargs, spec.keywords,
            defaults + (None,))
        shortsignature = inspect.formatargspec(
            spec.args, spec.varargs, spec.keywords, None)

        # Check required argument
        assert 'fix_parameters' in spec.args, \
            "A parametric function must take `fix_parameters` as an argument." \
            " `{}{}` doesn't have it.".format(name, signature)

        code = """
def {name}{signature}:
    if name is None:
        with parameter_scope(scope_name):
            return func{shortsignature}
    with parameter_scope(name):
        with parameter_scope(scope_name):
            return func{shortsignature}
        """.format(**locals())
        execdict = dict(
            func=func, parameter_scope=nn.parameter_scope, scope_name=scope_name)
        exec_(code, execdict)
        newfunc = execdict[name]
        newfunc.__doc__ = doc
        newfunc.__parametric_function_api_base__ = func
        newfunc.__scope_name__ = scope_name
        newfunc.__module__ = __name__
        return newfunc
    return parametric_function_api_inside


@parametric_function_api("affine")
def affine(inp, n_outmaps,
           base_axis=1,
           w_init=None, b_init=None,
           fix_parameters=False, rng=None, with_bias=True):
    """
    The affine layer, also known as the fully connected layer. Computes

    .. math::
        {\\mathbf y} = {\\mathbf A} {\\mathbf x} + {\\mathbf b}.

    where :math:`{\\mathbf x}, {\\mathbf y}` are the inputs and outputs respectively,
    and :math:`{\\mathbf A}, {\\mathbf b}` are constants.

    Args:
        inp (~nnabla.Variable): Input N-D array with shape (:math:`M_0 \\times \ldots \\times M_{B-1} \\times D_B \\times \ldots \\times D_N`). Dimensions before and after base_axis are flattened as if it is a matrix.
        n_outmaps (:obj:`int` or :obj:`tuple` of :obj:`int`): Number of output neurons per data.
        base_axis (int): Dimensions up to `base_axis` are treated as the sample dimensions.
        w_init (~nnabla.initializer.BaseInitializer): Initializer for weight.
        b_init (~nnabla.initializer.BaseInitializer): Initializer for bias.
        fix_parameters (bool): When set to `True`, the weights and biases will not be updated.
        rng (numpy.random.RandomState): Random generator for Initializer.
        with_bias (bool): Specify whether to include the bias term.

    Returns:
        :class:`~nnabla.Variable`: :math:`(B + 1)`-D array. (:math:`M_0 \\times \ldots \\times M_{B-1} \\times L`)f

    """
    if not hasattr(n_outmaps, '__iter__'):
        n_outmaps = [n_outmaps]
    n_outmaps = list(n_outmaps)
    n_outmap = int(np.prod(n_outmaps))
    if w_init is None:
        inmaps = np.prod(inp.shape[base_axis:])
        w_init = UniformInitializer(
            calc_uniform_lim_glorot(inmaps, n_outmap), rng=rng)
    if with_bias and b_init is None:
        b_init = ConstantInitializer()
    w = get_parameter_or_create(
        "W", [int(np.prod(inp.shape[base_axis:]))] + n_outmaps,
        w_init, not fix_parameters)
    b = None
    if with_bias:
        b = get_parameter_or_create(
            "b", n_outmaps, b_init, not fix_parameters)
    return F.affine(inp, w, b, base_axis)


@parametric_function_api("bicon_affine")
def binary_connect_affine(inp, n_outmaps,
                          base_axis=1,
                          w_init=None, wb_init=None, b_init=None,
                          fix_parameters=False, rng=None, with_bias=True):
    """Binary Connect Affine, multiplier-less inner-product.

    Binary Connect Affine is an affine function,
    except the definition of the inner product is modified.
    The input-output relation of this function is as follows:

    .. math::

        y_i = \sum_{i} sign(w_i) x_i.

    Therefore :math:`sign(w_i)` is either :math:`1` or :math:`-1` and the inner product
    simplifies to addition.

    This function should be used together with Batch Normalization.

    References:

        M. Courbariaux, Y. Bengio, and J.-P. David. "BinaryConnect:
        Training Deep Neural Networks with binary weights during propagations."
        Advances in Neural Information Processing Systems. 2015.

    .. note::

        1) if you would like to share weights between some layers, please
        make sure to share the standard, floating value weights (`weight`)
        and not the binarized weights (`binary_weight`)

        2) The weights and the binary weights become synced only after :func:`~nnabla._variable.Variable.forward` is called,
        and not after a call to :func:`~nnabla._variable.Variable.backward`.
        To access the parameters of the network, remember to call :func:`~nnabla._variable.Variable.forward` once before doing so, otherwise the
        float weights and the binary weights will not be in sync.

        3) Quantized values are stored as floating point number for `binary_weight`,
        since this function is only for simulation purposes.

    Args:
        inp (~nnabla.Variable): Input N-D array with shape (:math:`M_0 \\times \ldots \\times M_{B-1} \\times D_B \\times \ldots \\times D_N`). Dimensions before and after base_axis are flattened as if it is a matrix.
        n_outmaps (int or :obj:`tuple` of :obj:`int`): Number of output neurons per data.
        base_axis (int): Dimensions up to `base_axis` are treated as the sample dimensions.
        w_init (~nnabla.initializer.BaseInitializer): Initializer for weight.
        wb_init (~nnabla.initializer.BaseInitializer): Initializer for binary weight.
        b_init (~nnabla.initializer.BaseInitializer): Initializer for bias.
        fix_parameters (bool): When set to `True`, the weights and biases will not be updated.
        rng (numpy.random.RandomState): Random generator for Initializer.

    Returns:
        :class:`~nnabla.Variable`

    """
    if not hasattr(n_outmaps, '__iter__'):
        n_outmaps = [n_outmaps]
    n_outmaps = list(n_outmaps)
    n_outmap = int(np.prod(n_outmaps))
    if w_init is None:
        fan_in = np.prod(inp.shape[base_axis:])
        w_init = UniformInitializer(
            calc_uniform_lim_glorot(fan_in, n_outmap), rng=rng)
    if wb_init is None:
        fan_in = np.prod(inp.shape[base_axis:])
        wb_init = UniformInitializer(
            calc_uniform_lim_glorot(fan_in, n_outmap), rng=rng)
    if b_init is None:
        b_init = ConstantInitializer()
    w = get_parameter_or_create(
        "W", [int(np.prod(inp.shape[base_axis:]))] + n_outmaps,
        w_init, not fix_parameters)
    wb = get_parameter_or_create(
        "Wb", [int(np.prod(inp.shape[base_axis:]))] + n_outmaps,
        wb_init, not fix_parameters)
    b = None
    if with_bias:
        b = get_parameter_or_create(
            "b", n_outmaps, b_init, not fix_parameters)
    return F.binary_connect_affine(inp, w, wb, b, base_axis)


@parametric_function_api("bwn_affine")
def binary_weight_affine(inp, n_outmaps,
                         base_axis=1,
                         w_init=None, wb_init=None, b_init=None,
                         fix_parameters=False, rng=None, with_bias=True):
    """Binary Weight Affine, multiplier-less inner-product with a scale factor.

    Binary Weight Affine is the affine function, but the inner product
    in this function is the following,

    .. math::

        y_j = \\frac{1}{\\|\\mathbf{w}_j\\|_{\\ell_1}} \sum_{i} sign(w_{ji}) x_i

    Therefore :math:`sign(w_{ji})` is either :math:`1` or :math:`-1` and the inner product
    simplifies to addition followed by scaling factor :math:`\\alpha = \\frac{1}{\\|\\mathbf{w}_j\\|_{\\ell_1}}`.
    The number of ::math:`\\alpha` is the outmaps of the affine function.

    References:

        Rastegari, Mohammad, et al. "XNOR-Net: ImageNet Classification Using
        Binary Convolutional Neural Networks." arXiv preprint
        arXiv:1603.05279 (2016).

    .. note::

        1) if you would like to share weights between some layers, please
        make sure to share the standard, floating value weights (`weight`)
        and not the binarized weights (`binary_weight`)

        2) The weights and the binary weights become synced only after :func:`~nnabla._variable.Variable.forward` is called,
        and not after a call to :func:`~nnabla._variable.Variable.backward`.
        To access the parameters of the network, remember to call :func:`~nnabla._variable.Variable.forward` once before doing so, otherwise the
        float weights and the binary weights will not be in sync.

        3) Quantized values are stored as floating point number for `binary_weight`,
        since this function is only for simulation purposes.

    Args:
        inp (~nnabla.Variable): Input N-D array with shape (:math:`M_0 \\times \ldots \\times M_{B-1} \\times D_B \\times \ldots \\times D_N`). Dimensions before and after base_axis are flattened as if it was a matrix.
        n_outmaps (int or :obj:`tuple` of :obj:`int`): Number of output neurons per data.
        base_axis (int): Dimensions up to `base_axis` are treated as the sample dimensions.
        w_init (~nnabla.initializer.BaseInitializer): Initializer for the weight.
        wb_init (~nnabla.initializer.BaseInitializer): Initializer for the binary weight.
        b_init (~nnabla.initializer.BaseInitializer): Initializer for the bias.
        fix_parameters (bool): When set to `True`, the weight and bias will not be updated.
        rng (numpy.random.RandomState): Random generator for Initializer.
        with_bias (bool): Specify whether to include the bias term.

    Returns:
        :class:`~nnabla.Variable`

    """
    if not hasattr(n_outmaps, '__iter__'):
        n_outmaps = [n_outmaps]
    n_outmaps = list(n_outmaps)
    n_outmap = int(np.prod(n_outmaps))
    if w_init is None:
        fan_in = np.prod(inp.shape[base_axis:])
        w_init = UniformInitializer(
            calc_uniform_lim_glorot(fan_in, n_outmap), rng=rng)
    if wb_init is None:
        fan_in = np.prod(inp.shape[base_axis:])
        wb_init = UniformInitializer(
            calc_uniform_lim_glorot(fan_in, n_outmap), rng=rng)
    if b_init is None:
        b_init = ConstantInitializer()
    w = get_parameter_or_create(
        "W", [int(np.prod(inp.shape[base_axis:]))] + n_outmaps,
        w_init, not fix_parameters)
    wb = get_parameter_or_create(
        "Wb", [int(np.prod(inp.shape[base_axis:]))] + n_outmaps,
        wb_init, not fix_parameters)
    alpha = get_parameter_or_create(
        "alpha", n_outmaps, ConstantInitializer(0), False)
    b = None
    if with_bias:
        b = get_parameter_or_create(
            "b", n_outmaps, b_init, not fix_parameters)
    return F.binary_weight_affine(inp, w, wb, alpha, b, base_axis)


@parametric_function_api("inq_affine")
def inq_affine(inp, n_outmaps, base_axis=1, num_bits=4,
               inq_iterations=(), selection_algorithm='random',
               seed=-1, w_init=None, i_init=None, b_init=None,
               fix_parameters=False, rng=None, with_bias=True):
    """Incremental Network Quantization Affine Layer

    During training, the weights are sequentially quantized to power-of-two
    values, which allows the training of a multiplierless network.

    Using `inq_iterations`, one can specify after how many forward passes
    half of the learnable weights are fixed and quantized to powers-of-two.
    After reaching the last value in `inq_iterations`, all weights are fixed.

    For more details, please refer to the reference.

    Reference:
    Zhou A, Yao A, Guo Y, Xu L, Chen Y. Incremental network quantization:
    Towards lossless CNNs with low-precision weights.
    <https://arxiv.org/abs/1702.03044>

    Args:
        inp (~nnabla.Variable): Input N-D array with shape (:math:`M_0 \\times \ldots \\times M_{B-1} \\times D_B \\times \ldots \\times D_N`). Dimensions before and after base_axis are flattened as if it was a matrix.
        n_outmaps (int or :obj:`tuple` of :obj:`int`): Number of output neurons per data.
        base_axis (int): Dimensions up to `base_axis` are treated as the sample dimensions.
        num_bits (int): Number of bits per weight. Value has to be larger than 1 as one bit is already used to code the value "0"
        inq_iterations (tuple of int): Tuple of iteration numbers at which we fix half of the weights.
        selection_algorithm (str): Chooses algorithm that is used to decide which weights are fixed. ("largest_abs" ... fix weights with largest absolute value, "random" ... fix weights randomly)
        seed (int): Random seed for INQ algorithm
        w_init (~nnabla.initializer.BaseInitializer): Initializer for the weight.
        i_init (~nnabla.initializer.BaseInitializer): Initializer for the indicators (0 ... learnable, 1 ... fixed).
        b_init (~nnabla.initializer.BaseInitializer): Initializer for the bias.
        fix_parameters (bool): When set to `True`, the weight and bias will not be updated.
        rng (numpy.random.RandomState): Random generator for Initializer.
        with_bias (bool): Specify whether to include the bias term.

    Returns:
        :class:`~nnabla.Variable`

    """
    if not hasattr(n_outmaps, '__iter__'):
        n_outmaps = [n_outmaps]
    n_outmaps = list(n_outmaps)
    n_outmap = int(np.prod(n_outmaps))
    if w_init is None:
        fan_in = np.prod(inp.shape[base_axis:])
        w_init = UniformInitializer(
            calc_uniform_lim_glorot(fan_in, n_outmap), rng=rng)
    if i_init is None:
        fan_in = np.prod(inp.shape[base_axis:])
        i_init = ConstantInitializer()
    if b_init is None:
        b_init = ConstantInitializer()
    w = get_parameter_or_create(
        "W", [int(np.prod(inp.shape[base_axis:]))] + n_outmaps,
        w_init, not fix_parameters)
    i = get_parameter_or_create(
        "I", [int(np.prod(inp.shape[base_axis:]))] + n_outmaps,
        i_init, False)
    b = None
    if with_bias:
        b = get_parameter_or_create(
            "b", n_outmaps, b_init, not fix_parameters)
    return F.inq_affine(inp, w, i, b, base_axis, num_bits, inq_iterations, selection_algorithm, seed)


@parametric_function_api("conv")
def convolution(inp, outmaps, kernel,
                pad=None, stride=None, dilation=None, group=1,
                w_init=None, b_init=None,
                base_axis=1, fix_parameters=False, rng=None, with_bias=True):
    """
    N-D Convolution with a bias term.

    For Dilated Convolution (a.k.a. Atrous Convolusion), refer to:

    - Chen et al., DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs. https://arxiv.org/abs/1606.00915

    - Yu et al., Multi-Scale Context Aggregation by Dilated Convolutions. https://arxiv.org/abs/1511.07122

    Args:
        inp (~nnabla.Variable): N-D array.
        outmaps (int): Number of convolution kernels (which is equal to the number of output channels). For example, to apply convolution on an input with 16 types of filters, specify 16.
        kernel (:obj:`tuple` of :obj:`int`): Convolution kernel size. For example, to apply convolution on an image with a 3 (height) by 5 (width) two-dimensional kernel, specify (3,5).
        pad (:obj:`tuple` of :obj:`int`): Padding sizes for dimensions.
        stride (:obj:`tuple` of :obj:`int`): Stride sizes for dimensions.
        dilation (:obj:`tuple` of :obj:`int`): Dilation sizes for dimensions.
        group (int): Number of groups of channels. This makes connections across channels more sparse by grouping connections along map direction.
        w_init (~nnabla.initializer.BaseInitializer): Initializer for weight.
        b_init (~nnabla.initializer.BaseInitializer): Initializer for bias.
        base_axis (int): Dimensions up to `base_axis` are treated as the sample dimensions.
        fix_parameters (bool): When set to `True`, the weights and biases will not be updated.
        rng (numpy.random.RandomState): Random generator for Initializer.
        with_bias (bool): Specify whether to include the bias term.

    Returns:
        :class:`~nnabla.Variable`: N-D array.

    """
    if w_init is None:
        w_init = UniformInitializer(
            calc_uniform_lim_glorot(inp.shape[base_axis], outmaps, tuple(kernel)), rng=rng)
    if with_bias and b_init is None:
        b_init = ConstantInitializer()
    w = get_parameter_or_create(
        "W", (outmaps, inp.shape[base_axis] / group) + tuple(kernel),
        w_init, not fix_parameters)
    b = None
    if with_bias:
        b = get_parameter_or_create(
            "b", (outmaps,), b_init, not fix_parameters)
    return F.convolution(inp, w, b, base_axis, pad, stride, dilation, group)


@parametric_function_api("bicon_conv")
def binary_connect_convolution(inp, outmaps, kernel,
                               pad=None, stride=None, dilation=None, group=1,
                               w_init=None, wb_init=None, b_init=None,
                               base_axis=1, fix_parameters=False, rng=None,
                               with_bias=True):
    """Binary Connect Convolution, multiplier-less inner-product.

    Binary Connect Convolution is the convolution function, 
    except the definition of the inner product is modified.
    The input-output relation of this function is as follows:

    .. math::

        y_{n, a, b} = \sum_{m} \sum_{i} \sum_{j} sign(w_{n, m, i, j}) x_{m, a + i, b + j}.

    Therefore :math:`sign(w_i)` is either :math:`1` or :math:`-1` and the inner product
    simplifies to addition.

    This function should be used together with BatchNormalization.

    References:

        M. Courbariaux, Y. Bengio, and J.-P. David. "BinaryConnect:
        Training Deep Neural Networks with binary weights during propagations."
        Advances in Neural Information Processing Systems. 2015.

    .. note::

        1) if you would like to share weights between some layers, please
        make sure to share the standard, floating value weights (`weight`)
        and not the binarized weights (`binary_weight`)

        2) The weights and the binary weights become synced only after :func:`~nnabla._variable.Variable.forward` is called,
        and not after a call to :func:`~nnabla._variable.Variable.backward`.
        To access the parameters of the network, remember to call :func:`~nnabla._variable.Variable.forward` once before doing so, otherwise the
        float weights and the binary weights will not be in sync.

        3) Quantized values are stored as floating point number for `binary_weight`,
        since this function is only for simulation purposes.

    Args:
        inp (~nnabla.Variable): N-D array.
        outmaps (int): Number of convolution kernels (which is equal to the number of output channels). For example, to apply convolution on an input with 16 types of filters, specify 16.
        kernel (:obj:`tuple` of :obj:`int`): Convolution kernel size. For example, to apply convolution on an image with a 3 (height) by 5 (width) two-dimensional kernel, specify (3,5).
        pad (:obj:`tuple` of :obj:`int`): Padding sizes for dimensions.
        stride (:obj:`tuple` of :obj:`int`): Stride sizes for dimensions.
        dilation (:obj:`tuple` of :obj:`int`): Dilation sizes for dimensions.
        group (int): Number of groups of channels. This makes connections across channels sparser by grouping connections along map direction.
        w_init (~nnabla.initializer.BaseInitializer): Initializer for weight.
        wb_init (~nnabla.initializer.BaseInitializer): Initializer for binary weight.
        b_init (~nnabla.initializer.BaseInitializer): Initializer for bias.
        base_axis (int): Dimensions up to `base_axis` are treated as the sample dimensions.
        fix_parameters (bool): When set to `True`, the weights and biases will not be updated.
        rng (numpy.random.RandomState): Random generator for Initializer.
        with_bias (bool): Specify whether to include the bias term.

    Returns:
        :class:`~nnabla.Variable`

    """
    if w_init is None:
        w_init = UniformInitializer(
            calc_uniform_lim_glorot(inp.shape[base_axis], outmaps, tuple(kernel)), rng=rng)
    if wb_init is None:
        wb_init = UniformInitializer(
            calc_uniform_lim_glorot(inp.shape[base_axis], outmaps, tuple(kernel)), rng=rng)
    if b_init is None:
        b_init = ConstantInitializer()
    w = get_parameter_or_create(
        "W", (outmaps, inp.shape[base_axis]) + tuple(kernel),
        w_init, not fix_parameters)
    wb = get_parameter_or_create(
        "Wb", (outmaps, inp.shape[base_axis]) + tuple(kernel),
        wb_init, not fix_parameters)
    b = None
    if with_bias:
        b = get_parameter_or_create(
            "b", (outmaps,), b_init, not fix_parameters)
    return F.binary_connect_convolution(inp, w, wb, b, base_axis, pad, stride, dilation, group)


@parametric_function_api("bwn_conv")
def binary_weight_convolution(inp, outmaps, kernel,
                              pad=None, stride=None, dilation=None, group=1,
                              w_init=None, wb_init=None, b_init=None,
                              base_axis=1, fix_parameters=False, rng=None,
                              with_bias=True):
    """Binary Weight Convolution, multiplier-less inner-product with a scale factor.

    Binary Weight Convolution is the convolution function, but the
    inner product in this function is the following,

    .. math::

        y_{n, a, b} = \\frac{1}{\\|\\mathbf{w}_n\\|_{\\ell_1}} \sum_{m} \sum_{i} \sum_{j} sign(w_{n, m, i, j}) x_{m, a + i, b + j}.


    Therefore :math:`sign(w_{n, m, i, j})`  is either :math:`1` or :math:`-1` and the inner product
    simplifies to addition followed by scaling factor :math:`\\alpha = \\frac{1}{\\|\\mathbf{w}_n\\|_{\\ell_1}}`.
    The number of :math:`n` is the number of outmaps of the convolution
    function.

    References:

        Rastegari, Mohammad, et al. "XNOR-Net: ImageNet Classification Using
        Binary Convolutional Neural Networks." arXiv preprint
        arXiv:1603.05279 (2016).

    .. note::

        1) if you would like to share weights between some layers, please
        make sure to share the standard, floating value weights (`weight`)
        and not the binarized weights (`binary_weight`)

        2) The weights and the binary weights become synced only after :func:`~nnabla._variable.Variable.forward` is called,
        and not after a call to :func:`~nnabla._variable.Variable.backward`.
        To access the parameters of the network, remember to call :func:`~nnabla._variable.Variable.forward` once before doing so, otherwise the
        float weights and the binary weights will not be in sync.

        3) Quantized values are stored as floating point number for `binary_weight`,
        since this function is only for simulation purposes.

    Args:
        inp (~nnabla.Variable): N-D array.
        outmaps (int): Number of convolution kernels (which is equal to the number of output channels). For example, to apply convolution on an input with 16 types of filters, specify 16.
        kernel (:obj:`tuple` of :obj:`int`): Convolution kernel size. For example, to apply convolution on an image with a 3 (height) by 5 (width) two-dimensional kernel, specify (3,5).
        pad (:obj:`tuple` of :obj:`int`): Padding sizes for dimensions.
        stride (:obj:`tuple` of :obj:`int`): Stride sizes for dimensions.
        dilation (:obj:`tuple` of :obj:`int`): Dilation sizes for dimensions.
        group (int): Number of groups of channels. This makes connections across channels sparser by grouping connections along map direction.
        w_init (~nnabla.initializer.BaseInitializer): Initializer for weight.
        wb_init (~nnabla.initializer.BaseInitializer): Initializer for binary weight.
        b_init (~nnabla.initializer.BaseInitializer): Initializer for bias.
        base_axis (int): Dimensions up to `base_axis` are treated as the sample dimensions.
        fix_parameters (bool): When set to `True`, the weights and biases will not be updated.
        rng (numpy.random.RandomState): Random generator for Initializer.
        with_bias (bool): Specify whether to include the bias term.

    Returns:
        :class:`~nnabla.Variable`

    """
    if w_init is None:
        w_init = UniformInitializer(
            calc_uniform_lim_glorot(inp.shape[base_axis], outmaps, tuple(kernel)), rng=rng)
    if wb_init is None:
        wb_init = UniformInitializer(
            calc_uniform_lim_glorot(inp.shape[base_axis], outmaps, tuple(kernel)), rng=rng)
    if b_init is None:
        b_init = ConstantInitializer()
    w = get_parameter_or_create(
        "W", (outmaps, inp.shape[base_axis]) + tuple(kernel),
        w_init, not fix_parameters)
    wb = get_parameter_or_create(
        "Wb", (outmaps, inp.shape[base_axis]) + tuple(kernel),
        wb_init, not fix_parameters)
    alpha = get_parameter_or_create(
        "alpha", (outmaps, ), ConstantInitializer(0), False)
    b = None
    if with_bias:
        b = get_parameter_or_create(
            "b", (outmaps,), b_init, not fix_parameters)
    return F.binary_weight_convolution(inp, w, wb, alpha, b, base_axis, pad, stride, dilation, group)


@parametric_function_api("inq_conv")
def inq_convolution(inp, outmaps, kernel,
                    pad=None, stride=None, dilation=None, group=1,
                    num_bits=4, inq_iterations=(), selection_algorithm='random',
                    seed=-1, w_init=None, i_init=None, b_init=None,
                    base_axis=1, fix_parameters=False, rng=None,
                    with_bias=True):
    """Incremental Network Quantization Convolution Layer

    During training, the weights are sequentially quantized to power-of-two
    values, which allows the training of a multiplierless network.

    Using `inq_iterations`, one can specify after how many forward passes
    half of the learnable weights are fixed and quantized to powers-of-two.
    After reaching the last value in `inq_iterations`, all weights are fixed.

    For more details, please refer to the reference.

    Reference:
    Zhou A, Yao A, Guo Y, Xu L, Chen Y. Incremental network quantization:
    Towards lossless CNNs with low-precision weights.
    <https://arxiv.org/abs/1702.03044>

    Args:
        inp (~nnabla.Variable): Input N-D array with shape (:math:`M_0 \\times \ldots \\times M_{B-1} \\times D_B \\times \ldots \\times D_N`). Dimensions before and after base_axis are flattened as if it was a matrix.
        n_outmaps (int or :obj:`tuple` of :obj:`int`): Number of output neurons per data.
        base_axis (int): Dimensions up to `base_axis` are treated as the sample dimensions.
        num_bits (int): Number of bits per weight. Value has to be larger than 1 as one bit is already used to code the value "0"
        inq_iterations (tuple of int): Tuple of iteration numbers at which we fix half of the weights.
        selection_algorithm (str): Chooses algorithm that is used to decide which weights are fixed. ("largest_abs" ... fix weights with largest absolute value, "random" ... fix weights randomly)
        seed (int): Random seed for INQ algorithm
        w_init (~nnabla.initializer.BaseInitializer): Initializer for the weight.
        i_init (~nnabla.initializer.BaseInitializer): Initializer for the indicators (0 ... learnable, 1 ... fixed).
        b_init (~nnabla.initializer.BaseInitializer): Initializer for the bias.
        fix_parameters (bool): When set to `True`, the weight and bias will not be updated.
        rng (numpy.random.RandomState): Random generator for Initializer.
        with_bias (bool): Specify whether to include the bias term.

    Returns:
        :class:`~nnabla.Variable`

    """
    if w_init is None:
        w_init = UniformInitializer(
            calc_uniform_lim_glorot(inp.shape[base_axis], outmaps, tuple(kernel)), rng=rng)
    if i_init is None:
        i_init = ConstantInitializer()
    if b_init is None:
        b_init = ConstantInitializer()
    w = get_parameter_or_create(
        "W", (outmaps, inp.shape[base_axis]) + tuple(kernel),
        w_init, not fix_parameters)
    i = get_parameter_or_create(
        "I", (outmaps, inp.shape[base_axis]) + tuple(kernel),
        i_init, False)
    b = None
    if with_bias:
        b = get_parameter_or_create(
            "b", (outmaps,), b_init, not fix_parameters)
    return F.inq_convolution(inp, w, i, b, base_axis, pad, stride, dilation, group, num_bits, inq_iterations, selection_algorithm, seed)


@parametric_function_api("depthwise_conv")
def depthwise_convolution(inp, kernel,
                          pad=None, stride=None, dilation=None, multiplier=1,
                          w_init=None, b_init=None,
                          base_axis=1, fix_parameters=False, rng=None, with_bias=True):
    """
    N-D Deptwise Convolution with a bias term.

    Reference:

    - F. Chollet: Chollet, Francois. "Xception: Deep Learning with Depthwise Separable Convolutions. https://arxiv.org/abs/1610.02357

    Args:
        inp (~nnabla.Variable): N-D array.
        kernel (:obj:`tuple` of :obj:`int`): Convolution kernel size. For example, to apply convolution on an image with a 3 (height) by 5 (width) two-dimensional kernel, specify (3,5).
        pad (:obj:`tuple` of :obj:`int`): Padding sizes for dimensions.
        stride (:obj:`tuple` of :obj:`int`): Stride sizes for dimensions.
        dilation (:obj:`tuple` of :obj:`int`): Dilation sizes for dimensions.
        multiplier (:obj:`int`): Number of output feature maps per input feature map.
        w_init (~nnabla.initializer.BaseInitializer): Initializer for weight.
        b_init (~nnabla.initializer.BaseInitializer): Initializer for bias.
        base_axis (int): Dimensions up to `base_axis` are treated as the sample dimensions.
        fix_parameters (bool): When set to `True`, the weights and biases will not be updated.
        rng (numpy.random.RandomState): Random generator for Initializer.
        with_bias (bool): Specify whether to include the bias term.

    Returns:
        :class:`~nnabla.Variable`: N-D array.

    """
    if w_init is None:
        w_init = UniformInitializer(
            calc_uniform_lim_glorot(inp.shape[base_axis], inp.shape[base_axis], tuple(kernel)), rng=rng)
    if with_bias and b_init is None:
        b_init = ConstantInitializer()
    w = get_parameter_or_create(
        "W", (inp.shape[base_axis],) + tuple(kernel),
        w_init, not fix_parameters)
    b = None
    if with_bias:
        b = get_parameter_or_create(
            "b", (inp.shape[base_axis],), b_init, not fix_parameters)
    return F.depthwise_convolution(inp, w, b, base_axis, pad, stride, dilation,
                                   multiplier)


@parametric_function_api("deconv")
def deconvolution(inp, outmaps, kernel,
                  pad=None, stride=None, dilation=None, group=1,
                  w_init=None, b_init=None,
                  base_axis=1, fix_parameters=False, rng=None, with_bias=True):
    """
    Deconvolution layer.

    Args:
        inp (~nnabla.Variable): N-D array.
        outmaps (int): Number of deconvolution kernels (which is equal to the number of output channels). For example, to apply deconvolution on an input with 16 types of filters, specify 16.
        kernel (:obj:`tuple` of :obj:`int`): Convolution kernel size. For example, to apply deconvolution on an image with a 3 (height) by 5 (width) two-dimensional kernel, specify (3,5).
        pad (:obj:`tuple` of :obj:`int`): Padding sizes for dimensions.
        stride (:obj:`tuple` of :obj:`int`): Stride sizes for dimensions.
        dilation (:obj:`tuple` of :obj:`int`): Dilation sizes for dimensions.
        group (int): Number of groups of channels. This makes connections across channels sparser by grouping connections along map direction.
        w_init (~nnabla.initializer.BaseInitializer): Initializer for weight.
        b_init (~nnabla.initializer.BaseInitializer): Initializer for bias.
        base_axis (int): Dimensions up to `base_axis` are treated as the sample dimensions.
        fix_parameters (bool): When set to `True`, the weights and biases will not be updated.
        rng (numpy.random.RandomState): Random generator for Initializer.
        with_bias (bool): Specify whether to include the bias term.

    Returns:
        :class:`~nnabla.Variable`: N-D array.

    """
    if w_init is None:
        w_init = UniformInitializer(
            calc_uniform_lim_glorot(outmaps, inp.shape[base_axis], tuple(kernel)), rng=rng)
    if with_bias and b_init is None:
        b_init = ConstantInitializer()
    w = get_parameter_or_create(
        "W", (inp.shape[base_axis], outmaps / group) + tuple(kernel),
        w_init, not fix_parameters)
    b = None
    if with_bias:
        b = get_parameter_or_create(
            "b", (outmaps,), b_init, not fix_parameters)
    return F.deconvolution(inp, w, b, base_axis, pad, stride, dilation, group)


@parametric_function_api("bn")
def batch_normalization(inp, axes=[1], decay_rate=0.9, eps=1e-5,
                        batch_stat=True, output_stat=False, fix_parameters=False):
    """
    Batch normalization layer.

    .. math::
        \\begin{array}{lcl}
        \\mu &=& \\frac{1}{M} \\sum x_i\\\\
        \\sigma^2 &=& \\frac{1}{M} \\left(\\sum x_i - \\mu\\right)^2\\\\
        \\hat{
          x
        } _i &= & \\frac{x_i - \\mu} {
          \\sqrt {\\sigma ^ 2 + \\epsilon }
        }
        \\\ y_i &= & \\hat { x }
        _i \\gamma + \\beta.
        \\end { array }

    where :math:`x_i, y_i` are the inputs.
    In testing, the mean and variance computed by moving average calculated during training are used.

    Args:
        inp (~nnabla.Variable): N-D array of input.
        axes (:obj:`tuple` of :obj:`int`): Axes mean and variance are taken.
        decay_rate (float): Decay rate of running mean and variance.
        eps (float): Tiny value to avoid zero division by std.
        batch_stat (bool): Use mini-batch statistics rather than running ones.
        output_stat (bool): Output batch mean and variance.
        fix_parameters (bool): When set to `True`, the beta and gamma will not be updated.

    Returns:
        :class:`~nnabla.Variable`: N-D array.

    References:

        - Ioffe and Szegedy, Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. https://arxiv.org/abs/1502.03167

    """
    assert len(axes) == 1
    shape_stat = [1 for _ in inp.shape]
    shape_stat[axes[0]] = inp.shape[axes[0]]
    beta = get_parameter_or_create(
        "beta", shape_stat, ConstantInitializer(0), not fix_parameters)
    gamma = get_parameter_or_create(
        "gamma", shape_stat, ConstantInitializer(1), not fix_parameters)
    mean = get_parameter_or_create(
        "mean", shape_stat, ConstantInitializer(0), False)
    var = get_parameter_or_create(
        "var", shape_stat, ConstantInitializer(0), False)
    return F.batch_normalization(inp, beta, gamma, mean, var, axes,
                                 decay_rate, eps, batch_stat, output_stat)


@parametric_function_api("embed")
def embed(inp, n_inputs, n_features, fix_parameters=False):
    """ Embed.

    Embed slices a matrix/tensor with indexing array/tensor

    Args:
        x(~nnabla.Variable): [Integer] Indices with shape :math:`(I_0, ..., I_N)`
        n_inputs : number of possible inputs, words or vocabraries
        n_features : number of embedding features
        fix_parameters (bool): When set to `True`, the embedding weight matrix
            will not be updated.

    Returns:
        ~nnabla.Variable: Output with shape :math:`(I_0, ..., I_N, W_1, ..., W_M)`
    """
    w = get_parameter_or_create("W", [n_inputs, n_features],
                                UniformInitializer((-np.sqrt(3.), np.sqrt(3))), not fix_parameters)
    return F.embed(inp, w)


@parametric_function_api("prelu")
def prelu(inp, base_axis=1, shared=True, fix_parameters=False):
    """
    Parametrized Rectified Linear Unit function defined as

    .. math::
        y_i = \max(0, x_i) + w_i \min(0, -x_i)

    where nagative slope :math:`w` is learned and can vary accros channels (an
    axis specified with base_axis).

    Args:
        x(~nnabla.Variable): N-D array as input
        base_axis(int): Dimensions up to base_axis is treated as sample dimension.
        shared(bool): Use shared weight value or not 
        fix_parameters (bool): When set to `True`, the negative slope values
            will not be updated.

    Returns:
        ~nnabla.Variable: N-D array.

    """
    shape = tuple() if shared else inp.shape[base_axis]
    w = get_parameter_or_create("slope", shape,
                                ConstantInitializer(-1), not fix_parameters)
    return F.prelu(inp, w, base_axis)


@parametric_function_api("fp_quantized_affine")
def fixed_point_quantized_affine(inp, n_outmaps,
                                 base_axis=1,
                                 w_init=None, b_init=None,
                                 fix_parameters=False, rng=None, with_bias=True,
                                 quantize_w=True, sign_w=True, n_w=8, delta_w=2**-4, ste_fine_grained_w=True,
                                 quantize_b=True, sign_b=True, n_b=8, delta_b=2**-4, ste_fine_grained_b=True):
    """Fixed-Point Quantized Affine.

    Fixed-Point Quantized Affine is the affine function, 
    except the definition of the inner product is modified.
    The input-output relation of this function is as follows:

    .. math::

        y_j = \sum_{i} Q(w_{ji}) x_i, 

    where :math:`Q(w_{ji})` is the fixed-point quantization function.

    .. note::

        1) if you would like to share weights between some layers, please
        make sure to share the standard, floating value weights (`weight`)
        and not the quantized weights (`quantized weight`)

        2) The weights and the quantized weights become synced only after :func:`~nnabla._variable.Variable.forward` is called,
        and not after a call to :func:`~nnabla._variable.Variable.backward`.
        To access the parameters of the network, remember to call :func:`~nnabla._variable.Variable.forward` once before doing so, otherwise the
        float weights and the quantized weights will not be in sync.

        3) CPU and GPU implementations now use float value for `quantized weight`,
        since this function is only for simulation purposes.

    Args:
        inp (~nnabla.Variable): Input N-D array with shape (:math:`M_0 \\times \ldots \\times M_{B-1} \\times D_B \\times \ldots \\times D_N`). Dimensions before and after base_axis are flattened as if it is a matrix.
        n_outmaps (:obj:`int` or :obj:`tuple` of :obj:`int`): Number of output neurons per data.
        base_axis (int): Dimensions up to `base_axis` are treated as the sample dimensions.
        w_init (~nnabla.initializer.BaseInitializer): Initializer for weight.
        b_init (~nnabla.initializer.BaseInitializer): Initializer for bias.
        fix_parameters (bool): When set to `True`, the weights and biases will not be updated.
        rng (numpy.random.RandomState): Random generator for Initializer.
        with_bias (bool): Specify whether to include the bias term.
        quantize_w (bool): Quantize weights if `True`.
        sign_w (bool): Use signed quantization if `True`.
        n_w (int): Bit witdh used for weight.
        delta_w (float): Step size for weight.
        ste_fine_grained_w (bool): STE is fine-grained if `True`.
        quantize_b (bool): Quantize bias if `True`.
        n_b (int): Bit witdh used for bias.
        delta_w (float): Step size for bias.
        ste_fine_grained_b (bool): STE is fine-grained if `True`.

    Returns:
        :class:`~nnabla.Variable`: :math:`(B + 1)`-D array. (:math:`M_0 \\times \ldots \\times M_{B-1} \\times L`)

    """

    if not hasattr(n_outmaps, '__iter__'):
        n_outmaps = [n_outmaps]
    n_outmaps = list(n_outmaps)
    n_outmap = int(np.prod(n_outmaps))
    if w_init is None:
        inmaps = np.prod(inp.shape[base_axis:])
        w_init = UniformInitializer(
            calc_uniform_lim_glorot(inmaps, n_outmap), rng=rng)
    if with_bias and b_init is None:
        b_init = ConstantInitializer()

    # Floating Weight
    w = get_parameter_or_create(
        "W", [int(np.prod(inp.shape[base_axis:]))] + n_outmaps,
        w_init, not fix_parameters)

    # Quantized Weight
    if quantize_w:
        w_q = get_parameter_or_create(
            "W_q", [int(np.prod(inp.shape[base_axis:]))] + n_outmaps,
            w_init, not fix_parameters)
        # Link computation graph
        real_w_q = F.fixed_point_quantize(w, quantize=quantize_w,
                                          sign=sign_w, n=n_w, delta=delta_w,
                                          ste_fine_grained=ste_fine_grained_w,
                                          outputs=[w_q.data])
        real_w_q.persistent = True
    else:
        real_w_q = w

    # Bias
    # Floating
    b = None
    b_q = None
    real_b_q = None
    if with_bias:
        b = get_parameter_or_create(
            "b", n_outmaps, b_init, not fix_parameters)
        if quantize_b:
            b_q = get_parameter_or_create(
                "b_q", n_outmaps, b_init, not fix_parameters)
            # Link computation graph
            real_b_q = F.fixed_point_quantize(b, quantize=quantize_b,
                                              sign=sign_b, n=n_b, delta=delta_b,
                                              ste_fine_grained=ste_fine_grained_b,
                                              outputs=[b_q.data])
            real_b_q.persistent = True
        else:
            real_b_q = b

    return F.affine(inp, real_w_q, real_b_q, base_axis)


@parametric_function_api("fp_quantized_conv")
def fixed_point_quantized_convolution(inp, outmaps, kernel,
                                      pad=None, stride=None, dilation=None, group=1,
                                      w_init=None, b_init=None,
                                      base_axis=1, fix_parameters=False, rng=None, with_bias=True,
                                      quantize_w=True, sign_w=True, n_w=8, delta_w=2**-4, ste_fine_grained_w=True,
                                      quantize_b=True, sign_b=True, n_b=8, delta_b=2**-4, ste_fine_grained_b=True,):
    """Fixed-Point Quantized Convolution.

    Fixed-Point Quantized Convolution is the convolution function, 
    except the definition of the inner product is modified.
    The input-output relation of this function is as follows:

    .. math::

        y_{n, a, b} = \sum_{m} \sum_{i} \sum_{j} Q(w_{n, m, i, j}) x_{m, a + i, b + j}, 

    where :math:`Q(w_{n, m, i, j})` is the fixed-point quantization function.

    .. note::

        1) if you would like to share weights between some layers, please
        make sure to share the standard, floating value weights (`weight`)
        and not the quantized weights (`quantized weight`)

        2) The weights and the quantized weights become synced only after :func:`~nnabla._variable.Variable.forward` is called,
        and not after a call to :func:`~nnabla._variable.Variable.backward`.
        To access the parameters of the network, remember to call :func:`~nnabla._variable.Variable.forward` once before doing so, otherwise the
        float weights and the quantized weights will not be in sync.

        3) CPU and GPU implementations now use float value for `quantized weight`,
        since this function is only for simulation purposes.

    Args:
        inp (~nnabla.Variable): N-D array.
        outmaps (int): Number of convolution kernels (which is equal to the number of output channels). For example, to apply convolution on an input with 16 types of filters, specify 16.
        kernel (:obj:`tuple` of :obj:`int`): Convolution kernel size. For example, to apply convolution on an image with a 3 (height) by 5 (width) two-dimensional kernel, specify (3,5).
        pad (:obj:`tuple` of :obj:`int`): Padding sizes for dimensions.
        stride (:obj:`tuple` of :obj:`int`): Stride sizes for dimensions.
        dilation (:obj:`tuple` of :obj:`int`): Dilation sizes for dimensions.
        group (int): Number of groups of channels. This makes connections across channels more sparse by grouping connections along map direction.
        w_init (~nnabla.initializer.BaseInitializer): Initializer for weight.
        b_init (~nnabla.initializer.BaseInitializer): Initializer for bias.
        base_axis (int): Dimensions up to `base_axis` are treated as the sample dimensions.
        fix_parameters (bool): When set to `True`, the weights and biases will not be updated.
        rng (numpy.random.RandomState): Random generator for Initializer.
        with_bias (bool): Specify whether to include the bias term.
        quantize_w (bool): Quantize weights if `True`.
        quantize_bias (bool): Quantize bias if `True`.
        sign_w (bool): Use signed quantization if `True`.
        n_w (int): Bit witdh used for weight.
        delta_w (float): Step size for weight.
        ste_fine_grained_w (bool): STE is fine-grained if `True`.
        quantize_b (bool): Quantize bias if `True`.
        n_b (int): Bit witdh used for bias.
        delta_w (float): Step size for bias.
        ste_fine_grained_b (bool): STE is fine-grained if `True`.

    Returns:
        :class:`~nnabla.Variable`: N-D array.

    """
    if w_init is None:
        w_init = UniformInitializer(
            calc_uniform_lim_glorot(inp.shape[base_axis], outmaps, tuple(kernel)), rng=rng)
    if with_bias and b_init is None:
        b_init = ConstantInitializer()

    # Floating Weight
    w = get_parameter_or_create(
        "W", (outmaps, inp.shape[base_axis] / group) + tuple(kernel),
        w_init, not fix_parameters)

    # Quantized Weight
    if quantize_w:
        w_q = get_parameter_or_create(
            "W_q", (outmaps, inp.shape[base_axis] / group) + tuple(kernel),
            w_init, not fix_parameters)
        # Link computation graph
        real_w_q = F.fixed_point_quantize(w, quantize=quantize_w,
                                          sign=sign_w, n=n_w, delta=delta_w,
                                          ste_fine_grained=ste_fine_grained_w,
                                          outputs=[w_q.data])
        real_w_q.persistent = True
    else:
        real_w_q = w

    # Bias
    # Floating
    b = None
    b_q = None
    real_b_q = None

    if with_bias:
        b = get_parameter_or_create(
            "b", (outmaps,), b_init, not fix_parameters)
        if quantize_b:
            b_q = get_parameter_or_create(
                "b_q", (outmaps,), b_init, not fix_parameters)
            # Link computation graph
            real_b_q = F.fixed_point_quantize(b, quantize=quantize_b,
                                              sign=sign_b, n=n_b, delta=delta_b,
                                              ste_fine_grained=ste_fine_grained_b,
                                              outputs=[b_q.data])
            real_b_q.persistent = True
        else:
            real_b_q = b

    return F.convolution(inp, real_w_q, real_b_q, base_axis, pad, stride, dilation, group)


@parametric_function_api("pow2_quantized_affine")
def pow2_quantized_affine(inp, n_outmaps,
                          base_axis=1,
                          w_init=None, b_init=None,
                          fix_parameters=False, rng=None, with_bias=True,
                          quantize_w=True, sign_w=True, with_zero_w=False, n_w=8, m_w=2, ste_fine_grained_w=True,
                          quantize_b=True, sign_b=True, with_zero_b=False, n_b=8, m_b=2, ste_fine_grained_b=True):
    """Pow2 Quantized Affine.

    Pow2 Quantized Affine is the affine function, 
    except the definition of the inner product is modified.
    The input-output relation of this function is as follows:

    .. math::

        y_j = \sum_{i} Q(w_{ji}) x_i, 

    where :math:`Q(w_{ji})` is the power-of-2 quantization function.

    .. note::

        1) if you would like to share weights between some layers, please
        make sure to share the standard, floating value weights (`weight`)
        and not the quantized weights (`quantized weight`)

        2) The weights and the quantized weights become synced only after :func:`~nnabla._variable.Variable.forward` is called,
        and not after a call to :func:`~nnabla._variable.Variable.backward`.
        To access the parameters of the network, remember to call :func:`~nnabla._variable.Variable.forward` once before doing so, otherwise the
        float weights and the quantized weights will not be in sync.

        3) Quantized values are stored as floating point number for `quantized weight`,
        since this function is only for simulation purposes.

    Args:
        inp (~nnabla.Variable): Input N-D array with shape (:math:`M_0 \\times \ldots \\times M_{B-1} \\times D_B \\times \ldots \\times D_N`). Dimensions before and after base_axis are flattened as if it is a matrix.
        n_outmaps (:obj:`int` or :obj:`tuple` of :obj:`int`): Number of output neurons per data.
        base_axis (int): Dimensions up to `base_axis` are treated as the sample dimensions.
        w_init (~nnabla.initializer.BaseInitializer): Initializer for weight.
        b_init (~nnabla.initializer.BaseInitializer): Initializer for bias.
        fix_parameters (bool): When set to `True`, the weights and biases will not be updated.
        rng (numpy.random.RandomState): Random generator for Initializer.
        with_bias (bool): Specify whether to include the bias term.
        quantize_w (bool): Quantize weights if `True`.
        sign_w (bool): Use signed quantization if `True`.
        with_zero_w (bool): Indicate using zero as a quantized value. Default is false.
        n_w (int): Bit witdh used for weight.
        m_w (int): :math:`2^m` is upper bound and :math:`-2^m` is lower bound for weights. Default is 2.
        ste_fine_grained_w (bool): STE is fine-grained if `True`.
        quantize_b (bool): Quantize bias if `True`.
        with_zero_b (bool): Indicate using zero as a quantized value. Default is false.
        n_b (int): Bit witdh used for bias.
        m_b (int): :math:`2^m` is upper bound and :math:`-2^m` is lower bound for bias. Default is 2.
        ste_fine_grained_b (bool): STE is fine-grained if `True`.  
    Returns:
        :class:`~nnabla.Variable`: :math:`(B + 1)`-D array. (:math:`M_0 \\times \ldots \\times M_{B-1} \\times L`)

    """

    if not hasattr(n_outmaps, '__iter__'):
        n_outmaps = [n_outmaps]
    n_outmaps = list(n_outmaps)
    n_outmap = int(np.prod(n_outmaps))
    if w_init is None:
        inmaps = np.prod(inp.shape[base_axis:])
        w_init = UniformInitializer(
            calc_uniform_lim_glorot(inmaps, n_outmap), rng=rng)
    if with_bias and b_init is None:
        b_init = ConstantInitializer()

    # Floating Weight
    w = get_parameter_or_create(
        "W", [int(np.prod(inp.shape[base_axis:]))] + n_outmaps,
        w_init, not fix_parameters)

    # Quantized Weight
    if quantize_w:
        w_q = get_parameter_or_create(
            "W_q", [int(np.prod(inp.shape[base_axis:]))] + n_outmaps,
            w_init, not fix_parameters)
        # Link computation graph
        real_w_q = F.pow2_quantize(w, quantize=quantize_w,
                                   sign=sign_w, with_zero=with_zero_w,
                                   n=n_w, m=m_w, ste_fine_grained=ste_fine_grained_w,
                                   outputs=[w_q.data])
        real_w_q.persistent = True
    else:
        real_w_q = w

    # Bias
    # Floating
    b = None
    b_q = None
    real_b_q = None
    if with_bias:
        b = get_parameter_or_create(
            "b", n_outmaps, b_init, not fix_parameters)
        if quantize_b:
            b_q = get_parameter_or_create(
                "b_q", n_outmaps, b_init, not fix_parameters)
            real_b_q = F.pow2_quantize(b, quantize=quantize_b,
                                       sign=sign_b, with_zero=with_zero_b,
                                       n=n_b, m=m_b, ste_fine_grained=ste_fine_grained_b,
                                       outputs=[b_q.data])
            real_b_q.persistent = True
        else:
            real_b_q = b

    return F.affine(inp, real_w_q, real_b_q, base_axis)


@parametric_function_api("pow2_quantized_conv")
def pow2_quantized_convolution(inp, outmaps, kernel,
                               pad=None, stride=None, dilation=None, group=1,
                               w_init=None, b_init=None,
                               base_axis=1, fix_parameters=False, rng=None, with_bias=True,
                               quantize_w=True, with_zero_w=False, sign_w=True, n_w=8, m_w=2, ste_fine_grained_w=True,
                               quantize_b=True, with_zero_b=False, sign_b=True, n_b=8, m_b=2, ste_fine_grained_b=True,):
    """Pow2 Quantized Convolution.

    Pow2 Quantized Convolution is the convolution function, 
    except the definition of the inner product is modified.
    The input-output relation of this function is as follows:

    .. math::

        y_{n, a, b} = \sum_{m} \sum_{i} \sum_{j} Q(w_{n, m, i, j}) x_{m, a + i, b + j}, 

    where :math:`Q(w_{n, m, i, j})` is the power-of-2 quantization function.

    .. note::

        1) if you would like to share weights between some layers, please
        make sure to share the standard, floating value weights (`weight`)
        and not the quantized weights (`quantized weight`)

        2) The weights and the quantized weights become synced only after :func:`~nnabla._variable.Variable.forward` is called,
        and not after a call to :func:`~nnabla._variable.Variable.backward`.
        To access the parameters of the network, remember to call :func:`~nnabla._variable.Variable.forward` once before doing so, otherwise the
        float weights and the quantized weights will not be in sync.

        3) Quantized values are stored as floating point number for `quantized weight`,
        since this function is only for simulation purposes.

    Args:
        inp (~nnabla.Variable): N-D array.
        outmaps (int): Number of convolution kernels (which is equal to the number of output channels). For example, to apply convolution on an input with 16 types of filters, specify 16.
        kernel (:obj:`tuple` of :obj:`int`): Convolution kernel size. For example, to apply convolution on an image with a 3 (height) by 5 (width) two-dimensional kernel, specify (3,5).
        pad (:obj:`tuple` of :obj:`int`): Padding sizes for dimensions.
        stride (:obj:`tuple` of :obj:`int`): Stride sizes for dimensions.
        dilation (:obj:`tuple` of :obj:`int`): Dilation sizes for dimensions.
        group (int): Number of groups of channels. This makes connections across channels more sparse by grouping connections along map direction.
        w_init (~nnabla.initializer.BaseInitializer): Initializer for weight.
        b_init (~nnabla.initializer.BaseInitializer): Initializer for bias.
        base_axis (int): Dimensions up to `base_axis` are treated as the sample dimensions.
        fix_parameters (bool): When set to `True`, the weights and biases will not be updated.
        rng (numpy.random.RandomState): Random generator for Initializer.
        with_bias (bool): Specify whether to include the bias term.
        quantize_w (bool): Quantize weights if `True`.
        sign_w (bool): Use signed quantization if `True`.
        n_w (int): Bit witdh used for weight.
        m_w (int): :math:`2^m` is upper bound and :math:`-2^m` is lower bound for weights. Default is 2.
        ste_fine_grained_w (bool): STE is fine-grained if `True`.
        quantize_b (bool): Quantize bias if `True`.
        sign_b (bool): Use signed quantization if `True`.
        n_b (int): Bit witdh used for bias.
        m_b (int): :math:`2^m` is upper bound and :math:`-2^m` is lower bound for bias. Default is 2.
        ste_fine_grained_b (bool): STE is fine-grained if `True`.

    Returns:
        :class:`~nnabla.Variable`: N-D array.

    """
    if w_init is None:
        w_init = UniformInitializer(
            calc_uniform_lim_glorot(inp.shape[base_axis], outmaps, tuple(kernel)), rng=rng)
    if with_bias and b_init is None:
        b_init = ConstantInitializer()

    # Floating Weight
    w = get_parameter_or_create(
        "W", (outmaps, inp.shape[base_axis] / group) + tuple(kernel),
        w_init, not fix_parameters)

    # Quantized Weight
    if quantize_w:
        w_q = get_parameter_or_create(
            "W_q", (outmaps, inp.shape[base_axis] / group) + tuple(kernel),
            w_init, not fix_parameters)

        # Link computation graph
        real_w_q = F.pow2_quantize(w, quantize=quantize_w,
                                   sign=sign_w, with_zero=with_zero_w,
                                   n=n_w, m=m_w, ste_fine_grained=ste_fine_grained_w,
                                   outputs=[w_q.data])
        real_w_q.persistent = True
    else:
        real_w_q = w

    # Bias
    # Floating
    b = None
    b_q = None
    real_b_q = None

    if with_bias:
        b = get_parameter_or_create(
            "b", (outmaps,), b_init, not fix_parameters)
        if quantize_b:
            b_q = get_parameter_or_create(
                "b_q", (outmaps,), b_init, not fix_parameters)
            # Link computation graph
            real_b_q = F.pow2_quantize(b, quantize=quantize_b,
                                       sign=sign_b, with_zero=with_zero_b,
                                       n=n_b, m=m_b, ste_fine_grained=ste_fine_grained_b,
                                       outputs=[b_q.data])
            real_b_q.persistent = True
        else:
            real_b_q = b

    return F.convolution(inp, real_w_q, real_b_q, base_axis, pad, stride, dilation, group)
