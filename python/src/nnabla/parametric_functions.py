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


def parametric_function_api(scope_name=None, param_desc=None):
    """Decorator for parametric functions.

    The decorated function is always called under
    a parameter scope ``scope_name``.
    Also, the decorator adds an additional argument ``name`` (:obj:`str`,
    default is ``None``) at the end. If ``name`` is specified, the
    scope ``scope_name`` comes under a scope ``name``. This feature
    could reduce vertical space usage of the source code.
    Any parametric function should be decorated by this.

    Args:
        scope_name (str, optional): The original function will be called
            under a parameter scope named by ``scope_name``.
        param_desc (list, optional):
            Descriptions of parameters will be automatically included into docstring.
            This must be a list of tuples with 4 elements composed of
            (name (str), description (str), shape info (str), need_grad (bool)).

    Returns:
        function: A decorated parametric function.

    """
    if scope_name is None:
        scope_name = name

    def parametric_function_api_inside(func):
        from nnabla.utils.py23_compatible import getargspec
        import inspect

        name = func.__name__
        doc = func.__doc__

        if param_desc:
            indent = 8
            try:
                desc = map(lambda d: ' ' * indent +
                           '* {} (``need_grad={}``) : {}. (shape: ``{}``)'.format(d[0], d[3], d[1], d[2]), param_desc)
            except:
                ValueError(
                    'param_desc argument of parametric_function_api must be '
                    'None or a list of tuple with three elements composed of '
                    '(name(str), description(str), need_grad(bool)).')
            doc += '''
    Parameters to be registered
        The following variables are registered in a parameter scope ``"{}"``;

{}

            '''.format(scope_name, '\n'.join(desc))

        doc += """
    Note:

        If the ``name`` option is passed, the parameters become wrapped inside the parameter scope
        with the specified name, yielding the same results as the following code.
        This can be used to simplify the code.

        .. code-block:: python

            with parametric_scope(name):
                output = {name}(<args>)

        """.format(name=name)

        spec = getargspec(func)
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


@parametric_function_api("affine", [
    ('W', 'Weight matrix', '(inmaps, outmaps)', True),
    ('b', 'bias vector', '(outputs,)', True),
])
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
        w_init (:obj:`nnabla.initializer.BaseInitializer` or :obj:`numpy.ndarray`): Initializer for weight. By default, it is initialized with :obj:`nnabla.initializer.UniformInitializer` within the range determined by :obj:`nnabla.initializer.calc_uniform_lim_glorot`.
        b_init (:obj:`nnabla.initializer.BaseInitializer` or :obj:`numpy.ndarray`): Initializer for bias. By default, it is initialized with zeros if `with_bias` is `True`.
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
        w_init, True, not fix_parameters)
    b = None
    if with_bias:
        b = get_parameter_or_create(
            "b", n_outmaps, b_init, True, not fix_parameters)
    return F.affine(inp, w, b, base_axis)


@parametric_function_api("svd_affine", [
    ('U', ':math:`{\\mathbf U}`', '(inmaps, r)', True),
    ('V', ':math:`{\\mathbf V}`', '(r, outmaps)', True),
    ('b', 'Bias vector', '(outmaps,)', True),
])
def svd_affine(inp, n_outmaps, r, base_axis=1, uv_init=None,
               b_init=None, fix_parameters=False, rng=None,
               with_bias=True):
    """SVD affine is a low rank approximation of the affine layer. It can
    be seen as two consecutive affine layers with a bottleneck. It
    computes:

    .. math::
        {\\mathbf y} = {\\mathbf U} {\\mathbf V} {\\mathbf x} + {\\mathbf b}.

    where :math:`{\\mathbf x}, {\\mathbf y}` are the inputs and
    outputs respectively, and :math:`{\\mathbf U}, {\\mathbf V},
    {\\mathbf b}` are constants.

    The weights :math:`{\\mathbf U}` and :math:`{\\mathbf V}` are
    approximated with singular value decomposition (SVD) of the
    original weight matrix :math:`{\\mathbf W}` and by selecting the
    :math:`{R}` dominant singular values and the corresponding
    singular vectors. Therefore the low rank :math:`{R}` is the size
    of the bottleneck.

    If `uv_init` is a numpy array, :math:`{\\mathbf U}` and
    :math:`{\\mathbf V}` are computed such that `uv_init` is
    approximated by :math:`{\\mathbf{UV}}`. If `uv_init` is `None` or
    an initializer, the product of :math:`{\\mathbf U}` and
    :math:`{\\mathbf V}` approximates the random initialization.

    If :math:`{\\mathbf U}` and :math:`{\\mathbf V}` exist in the context,
    they take precedence over `uv_init`.

    Suppose the weight of the affine is of :math:`{I \\times O}` and
    the compression rate you want to specify is :math:`{CR}`, then you
    set :math:`{R}` as

    .. math::

        R = \\left\\lfloor \\frac{(1 - CR)OI}{O + I} \\right\\rfloor.

    Args:

        inp (~nnabla.Variable): Input N-D array with shape (:math:`M_0
          \\times \ldots \\times M_{B-1} \\times D_B \\times \ldots
          \\times D_N`). Dimensions before and after base_axis are
          flattened as if it is a matrix.

        n_outmaps (int or tuple): Number of output neurons per data.

        r (int): rank of the factorized layer (size of the bottleneck)

        base_axis (int): Dimensions up to `base_axis` are treated as
          the sample dimensions.

        uv_init (:obj:`nnabla.initializer.BaseInitializer` or :obj:`numpy.ndarray`):
          Initializer for weight. By default, it is initialized with :obj:`nnabla.initializer.UniformInitializer` within the range determined by :obj:`nnabla.initializer.calc_uniform_lim_glorot`. 

        b_init (:obj:`nnabla.initializer.BaseInitializer` or :obj:`numpy.ndarray`): Initializer for bias. By default, it is initialized with zeros if `with_bias` is `True`.
        fix_parameters (bool): When set to `True`, the weights
          and biases will not be updated.

        rng (numpy.random.RandomState): Random generator for Initializer.

        with_bias (bool): Specify whether to include the bias term.

    Returns:
        ~nnabla.Variable: :math:`(B + 1)`-D array.
        (:math:`M_0 \\times \ldots \\times M_{B-1} \\times L`)

    """
    if not hasattr(n_outmaps, '__iter__'):
        n_outmaps = [n_outmaps]

    n_outmaps = list(n_outmaps)
    n_outmap = int(np.prod(n_outmaps))
    inmaps = np.prod(inp.shape[base_axis:])

    if uv_init is None:
        uv_init = UniformInitializer(
            calc_uniform_lim_glorot(inmaps, n_outmap), rng=rng)

    if type(uv_init) is np.ndarray:
        # TODO: Assert that size of uv_init is correct
        # uv is initialize with numpy array
        uv = uv_init
    else:
        # uv is initialize from initializer
        uv = uv_init([int(np.prod(inp.shape[base_axis:])), ] +
                     list(n_outmaps))

    u = get_parameter('U')
    v = get_parameter('V')

    if (u is None) or (v is None):
        assert r > 0, "svd_ffine: The rank must larger than zero"
        u_, s_, v_ = np.linalg.svd(uv.reshape(inmaps, n_outmap),
                                   full_matrices=False)
        u_ = np.dot(u_, np.diag(s_))  # fold s into u
        u_ = u_[:, :r]
        v_ = v_[:r, :]
        v_ = v_.reshape([r] + n_outmaps)

        u = nn.Variable([int(np.prod(inp.shape[base_axis:])), r],
                        need_grad=True)
        u.d = u_
        nn.parameter.set_parameter("U", u)

        v = nn.Variable([r] + n_outmaps, need_grad=True)
        v.d = v_
        nn.parameter.set_parameter("V", v)
    if fix_parameters == u.need_grad:
        u = u.get_unlinked_variable(need_grad=not fix_parameters)
    if fix_parameters == v.need_grad:
        v = v.get_unlinked_variable(need_grad=not fix_parameters)
        v.need_grad = not fix_parameters

    if with_bias and b_init is None:
        b_init = ConstantInitializer()

    b = None
    if with_bias:
        b = get_parameter_or_create(
            "b", n_outmaps, b_init, True, not fix_parameters)

    return F.affine(F.affine(inp, u, bias=None, base_axis=base_axis),
                    v, bias=b, base_axis=base_axis)


@parametric_function_api("bicon_affine", [
    ('W', 'Weight matrix in floating type', '(inmaps, outmaps)', True),
    ('Wb', 'Binarized weights', '(inmaps, outmaps)', False),
    ('b', 'Bias vector', '(outmaps,)', True),
])
def binary_connect_affine(inp, n_outmaps,
                          base_axis=1, quantize_zero_to=1.0,
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
        quantize_zero_to (float): Input value at zero is quantized to this value.
        w_init (:obj:`nnabla.initializer.BaseInitializer` or :obj:`numpy.ndarray`): Initializer for weight. By default, it is initialized with :obj:`nnabla.initializer.UniformInitializer` within the range determined by :obj:`nnabla.initializer.calc_uniform_lim_glorot`.  
        wb_init (:obj:`nnabla.initializer.BaseInitializer` or :obj:`numpy.ndarray`): Initializer for binary weight. By default, it is initialized with :obj:`nnabla.initializer.UniformInitializer` within the range determined by :obj:`nnabla.initializer.calc_uniform_lim_glorot`.   
        b_init (:obj:`nnabla.initializer.BaseInitializer` or :obj:`numpy.ndarray`): Initializer for bias. By default, it is initialized with zeros if `with_bias` is `True`.
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
        w_init, True, not fix_parameters)
    wb = get_parameter_or_create(
        "Wb", [int(np.prod(inp.shape[base_axis:]))] + n_outmaps,
        wb_init, False)
    b = None
    if with_bias:
        b = get_parameter_or_create(
            "b", n_outmaps, b_init, True, not fix_parameters)
    return F.binary_connect_affine(inp, w, wb, b, base_axis, quantize_zero_to)


@parametric_function_api("bwn_affine", [
    ('W', 'Weight matrix in floating type', '(inmaps, outmaps)', True),
    ('Wb', 'Binarized weights', '(inmaps, outmaps)', False),
    ('alpha', 'Scaling factor :math:`\\alpha`', '(outmaps,)', False),
    ('b', 'Bias vector', '(outmaps,)', True),
])
def binary_weight_affine(inp, n_outmaps,
                         base_axis=1, quantize_zero_to=1.0,
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
        quantize_zero_to (float): Input value at zero is quantized to this value.
        w_init (:obj:`nnabla.initializer.BaseInitializer` or :obj:`numpy.ndarray`): Initializer for the weight. By default, it is initialized with :obj:`nnabla.initializer.UniformInitializer` within the range determined by :obj:`nnabla.initializer.calc_uniform_lim_glorot`. 
        wb_init (:obj:`nnabla.initializer.BaseInitializer` or :obj:`numpy.ndarray`): Initializer for the binary weight. By default, it is initialized with :obj:`nnabla.initializer.UniformInitializer` within the range determined by :obj:`nnabla.initializer.calc_uniform_lim_glorot`.
        b_init (:obj:`nnabla.initializer.BaseInitializer` or :obj:`numpy.ndarray`): Initializer for the bias. By defalut, it is initialized with zeros if `with_bias` is `True`.
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
        w_init, True, not fix_parameters)
    wb = get_parameter_or_create(
        "Wb", [int(np.prod(inp.shape[base_axis:]))] + n_outmaps,
        wb_init, False)
    alpha = get_parameter_or_create(
        "alpha", n_outmaps, ConstantInitializer(0), False)
    b = None
    if with_bias:
        b = get_parameter_or_create(
            "b", n_outmaps, b_init, True, not fix_parameters)
    return F.binary_weight_affine(inp, w, wb, alpha, b, base_axis, quantize_zero_to)


@parametric_function_api("inq_affine", [
    ('W', 'Weight matrix in floating type', '(inmaps, outmaps)', True),
    ('I', 'Binary indicator matrix of fixed weights', '(inmaps, outmaps)', False),
    ('b', 'Bias vector', '(outmaps,)', True),
])
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
        quantize_zero_to (float): Input value at zero is quantized to this value.
        num_bits (int): Number of bits per weight. Value has to be larger than 1 as one bit is already used to code the value "0"
        inq_iterations (tuple of int): Tuple of iteration numbers at which we fix half of the weights.
        selection_algorithm (str): Chooses algorithm that is used to decide which weights are fixed. ("largest_abs" ... fix weights with largest absolute value, "random" ... fix weights randomly)
        seed (int): Random seed for INQ algorithm
        w_init (:obj:`nnabla.initializer.BaseInitializer` or :obj:`numpy.ndarray`): Initializer for weight. By default, it is initialized with :obj:`nnabla.initializer.UniformInitializer` within the range determined by :obj:`nnabla.initializer.calc_uniform_lim_glorot`.  
        i_init (:obj:`nnabla.initializer.BaseInitializer` or :obj:`numpy.ndarray`): Initializer for indicators (0 ... learnable, 1 ... fixed). By default, it is initialized with zeros.
        b_init (:obj:`nnabla.initializer.BaseInitializer` or :obj:`numpy.ndarray`): Initializer for bias. By default, it is initialized with zeros if `with_bias` is `True`.
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
        w_init, True, not fix_parameters)
    i = get_parameter_or_create(
        "I", [int(np.prod(inp.shape[base_axis:]))] + n_outmaps,
        i_init, False)
    b = None
    if with_bias:
        b = get_parameter_or_create(
            "b", n_outmaps, b_init, True, not fix_parameters)
    return F.inq_affine(inp, w, i, b, base_axis, num_bits, inq_iterations, selection_algorithm, seed)


@parametric_function_api("conv", [
    ('W', 'Filter weights', '(outmaps, inmaps // group, *kernel)', True),
    ('b', 'Bias vector', '(outmaps,)', True),
])
def convolution(inp, outmaps, kernel,
                pad=None, stride=None, dilation=None, group=1,
                w_init=None, b_init=None,
                base_axis=1, fix_parameters=False, rng=None, with_bias=True):
    """N-D Convolution with a bias term.

    For Dilated Convolution (a.k.a. Atrous Convolution), refer to:

    - Chen et al., DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs. https://arxiv.org/abs/1606.00915

    - Yu et al., Multi-Scale Context Aggregation by Dilated Convolutions. https://arxiv.org/abs/1511.07122

    Note:

        Convolution is a computationally intensive operation that
        should preferrably be run with the `cudnn` backend. NNabla
        then uses CuDNN library functions to determine and cache the
        fastest algorithm for the given set of convolution parameters,
        which results in additional memory consumption which may pose
        a problem for GPUs with insufficient memory size. In that
        case, the `NNABLA_CUDNN_WORKSPACE_LIMIT` environment variable
        can be used to restrict the choice of algorithms to those that
        fit the given workspace memory limit, expressed in bytes. In
        some cases it may also be desired to restrict the automatic
        search to algorithms that produce deterministic (reproducable)
        results. This can be requested by setting the the environment
        variable `NNABLA_CUDNN_DETERMINISTIC` to a non-zero value.

    Args:
        inp (~nnabla.Variable): N-D array.
        outmaps (int): Number of convolution kernels (which is equal to the number of output channels). For example, to apply convolution on an input with 16 types of filters, specify 16.
        kernel (:obj:`tuple` of :obj:`int`): Convolution kernel size. For example, to apply convolution on an image with a 3 (height) by 5 (width) two-dimensional kernel, specify (3,5).
        pad (:obj:`tuple` of :obj:`int`): Padding sizes for dimensions.
        stride (:obj:`tuple` of :obj:`int`): Stride sizes for dimensions.
        dilation (:obj:`tuple` of :obj:`int`): Dilation sizes for dimensions.
        group (int): Number of groups of channels. This makes connections across channels more sparse by grouping connections along map direction.
        w_init (:obj:`nnabla.initializer.BaseInitializer` or :obj:`numpy.ndarray`): Initializer for weight. By default, it is initialized with :obj:`nnabla.initializer.UniformInitializer` within the range determined by :obj:`nnabla.initializer.calc_uniform_lim_glorot`.  
        b_init (:obj:`nnabla.initializer.BaseInitializer` or :obj:`numpy.ndarray`): Initializer for bias. By default, it is initialized with zeros if `with_bias` is `True`.
        base_axis (int): Dimensions up to `base_axis` are treated as the sample dimensions.
        fix_parameters (bool): When set to `True`, the weights and biases will not be updated.
        rng (numpy.random.RandomState): Random generator for Initializer.
        with_bias (bool): Specify whether to include the bias term.

    Returns:
        :class:`~nnabla.Variable`: N-D array. See :obj:`~nnabla.functions.convolution` for the output shape.

    """
    if w_init is None:
        w_init = UniformInitializer(
            calc_uniform_lim_glorot(inp.shape[base_axis], outmaps, tuple(kernel)), rng=rng)
    if with_bias and b_init is None:
        b_init = ConstantInitializer()
    w = get_parameter_or_create(
        "W", (outmaps, inp.shape[base_axis] // group) + tuple(kernel),
        w_init, True, not fix_parameters)
    b = None
    if with_bias:
        b = get_parameter_or_create(
            "b", (outmaps,), b_init, True, not fix_parameters)
    return F.convolution(inp, w, b, base_axis, pad, stride, dilation, group)


@parametric_function_api("svd_conv", [
    ('U',
     'Decomposed filter weights :math:`{\\mathbf U}`', '(inmaps * r, *kernel)', True),
    ('V', 'Decomposed filter weights :math:`{\\mathbf V}`',
     '(outmaps, inmaps * r, 1, ...)', True),
    ('b', 'Bias vector', '(outmaps,)', True),
])
def svd_convolution(inp, outmaps, kernel, r, pad=None, stride=None,
                    dilation=None, uv_init=None, b_init=None, base_axis=1,
                    fix_parameters=False, rng=None, with_bias=True):
    """SVD convolution is a low rank approximation of the convolution
    layer. It can be seen as a depth wise convolution followed by a
    1x1 convolution.

    The flattened kernels for the i-th input map are expressed by
    their low rank approximation. The kernels for the i-th input
    :math:`{\\mathbf W_i}` are approximated with the singular value
    decomposition (SVD) and by selecting the :math:`{R}` dominant
    singular values and the corresponding singular vectors.

    .. math::
        {\\mathbf W_{:,i,:}} ~ {\\mathbf U_i} {\\mathbf V_i}.

    :math:`{\\mathbf U}` contains the weights of the depthwise
    convolution with multiplier :math:`{R}` and :math:`{\\mathbf V}`
    contains the weights of the 1x1 convolution.

    If `uv_init` is a numpy array, :math:`{\\mathbf U}` and
    :math:`{\\mathbf V}` are computed such that `uv_init` is
    approximated by :math:`{\\mathbf{UV}}`. If `uv_init` is `None` or
    an initializer, the product of :math:`{\\mathbf U}` and
    :math:`{\\mathbf V}` approximates the random initialization.

    If :math:`{\\mathbf U}` and :math:`{\\mathbf V}` exist in the
    context, they take precedence over `uv_init`.

    Suppose the kernel tensor of the convolution is of :math:`{O \\times I \\times K \\times K}` and
    the compression rate you want to specify is :math:`{CR}`, then you
    set :math:`{R}` as

    .. math::

        R = \\left\\lfloor \\frac{(1 - CR)OIK^2}{I(O + K^2)} \\right\\rfloor.

    Args:
        inp (~nnabla.Variable): N-D array.

        outmaps (int): Number of convolution kernels (which is equal
          to the number of output channels). For example, to apply
          convolution on an input with 16 types of filters, specify
          16.

        kernel (tuple): Convolution kernel size. For example,
          to apply convolution on an image with a 3 (height) by 5
          (width) two-dimensional kernel, specify (3, 5).

        r (int): Rank of the factorized layer.

        pad (tuple): Padding sizes (`int`) for dimensions.

        stride (tuple): Stride sizes (`int`) for dimensions.

        dilation (tuple): Dilation sizes (`int`) for dimensions.

        uv_init (:obj:`nnabla.initializer.BaseInitializer` or :obj:`numpy.ndarray`):
          Initializer for weight. By default, it is initialized with :obj:`nnabla.initializer.UniformInitializer` within the range determined by :obj:`nnabla.initializer.calc_uniform_lim_glorot`. 

        b_init (:obj:`nnabla.initializer.BaseInitializer` or :obj:`numpy.ndarray`): Initializer for bias. By default, it is initialized with zeros if `with_bias` is `True`.

        base_axis (int): Dimensions up to `base_axis` are treated as the
          sample dimensions.

        fix_parameters (bool): When set to `True`, the weights and
          biases will not be updated.

        rng (numpy.random.RandomState): Random generator for Initializer.

        with_bias (bool): Specify whether to include the bias term.

    Returns:
        :class:`~nnabla.Variable`: :math:`(B + 1)`-D array.
        (:math:`M_0 \\times \ldots \\times M_{B-1} \\times L`)

    """
    assert r > 0, "svd_convolution: The rank must larger than zero"

    if uv_init is None:
        uv_init = UniformInitializer(
            calc_uniform_lim_glorot(inp.shape[base_axis], outmaps,
                                    tuple(kernel)), rng=rng)

    if type(uv_init) is np.ndarray:
        # TODO: Assert that size of uv_init is correct
        # uv is initialize with numpy array
        uv = uv_init
    else:
        # uv is initialize from initializer
        uv = uv_init((outmaps, inp.shape[base_axis]) + tuple(kernel))

    # flatten kernels
    uv = uv.reshape((outmaps, inp.shape[base_axis], np.prod(kernel)))

    u = get_parameter('U')
    v = get_parameter('V')

    if (u is None) or (v is None):

        inmaps = inp.shape[base_axis]
        u_low_rank = np.zeros((inmaps, np.prod(kernel), r))
        v_low_rank = np.zeros((inmaps, r, outmaps))

        for i in range(inmaps):
            K = np.transpose(uv[:, i, :])
            u_, s_, v_ = np.linalg.svd(K, full_matrices=False)
            u_low_rank[i, :, :] = np.dot(u_[:, :r], np.diag(s_[:r]))
            v_low_rank[i, :, :] = v_[:r, :]

        # reshape U : (I,K*K,r) -> (I*r,K,K) for depthwise conv
        u = nn.Variable((inmaps * r,) + tuple(kernel),
                        need_grad=True)

        u.d = (np.transpose(u_low_rank, axes=(0, 2, 1))
               .reshape((inmaps * r,) + tuple(kernel)))

        nn.parameter.set_parameter("U", u)

        # reshape V :  (I,r,O) -> (O,I*r,1,1) for 1X1 conv
        kernel_one = (1,) * len(kernel)  # 1x1 for 2D convolution
        v = nn.Variable((outmaps, inmaps * r) + kernel_one,
                        need_grad=True)

        v.d = (np.transpose(v_low_rank, axes=(2, 0, 1))
               .reshape((outmaps, inmaps * r) + kernel_one))

        nn.parameter.set_parameter("V", v)

    if fix_parameters == u.need_grad:
        u = u.get_unlinked_variable(need_grad=not fix_parameters)
    if fix_parameters == v.need_grad:
        v = v.get_unlinked_variable(need_grad=not fix_parameters)
    if with_bias and b_init is None:
        b_init = ConstantInitializer()
    b = None
    if with_bias:
        b = get_parameter_or_create(
            "b", (outmaps,), b_init, True, not fix_parameters)

    y = F.depthwise_convolution(inp, u, bias=None, base_axis=base_axis,
                                pad=pad, stride=stride, dilation=dilation,
                                multiplier=r)

    y = F.convolution(y, v, bias=b, base_axis=base_axis, pad=None,
                      stride=None, dilation=None, group=1)
    return y


@parametric_function_api("cpd3_conv", [
    ('I',
     'Decomposed filter weights :math:`{\\mathbf I}`', '(r, inmaps, 1, ...)', True),
    ('K',
     'Decomposed filter weights :math:`{\\mathbf K}`', '(r, *kernel)', True),
    ('O',
     'Decomposed filter weights :math:`{\\mathbf O}`', '(outmaps, r, 1, ...)', True),
    ('b', 'Bias vector', '(outmaps,)', True),
])
def cpd3_convolution(inp, outmaps, kernel, r,
                     pad=None, stride=None, dilation=None,
                     oik_init=None, b_init=None,
                     base_axis=1, fix_parameters=False, rng=None, with_bias=True,
                     max_iter=500, stopping_criterion=1e-5, lambda_reg=0.0):
    """CP convolution is a low rank approximation of a convolution layer. A 3D tensor containing the parameter is built by collapsing the N-D kernels into 1D, then the tensor is decomposed into three matrices. The decomposed layer can be seen as linear combinations of the input feature maps to :math:`{R}` feature maps followed by a depthwise convolution and followed by linear combinations of the feature maps to compute the output feature maps.

    The CP decomposition allows to approximate the kernel tensor by :math:`{R}` rank-1 tensors of the form:

    .. math::

        \\sum_{r=1}^{R} \\lambda_r {\\mathbf{o}^{(r)} \\otimes \\mathbf{i}^{(r)} \\otimes \\mathbf{k}^{(r)}},

    where :math:`{\\lambda}_r` is the normalization coefficient and :math:`{\\otimes}` is the outer product.


    If `oik_init` is a numpy array, U and V are computed so that uv_init can be approximates from UV
    If `oik_init` is None or an initializer, the product of U and V approximate the randomly initialized array

    If `O`, `I` and `K` exist in context, they are used to initialize the layer and oik_init is not used.

    Suppose the kernel tensor of the affine is of :math:`{I \\times O}` and
    the compression rate you want to specify is :math:`{CR}`, then you
    set :math:`{R}` as

    .. math::

        R = \\left\\lfloor \\frac{(1 - CR)OIK^2}{O + I + K^2} \\right\\rfloor.

    References:
        - Lebedev, Vadim, Yaroslav Ganin, Maksim Rakhuba, Ivan Oseledets, and Victor Lempitsky,  "Speeding-up convolutional neural networks using fine-tuned cp-decomposition.", arXiv preprint arXiv:1412.6553 (2014).

        - Marcella Astrid, Seung-Ik Lee, "CP-decomposition with Tensor Power Method for Convolutional Neural Networks Compression", BigComp 2017.

    Args:
        inp (~nnabla.Variable): N-D array.
        outmaps (int): Number of convolution kernels (which is equal to the number of output channels). For example, to apply convolution on an input with 16 types of filters, specify 16.
        kernel (:obj:`tuple` of :obj:`int`): Convolution kernel size. For example, to apply convolution on an image with a 3 (height) by 5 (width) two-dimensional kernel, specify (3,5).
        r (int): rank of the factorized layer
        pad (:obj:`tuple` of :obj:`int`): Padding sizes for dimensions.
        stride (:obj:`tuple` of :obj:`int`): Stride sizes for dimensions.
        dilation (:obj:`tuple` of :obj:`int`): Dilation sizes for dimensions.
        oik_init (numpy array or :obj:`nnabla.initializer.BaseInitializer`): Initializer for weight. Initializer for weight. By default, it is initialized with :obj:`nnabla.initializer.UniformInitializer` within the range determined by :obj:`nnabla.initializer.calc_uniform_lim_glorot`. 
        b_init (:obj:`nnabla.initializer.BaseInitializer` or :obj:`numpy.ndarray`): Initializer for bias. It is initialized with zeros if `with_bias` is `True`.
        base_axis (int): Dimensions up to `base_axis` are treated as the sample dimensions.
        fix_parameters (bool): When set to `True`, the weights and biases will not be updated.
        rng (numpy.random.RandomState): Random generator for Initializer.
        with_bias (bool): Specify whether to include the bias term.
        max_iter (int): Max iteration of the ALS.
        stopping_criterion (float): Threshold for stopping the ALS.
                If the value is negative, the convergence check is ignored;
                in other words, it may reduce the computation time.
        lambda_reg (float): regularization parameter for the ALS. Larger
                lambda_reg means larger regularization.

    Returns:
        :class:`~nnabla.Variable`: :math:`(B + 1)`-D array. (:math:`M_0 \\times \ldots \\times M_{B-1} \\times L`)


    """

    if oik_init is None:
        oik_init = UniformInitializer(
            calc_uniform_lim_glorot(inp.shape[base_axis], outmaps, tuple(kernel)), rng=rng)

    if type(oik_init) is np.ndarray:
        # TODO: Assert that size of uv_init is correct
        # uv is initialize with numpy array
        oik = oik_init
    else:
        # uv is initialize from initializer
        oik = oik_init((outmaps, inp.shape[base_axis]) + tuple(kernel))

    # flatten kernels
    oik = oik.reshape((outmaps, inp.shape[base_axis], np.prod(kernel)))

    o = get_parameter('O')
    i = get_parameter('I')
    k = get_parameter('K')

    if (o is None) or (i is None) or (k is None):
        assert r > 0, "cpd3_convolution: The rank must larger than zero"
        from nnabla.utils.factorization import cpd
        als = cpd.ALS()
        U, lmbda = als.solve(X=oik, rank=r,
                             max_iter=max_iter,
                             stopping_criterion=stopping_criterion,
                             lambda_reg=lambda_reg,
                             dtype=oik.dtype,
                             rng=rng)

        o_ = U[0] * lmbda
        i_ = U[1]
        k_ = U[2]

        kernel_one = (1,) * len(kernel)  # 1x1 for 2D convolution
        inmaps = inp.shape[base_axis]

        # reshape I :  (I,r) -> (r,I,1,1)
        i = nn.Variable((r, inmaps) + kernel_one, need_grad=True)
        i.d = np.transpose(i_).reshape((r, inmaps) + kernel_one)
        nn.parameter.set_parameter("I", i)

        # reshape O :  (O,r) -> (O,r,1,1)
        o = nn.Variable((outmaps, r) + kernel_one,
                        need_grad=True)
        o.d = o_.reshape((outmaps, r) + kernel_one)
        nn.parameter.set_parameter("O", o)

        # reshape K :  (K*K,r) -> (r,K,K)
        k = nn.Variable((r,) + kernel, need_grad=True)
        k.d = np.transpose(k_).reshape((r,) + kernel)
        nn.parameter.set_parameter("K", k)

    if fix_parameters == o.need_grad:
        o = o.get_unlinked_variable(need_grad=not fix_parameters)
    if fix_parameters == i.need_grad:
        i = i.get_unlinked_variable(need_grad=not fix_parameters)
    if fix_parameters == k.need_grad:
        k = k.get_unlinked_variable(need_grad=not fix_parameters)
    if with_bias and b_init is None:
        b_init = ConstantInitializer()
    b = None
    if with_bias:
        b = get_parameter_or_create(
            "b", (outmaps,), b_init, True, not fix_parameters)

    y = F.convolution(inp, i, bias=None, base_axis=base_axis, pad=None, stride=None,
                      dilation=None, group=1)
    y = F.depthwise_convolution(y, k, bias=None, base_axis=base_axis,
                                pad=pad, stride=stride, dilation=dilation,
                                multiplier=1)
    y = F.convolution(y, o, bias=b, base_axis=base_axis, pad=None, stride=None,
                      dilation=None, group=1)
    return y


@parametric_function_api("bicon_conv", [
    ('W', 'Filter weights in float', '(outmaps, inmaps, *kernel)', True),
    ('Wb', 'Binarized filter weights', '(outmaps, inmaps, *kernel)', False),
    ('b', 'Bias vector', '(outmaps,)', True),
])
def binary_connect_convolution(inp, outmaps, kernel,
                               pad=None, stride=None, dilation=None, group=1,
                               quantize_zero_to=1.0,
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
        quantize_zero_to (float): Input value at zero is quantized to this value.
        w_init (:obj:`nnabla.initializer.BaseInitializer` or :obj:`numpy.ndarray`): Initializer for weight. By default, it is initialized with :obj:`nnabla.initializer.UniformInitializer` within the range determined by :obj:`nnabla.initializer.calc_uniform_lim_glorot`. 
        wb_init (:obj:`nnabla.initializer.BaseInitializer` or :obj:`numpy.ndarray`): Initializer for binary weight. By default, it is initialized with :obj:`nnabla.initializer.UniformInitializer` within the range determined by :obj:`nnabla.initializer.calc_uniform_lim_glorot`.   
        b_init (:obj:`nnabla.initializer.BaseInitializer` or :obj:`numpy.ndarray`): Initializer for bias. By default, it is initialized with zeros if `with_bias` is `True`.
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
        w_init, True, not fix_parameters)
    wb = get_parameter_or_create(
        "Wb", (outmaps, inp.shape[base_axis]) + tuple(kernel),
        wb_init, False)
    b = None
    if with_bias:
        b = get_parameter_or_create(
            "b", (outmaps,), b_init, True, not fix_parameters)
    return F.binary_connect_convolution(inp, w, wb, b, base_axis, pad, stride, dilation, group, quantize_zero_to)


@parametric_function_api("bwn_conv", [
    ('W', 'Filter weights in float', '(outmaps, inmaps, *kernel)', True),
    ('Wb', 'Binarized filter weights', '(outmaps, inmaps, *kernel)', False),
    ('alpha', 'Scaling factor :math:`\\alpha`', '(outmaps,)', False),
    ('b', 'Bias vector', '(outmaps,)', True),
])
def binary_weight_convolution(inp, outmaps, kernel,
                              pad=None, stride=None, dilation=None, group=1,
                              quantize_zero_to=1.0,
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
        quantize_zero_to (float): Input value at zero is quantized to this value.
        w_init (:obj:`nnabla.initializer.BaseInitializer` or :obj:`numpy.ndarray`): Initializer for weight. By default, it is initialized with :obj:`nnabla.initializer.UniformInitializer` within the range determined by :obj:`nnabla.initializer.calc_uniform_lim_glorot`. 
        wb_init (:obj:`nnabla.initializer.BaseInitializer` or :obj:`numpy.ndarray`): Initializer for binary weight. By default, it is initialized with :obj:`nnabla.initializer.UniformInitializer` within the range determined by :obj:`nnabla.initializer.calc_uniform_lim_glorot`.  
        b_init (:obj:`nnabla.initializer.BaseInitializer` or :obj:`numpy.ndarray`): Initializer for bias. By default, it is initialized with zeros if `with_bias` is `True`.
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
        w_init, True, not fix_parameters)
    wb = get_parameter_or_create(
        "Wb", (outmaps, inp.shape[base_axis]) + tuple(kernel),
        wb_init, False)
    alpha = get_parameter_or_create(
        "alpha", (outmaps, ), ConstantInitializer(0), False)
    b = None
    if with_bias:
        b = get_parameter_or_create(
            "b", (outmaps,), b_init, True, not fix_parameters)
    return F.binary_weight_convolution(inp, w, wb, alpha, b, base_axis, pad, stride, dilation, group, quantize_zero_to)


@parametric_function_api("inq_conv", [
    ('W', 'Filter weights in float', '(outmaps, inmaps, *kernel)', True),
    ('I', 'Binary indicator matrix of fixed weights',
     '(outmaps, inmaps, *kernel)', False),
    ('b', 'Bias vector', '(outmaps,)', True),
])
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
        w_init (:obj:`nnabla.initializer.BaseInitializer` or :obj:`numpy.ndarray`): Initializer for the weight. By default, it is initialized with :obj:`nnabla.initializer.UniformInitializer` within the range determined by :obj:`nnabla.initializer.calc_uniform_lim_glorot`. 
        i_init (:obj:`nnabla.initializer.BaseInitializer` or :obj:`numpy.ndarray`): Initializer for the indicators (0 ... learnable, 1 ... fixed). By default, it is initialized with zeros.
        b_init (:obj:`nnabla.initializer.BaseInitializer` or :obj:`numpy.ndarray`): Initializer for the bias. By default, it is initialized with zeros if `with_bias` is `True`.
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
        w_init, True, not fix_parameters)
    i = get_parameter_or_create(
        "I", (outmaps, inp.shape[base_axis]) + tuple(kernel),
        i_init, False)
    b = None
    if with_bias:
        b = get_parameter_or_create(
            "b", (outmaps,), b_init, True, not fix_parameters)
    return F.inq_convolution(inp, w, i, b, base_axis, pad, stride, dilation, group, num_bits, inq_iterations, selection_algorithm, seed)


@parametric_function_api("depthwise_conv", [
    ('W', 'Filter weights', '(inmaps * multiplier, *kernel)', True),
    ('b', 'Bias vector', '(inmaps * multiplier,)', True),
])
def depthwise_convolution(inp, kernel, pad=None, stride=None, dilation=None,
                          multiplier=1, w_init=None, b_init=None, base_axis=1,
                          fix_parameters=False, rng=None, with_bias=True):
    """
    N-D Depthwise Convolution with a bias term.

    Reference:

    - F. Chollet: Chollet, Francois. "Xception: Deep Learning with Depthwise Separable Convolutions. https://arxiv.org/abs/1610.02357

    Args:
        inp (~nnabla.Variable): N-D array.
        kernel (:obj:`tuple` of :obj:`int`): Convolution kernel size. For example, to apply convolution on an image with a 3 (height) by 5 (width) two-dimensional kernel, specify (3,5).
        pad (:obj:`tuple` of :obj:`int`): Padding sizes for dimensions.
        stride (:obj:`tuple` of :obj:`int`): Stride sizes for dimensions.
        dilation (:obj:`tuple` of :obj:`int`): Dilation sizes for dimensions.
        multiplier (:obj:`int`): Number of output feature maps per input feature map.
        w_init (:obj:`nnabla.initializer.BaseInitializer` or :obj:`numpy.ndarray`): Initializer for weight.  By default, it is initialized with :obj:`nnabla.initializer.UniformInitializer` within the range determined by :obj:`nnabla.initializer.calc_uniform_lim_glorot`. 
        b_init (:obj:`nnabla.initializer.BaseInitializer` or :obj:`numpy.ndarray`): Initializer for bias. By default, it is initialized with zeros if `with_bias` is `True`.
        base_axis (int): Dimensions up to `base_axis` are treated as the sample dimensions.
        fix_parameters (bool): When set to `True`, the weights and biases will not be updated.
        rng (numpy.random.RandomState): Random generator for Initializer.
        with_bias (bool): Specify whether to include the bias term.

    Returns:
        :class:`~nnabla.Variable`: N-D array. See :obj:`~nnabla.functions.depthwise_convolution` for the output shape.

    """
    if w_init is None:
        w_init = UniformInitializer(
            calc_uniform_lim_glorot(
                inp.shape[base_axis] * multiplier,
                inp.shape[base_axis],
                tuple(kernel)),
            rng=rng)
    if with_bias and b_init is None:
        b_init = ConstantInitializer()
    w = get_parameter_or_create(
        "W", (inp.shape[base_axis] * multiplier,) + tuple(kernel),
        w_init, True, not fix_parameters)
    b = None
    if with_bias:
        b = get_parameter_or_create(
            "b", (inp.shape[base_axis] * multiplier,),
            b_init, True, not fix_parameters)
    return F.depthwise_convolution(inp, w, b, base_axis, pad, stride, dilation,
                                   multiplier)


@parametric_function_api("deconv", [
    ('W', 'Filter weights', '(inmaps, outmaps // group, *kernel)', True),
    ('b', 'Bias vector', '(outmaps,)', True),
])
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
        w_init (:obj:`nnabla.initializer.BaseInitializer` or :obj:`numpy.ndarray`): Initializer for weight. By default, it is initialized with :obj:`nnabla.initializer.UniformInitializer` within the range determined by :obj:`nnabla.initializer.calc_uniform_lim_glorot`.  
        b_init (:obj:`nnabla.initializer.BaseInitializer` or :obj:`numpy.ndarray`): Initializer for bias. By default, it is initialized with zeros if `with_bias` is `True`.
        base_axis (int): Dimensions up to `base_axis` are treated as the sample dimensions.
        fix_parameters (bool): When set to `True`, the weights and biases will not be updated.
        rng (numpy.random.RandomState): Random generator for Initializer.
        with_bias (bool): Specify whether to include the bias term.

    Returns:
        :class:`~nnabla.Variable`: N-D array. See :obj:`~nnabla.functions.deconvolution` for the output shape.

    """
    if w_init is None:
        w_init = UniformInitializer(
            calc_uniform_lim_glorot(outmaps, inp.shape[base_axis], tuple(kernel)), rng=rng)
    if with_bias and b_init is None:
        b_init = ConstantInitializer()
    w = get_parameter_or_create(
        "W", (inp.shape[base_axis], outmaps // group) + tuple(kernel),
        w_init, True, not fix_parameters)
    b = None
    if with_bias:
        b = get_parameter_or_create(
            "b", (outmaps,), b_init, True, not fix_parameters)
    return F.deconvolution(inp, w, b, base_axis, pad, stride, dilation, group)


@parametric_function_api("depthwise_deconv", [
    ('W', 'Filter weights', '(inmaps,) + kernel', True),
    ('b', 'Bias vector', '(inmaps / divisor,)', True),
])
def depthwise_deconvolution(inp, kernel, pad=None, stride=None, dilation=None,
                            divisor=1, w_init=None, b_init=None, base_axis=1,
                            fix_parameters=False, rng=None, with_bias=True):
    """Depthwise deconvolution computes the transposed depthwise
    convolution for one-dimensional and two-dimensional input data.

    Args:
        inp (~nnabla.Variable): N-D array.
        kernel (:obj:`tuple` of :obj:`int`): Convolution kernel size. For example, to apply convolution on an image with a 3 (height) by 5 (width) two-dimensional kernel, specify (3,5).
        pad (:obj:`tuple` of :obj:`int`): Padding sizes for dimensions.
        stride (:obj:`tuple` of :obj:`int`): Stride sizes for dimensions.
        dilation (:obj:`tuple` of :obj:`int`): Dilation sizes for dimensions.
        divisor (:obj:`int`): Number of input feature maps per output feature map.
        w_init (:obj:`nnabla.initializer.BaseInitializer` or :obj:`numpy.ndarray`): Initializer for weight. By default, it is initialized with :obj:`nnabla.initializer.UniformInitializer` within the range determined by :obj:`nnabla.initializer.calc_uniform_lim_glorot`.  
        b_init (:obj:`nnabla.initializer.BaseInitializer` or :obj:`numpy.ndarray`): Initializer for bias. By default, it is initialized with zeros if `with_bias` is `True`.
        base_axis (int): Dimensions up to `base_axis` are treated as the sample dimensions.
        fix_parameters (bool): When set to `True`, the weights and biases will not be updated.
        rng (numpy.random.RandomState): Random generator for Initializer.
        with_bias (bool): Specify whether to include the bias term.

    Returns:
        :class:`~nnabla.Variable`: N-D array. See :obj:`~nnabla.functions.depthwise_deconvolution` for the output shape.

    """
    if w_init is None:
        w_init = UniformInitializer(
            calc_uniform_lim_glorot(
                inp.shape[base_axis],
                inp.shape[base_axis],
                tuple(kernel)),
            rng=rng)
    if with_bias and b_init is None:
        b_init = ConstantInitializer()
    w = get_parameter_or_create(
        "W", (inp.shape[base_axis],) + tuple(kernel),
        w_init, True, not fix_parameters)
    b = None
    if with_bias:
        b = get_parameter_or_create(
            "b", (inp.shape[base_axis] // divisor,),
            b_init, True, not fix_parameters)
    return F.depthwise_deconvolution(inp, w, b, base_axis, pad, stride,
                                     dilation, divisor)


@parametric_function_api("bn", [
    ('beta', 'Trainable bias :math:`\\beta`', '<see above>', True),
    ('gamma', 'Trainable scaling factor :math:`\\gamma`', '<see above>', True),
    ('mean', 'Moving average of batch mean', '<see above>', False),
    ('var', 'Moving average of batch variance', '<see above>', False),
])
def batch_normalization(inp, axes=[1], decay_rate=0.9, eps=1e-5,
                        batch_stat=True, output_stat=False, fix_parameters=False,
                        param_init=None):
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
            E.g. ``{'beta': ConstantIntializer(0), 'gamma': np.ones(gamma_shape) * 2}``.

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
    assert len(axes) == 1
    shape_stat = [1 for _ in inp.shape]
    shape_stat[axes[0]] = inp.shape[axes[0]]

    if param_init is None:
        param_init = {}
    beta_init = param_init.get('beta', ConstantInitializer(0))
    gamma_init = param_init.get('gamma', ConstantInitializer(1))
    mean_init = param_init.get('mean', ConstantInitializer(0))
    var_init = param_init.get('var', ConstantInitializer(1))
    beta = get_parameter_or_create(
        "beta", shape_stat, beta_init, True, not fix_parameters)
    gamma = get_parameter_or_create(
        "gamma", shape_stat, gamma_init, True, not fix_parameters)
    mean = get_parameter_or_create(
        "mean", shape_stat, mean_init, False)
    var = get_parameter_or_create(
        "var", shape_stat, var_init, False)
    return F.batch_normalization(inp, beta, gamma, mean, var, axes,
                                 decay_rate, eps, batch_stat, output_stat)


@parametric_function_api("mean_subtraction", [
    ('mean', 'Moving average', 'inp.shape[base_axis:]', False),
    ('t', 'Minibatch counter used in forward pass', '(1,)', False),
])
def mean_subtraction(inp, base_axis=1, update_running_mean=True, fix_parameters=False):
    """
    Mean subtraction layer.

    It subtracts the mean of the elements of the input array,
    and normalizes it to :math:`0`. Preprocessing arrays with this function has the effect of improving accuracy
    in various tasks such as image classification.

    At training time, this function is defined as

    .. math::

        \\begin{array}{lcl}
        \\mu &=& \\frac{1}{M} \\sum x_i \\\\
        y_i &=& x_i - \\mu
        \\end{array}

    At testing time, the mean values used are those that were computed during training by moving average.

    Note:
        The backward performs an approximated differentiation that takes into account only the latest mini-batch.

    Args:
        inp (~nnabla.Variable): N-D array of input.
        base_axis (int): Base axis of Mean Subtraction operation. Dimensions up to base_axis is treated as sample dimension.
        update_running_mean (bool): When set to `True`, the running mean will not be updated.
        fix_parameters (bool): dummy parameter. This argument dose not affect anything.

    Returns:
        ~nnabla.Variable: N-D array.
    """
    assert len(inp.shape) >= base_axis
    shape = inp.shape[base_axis:]
    mean = get_parameter_or_create(
        "mean", shape, ConstantInitializer(0), False)
    t = get_parameter_or_create(
        "t", (1, ), ConstantInitializer(0), False)
    return F.mean_subtraction(inp, mean, t, base_axis=base_axis, update_running_mean=update_running_mean)


@parametric_function_api("embed", [
    ('W', 'Embedding matrix', '(n_inputs, n_features)', True),
])
def embed(inp, n_inputs, n_features, fix_parameters=False):
    """ Embed.

    Embed slices a matrix/tensor with indexing array/tensor. Weights are initialized with :obj:`nnabla.initializer.UniformInitializer` within the range of :math:`-\\sqrt{3}` and :math:`\\sqrt{3}`.

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
                                UniformInitializer((-np.sqrt(3.), np.sqrt(3))), True, not fix_parameters)
    return F.embed(inp, w)


@parametric_function_api("prelu", [
    ('slope', 'Negative slope',
     'tuple() if shared else (inp.shape[base_axis],)', True),
])
def prelu(inp, base_axis=1, shared=True, fix_parameters=False):
    """
    Parametrized Rectified Linear Unit function defined as

    .. math::
        y_i = \max(0, x_i) + w_i \min(0, -x_i)

    where negative slope :math:`w` is learned and can vary across channels (an
    axis specified with base_axis). Weights are initialized with :math:`-1`.

    Args:
        x(~nnabla.Variable): N-D array as input
        base_axis(int): Dimensions up to base_axis is treated as sample dimension.
        shared(bool): Use shared weight value or not
        fix_parameters (bool): When set to `True`, the negative slope values
            will not be updated.

    Returns:
        ~nnabla.Variable: N-D array.

    """
    shape = tuple() if shared else (inp.shape[base_axis],)
    w = get_parameter_or_create("slope", shape,
                                ConstantInitializer(-1), True, not fix_parameters)
    return F.prelu(inp, w, base_axis)


@parametric_function_api("fp_quantized_affine", [
    ('W', 'Weight matrix in float', '(inmaps, outmaps)', True),
    ('b', 'Bias vector in float', '(outmaps,)', True),
    ('W_q', 'Quantized weights', '(inmaps, outmaps)', False),
    ('b_q', 'Quantized biases', '(outmaps,)', False),
])
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
        w_init (:obj:`nnabla.initializer.BaseInitializer` or :obj:`numpy.ndarray`): Initializer for weight. By default, it is initialized with :obj:`nnabla.initializer.UniformInitializer` within the range determined by :obj:`nnabla.initializer.calc_uniform_lim_glorot`. 
        b_init (:obj:`nnabla.initializer.BaseInitializer` or :obj:`numpy.ndarray`): Initializer for bias. By default, it is initialized with zeros if `with_bias` is `True`.
        fix_parameters (bool): When set to `True`, the weights and biases will not be updated.
        rng (numpy.random.RandomState): Random generator for Initializer.
        with_bias (bool): Specify whether to include the bias term.
        quantize_w (bool): Quantize weights if `True`.
        sign_w (bool): Use signed quantization if `True`.
        n_w (int): Bit width used for weight.
        delta_w (float): Step size for weight.
        ste_fine_grained_w (bool): STE is fine-grained if `True`.
        quantize_b (bool): Quantize bias if `True`.
        n_b (int): Bit width used for bias.
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
        w_init, True, not fix_parameters)

    # Quantized Weight
    if quantize_w:
        w_q = get_parameter_or_create(
            "W_q", [int(np.prod(inp.shape[base_axis:]))] + n_outmaps,
            w_init, False)
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
            "b", n_outmaps, b_init, True, not fix_parameters)
        if quantize_b:
            b_q = get_parameter_or_create(
                "b_q", n_outmaps, b_init, False)
            # Link computation graph
            real_b_q = F.fixed_point_quantize(b, quantize=quantize_b,
                                              sign=sign_b, n=n_b, delta=delta_b,
                                              ste_fine_grained=ste_fine_grained_b,
                                              outputs=[b_q.data])
            real_b_q.persistent = True
        else:
            real_b_q = b

    return F.affine(inp, real_w_q, real_b_q, base_axis)


@parametric_function_api("fp_quantized_conv", [
    ('W', 'Filter weights in float', '(outmaps, inmaps // group, *kernel)', True),
    ('b', 'Bias vector in float', '(outmaps,)', True),
    ('W_q', 'Quantized weights', '(outmaps, inmaps // group, *kernel)', False),
    ('b_q', 'Quantized biases', '(outmaps,)', False),
])
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
        w_init (:obj:`nnabla.initializer.BaseInitializer` or :obj:`numpy.ndarray`): Initializer for weight. By default, it is initialized with :obj:`nnabla.initializer.UniformInitializer` within the range determined by :obj:`nnabla.initializer.calc_uniform_lim_glorot`.  
        b_init (:obj:`nnabla.initializer.BaseInitializer` or :obj:`numpy.ndarray`): Initializer for bias. By default, it is initialized with zeros if `with_bias` is `True`.
        base_axis (int): Dimensions up to `base_axis` are treated as the sample dimensions.
        fix_parameters (bool): When set to `True`, the weights and biases will not be updated.
        rng (numpy.random.RandomState): Random generator for Initializer.
        with_bias (bool): Specify whether to include the bias term.
        quantize_w (bool): Quantize weights if `True`.
        quantize_bias (bool): Quantize bias if `True`.
        sign_w (bool): Use signed quantization if `True`.
        n_w (int): Bit width used for weight.
        delta_w (float): Step size for weight.
        ste_fine_grained_w (bool): STE is fine-grained if `True`.
        quantize_b (bool): Quantize bias if `True`.
        n_b (int): Bit width used for bias.
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
        "W", (outmaps, inp.shape[base_axis] // group) + tuple(kernel),
        w_init, True, not fix_parameters)

    # Quantized Weight
    if quantize_w:
        w_q = get_parameter_or_create(
            "W_q", (outmaps, inp.shape[base_axis] // group) + tuple(kernel),
            w_init, False)
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
            "b", (outmaps,), b_init, True, not fix_parameters)
        if quantize_b:
            b_q = get_parameter_or_create(
                "b_q", (outmaps,), b_init, False)
            # Link computation graph
            real_b_q = F.fixed_point_quantize(b, quantize=quantize_b,
                                              sign=sign_b, n=n_b, delta=delta_b,
                                              ste_fine_grained=ste_fine_grained_b,
                                              outputs=[b_q.data])
            real_b_q.persistent = True
        else:
            real_b_q = b

    return F.convolution(inp, real_w_q, real_b_q, base_axis, pad, stride, dilation, group)


@parametric_function_api("pow2_quantized_affine", [
    ('W', 'Weight matrix in float', '(inmaps, outmaps)', True),
    ('b', 'Bias vector in float', '(outmaps,)', True),
    ('W_q', 'Quantized weights', '(inmaps, outmaps)', False),
    ('b_q', 'Quantized biases', '(outmaps,)', False),
])
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
        w_init (:obj:`nnabla.initializer.BaseInitializer` or :obj:`numpy.ndarray`): Initializer for weight. By default, it is initialized with :obj:`nnabla.initializer.UniformInitializer` within the range determined by :obj:`nnabla.initializer.calc_uniform_lim_glorot`.  
        b_init (:obj:`nnabla.initializer.BaseInitializer` or :obj:`numpy.ndarray`): Initializer for bias. By default, it is initialized with zeros if `with_bias` is `True`.
        fix_parameters (bool): When set to `True`, the weights and biases will not be updated.
        rng (numpy.random.RandomState): Random generator for Initializer.
        with_bias (bool): Specify whether to include the bias term.
        quantize_w (bool): Quantize weights if `True`.
        sign_w (bool): Use signed quantization if `True`.
        with_zero_w (bool): Indicate using zero as a quantized value. Default is false.
        n_w (int): Bit width used for weight.
        m_w (int): :math:`2^m` is upper bound and :math:`-2^m` is lower bound for weights. Default is 2.
        ste_fine_grained_w (bool): STE is fine-grained if `True`.
        quantize_b (bool): Quantize bias if `True`.
        with_zero_b (bool): Indicate using zero as a quantized value. Default is false.
        n_b (int): Bit width used for bias.
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
        w_init, True, not fix_parameters)

    # Quantized Weight
    if quantize_w:
        w_q = get_parameter_or_create(
            "W_q", [int(np.prod(inp.shape[base_axis:]))] + n_outmaps,
            w_init, False)
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
            "b", n_outmaps, b_init, True, not fix_parameters)
        if quantize_b:
            b_q = get_parameter_or_create(
                "b_q", n_outmaps, b_init, False)
            real_b_q = F.pow2_quantize(b, quantize=quantize_b,
                                       sign=sign_b, with_zero=with_zero_b,
                                       n=n_b, m=m_b, ste_fine_grained=ste_fine_grained_b,
                                       outputs=[b_q.data])
            real_b_q.persistent = True
        else:
            real_b_q = b

    return F.affine(inp, real_w_q, real_b_q, base_axis)


@parametric_function_api("pow2_quantized_conv", [
    ('W', 'Filter weights in float', '(outmaps, inmaps // group, *kernel)', True),
    ('b', 'Bias vector in float', '(outmaps,)', True),
    ('W_q', 'Quantized weights', '(outmaps, inmaps // group, *kernel)', False),
    ('b_q', 'Quantized biases', '(outmaps,)', False),
])
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
        w_init (:obj:`nnabla.initializer.BaseInitializer` or :obj:`numpy.ndarray`): Initializer for weight. By default, it is initialized with :obj:`nnabla.initializer.UniformInitializer` within the range determined by :obj:`nnabla.initializer.calc_uniform_lim_glorot`.  
        b_init (:obj:`nnabla.initializer.BaseInitializer` or :obj:`numpy.ndarray`): Initializer for bias. By default, it is initialized with zeros if `with_bias` is `True`.
        base_axis (int): Dimensions up to `base_axis` are treated as the sample dimensions.
        fix_parameters (bool): When set to `True`, the weights and biases will not be updated.
        rng (numpy.random.RandomState): Random generator for Initializer.
        with_bias (bool): Specify whether to include the bias term.
        quantize_w (bool): Quantize weights if `True`.
        sign_w (bool): Use signed quantization if `True`.
        n_w (int): Bit width used for weight.
        m_w (int): :math:`2^m` is upper bound and :math:`-2^m` is lower bound for weights. Default is 2.
        ste_fine_grained_w (bool): STE is fine-grained if `True`.
        quantize_b (bool): Quantize bias if `True`.
        sign_b (bool): Use signed quantization if `True`.
        n_b (int): Bit width used for bias.
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
        "W", (outmaps, inp.shape[base_axis] // group) + tuple(kernel),
        w_init, True, not fix_parameters)

    # Quantized Weight
    if quantize_w:
        w_q = get_parameter_or_create(
            "W_q", (outmaps, inp.shape[base_axis] // group) + tuple(kernel),
            w_init, False)

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
            "b", (outmaps,), b_init, True, not fix_parameters)
        if quantize_b:
            b_q = get_parameter_or_create(
                "b_q", (outmaps,), b_init, False)
            # Link computation graph
            real_b_q = F.pow2_quantize(b, quantize=quantize_b,
                                       sign=sign_b, with_zero=with_zero_b,
                                       n=n_b, m=m_b, ste_fine_grained=ste_fine_grained_b,
                                       outputs=[b_q.data])
            real_b_q.persistent = True
        else:
            real_b_q = b

    return F.convolution(inp, real_w_q, real_b_q, base_axis, pad, stride, dilation, group)


@parametric_function_api("pruned_affine", [
    ('W', 'Weight matrix in float', '(inmaps, outmaps)', True),
    ('b', 'Bias vector in float', '(outmaps,)', True),
    ('W_q', 'Qunatized weights', '(inmaps, outmaps)', False),
    ('b_q', 'Quantized biases', '(outmaps,)', False),
])
def pruned_affine(inp, n_outmaps,
                  base_axis=1,
                  w_init=None, b_init=None,
                  fix_parameters=False, rng=None, with_bias=True,
                  prune_w=True, rate_w=0.9, prune_b=True, rate_b=0.9):
    """Pruned Affine.

    Pruned Affine is the affine function, 
    except the definition of the inner product is modified.
    The input-output relation of this function is as follows:

    .. math::

        y_j = \sum_{i} Q(w_{ji}) x_i, 

    where :math:`Q(w_{ji})` is the pruning function, i.e., `F.prune`.

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
        w_init (:obj:`nnabla.initializer.BaseInitializer` or :obj:`numpy.ndarray`): Initializer for weight.
        b_init (:obj:`nnabla.initializer.BaseInitializer` or :obj:`numpy.ndarray`): Initializer for bias.
        fix_parameters (bool): When set to `True`, the weights and biases will not be updated.
        rng (numpy.random.RandomState): Random generator for Initializer.
        with_bias (bool): Specify whether to include the bias term.
        prune_w (bool): Quantize weights if `True`.
        rate_w (float): Pruning rate for weights.
        prune_b (bool): Quantize bias if `True`.
        rate_b (float): Pruning rate for bias.


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
        w_init, True, not fix_parameters)

    # sparsed Weight
    if prune_w:
        w_q = get_parameter_or_create(
            "W_q", [int(np.prod(inp.shape[base_axis:]))] + n_outmaps,
            w_init, False)
        # Link computation graph
        real_w_q = F.prune(w, rate=rate_w, outputs=[w_q.data])
        real_w_q.persistent = True
    else:
        real_w_q = w

    # Bias
    # Floating
    real_b_q = None
    if with_bias:
        b = get_parameter_or_create(
            "b", n_outmaps, b_init, True, not fix_parameters)
        if prune_b:
            b_q = get_parameter_or_create(
                "b_q", n_outmaps, b_init, False)
            # Link computation graph
            real_b_q = F.prune(b, rate=rate_b, outputs=[b_q.data])
            real_b_q.persistent = True
        else:
            real_b_q = b

    return F.affine(inp, real_w_q, real_b_q, base_axis)


@parametric_function_api("pruned_conv", [
    ('W', 'Filter weights in float', '(outmaps, inmaps // group, *kernel)', True),
    ('b', 'Bias vector in float', '(outmaps,)', True),
    ('W_q', 'Qunatized weights', '(outmaps, inmaps // group, *kernel)', False),
    ('b_q', 'Quantized biases', '(outmaps,)', False),
])
def pruned_convolution(inp, outmaps, kernel,
                       pad=None, stride=None, dilation=None, group=1,
                       w_init=None, b_init=None,
                       base_axis=1, fix_parameters=False, rng=None, with_bias=True,
                       prune_w=True, rate_w=0.9, prune_b=True, rate_b=0.9):
    """Pruned Convolution.

    Pruned Convolution is the convolution function, 
    except the definition of the inner product is modified.
    The input-output relation of this function is as follows:

    .. math::

        y_{n, a, b} = \sum_{m} \sum_{i} \sum_{j} Q(w_{n, m, i, j}) x_{m, a + i, b + j}, 

    where :math:`Q(w_{ji})` is the pruning function, i.e., `F.prune`.

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
        w_init (:obj:`nnabla.initializer.BaseInitializer` or :obj:`numpy.ndarray`): Initializer for weight.
        b_init (:obj:`nnabla.initializer.BaseInitializer` or :obj:`numpy.ndarray`): Initializer for bias.
        base_axis (int): Dimensions up to `base_axis` are treated as the sample dimensions.
        fix_parameters (bool): When set to `True`, the weights and biases will not be updated.
        rng (numpy.random.RandomState): Random generator for Initializer.
        with_bias (bool): Specify whether to include the bias term.
        prune_w (bool): Quantize weights if `True`.
        rate_w (float): Pruning rate for weights.
        prune_b (bool): Quantize bias if `True`.
        rate_b (float): Pruning rate for bias.

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
        "W", (outmaps, inp.shape[base_axis] // group) + tuple(kernel),
        w_init, True, not fix_parameters)

    # Quantized Weight
    if prune_w:
        w_q = get_parameter_or_create(
            "W_q", (outmaps, inp.shape[base_axis] // group) + tuple(kernel),
            w_init, False)
        # Link computation graph
        real_w_q = F.prune(w, rate=rate_w, outputs=[w_q.data])
        real_w_q.persistent = True
    else:
        real_w_q = w

    # Bias
    # Floating
    real_b_q = None
    if with_bias:
        b = get_parameter_or_create(
            "b", (outmaps,), b_init, True, not fix_parameters)
        if prune_b:
            b_q = get_parameter_or_create(
                "b_q", (outmaps,), b_init, False)
            # Link computation graph
            real_b_q = F.prune(b, rate=rate_b, outputs=[b_q.data])
            real_b_q.persistent = True
        else:
            real_b_q = b

    return F.convolution(inp, real_w_q, real_b_q, base_axis, pad, stride, dilation, group)


@parametric_function_api("lstm", [
    ('affine/W', 'Stacked weight matrixes of LSTM block',
     '(inmaps, 4, state_size)', True),
    ('affine/b', 'Stacked bias vectors of LSTM block', '(4, state_size,)', True),
])
def lstm(x, h, c, state_size, w_init=None, b_init=None, fix_parameters=False):
    """Long Short-Term Memory.

    Long Short-Term Memory, or LSTM, is a building block for recurrent neural networks (RNN) layers.
    LSTM unit consists of a cell and input, output, forget gates whose functions are defined as following:

    .. math::
        f_t&&=\\sigma(W_fx_t+U_fh_{t-1}+b_f) \\\\
        i_t&&=\\sigma(W_ix_t+U_ih_{t-1}+b_i) \\\\
        o_t&&=\\sigma(W_ox_t+U_oh_{t-1}+b_o) \\\\
        c_t&&=f_t\\odot c_{t-1}+i_t\\odot\\tanh(W_cx_t+U_ch_{t-1}+b_c) \\\\
        h_t&&=o_t\\odot\\tanh(c_t).

    References:

        S. Hochreiter, and J. Schmidhuber. "Long Short-Term Memory."
        Neural Computation. 1997.

    Args:
        x (~nnabla.Variable): Input N-D array with shape (batch_size, input_size).
        h (~nnabla.Variable): Input N-D array with shape (batch_size, state_size).
        c (~nnabla.Variable): Input N-D array with shape (batch_size, state_size).
        state_size (int): Internal state size is set to `state_size`.
        w_init (:obj:`nnabla.initializer.BaseInitializer` or :obj:`numpy.ndarray`, optional): Initializer for weight. By default, it is initialized with :obj:`nnabla.initializer.UniformInitializer` within the range determined by :obj:`nnabla.initializer.calc_uniform_lim_glorot`.  
        b_init (:obj:`nnabla.initializer.BaseInitializer` or :obj:`numpy.ndarray`, optional): Initializer for bias. By default, it is initialized with zeros if `with_bias` is `True`.
        fix_parameters (bool): When set to `True`, the weights and biases will not be updated.

    Returns:
        :class:`~nnabla.Variable`

    """

    xh = F.concatenate(*(x, h), axis=1)
    iofc = affine(xh, (4, state_size), w_init=w_init,
                  b_init=b_init, fix_parameters=fix_parameters)
    i_t, o_t, f_t, gate = F.split(iofc, axis=1)
    c_t = F.sigmoid(f_t) * c + F.sigmoid(i_t) * F.tanh(gate)
    h_t = F.sigmoid(o_t) * F.tanh(c_t)
    return h_t, c_t


class LSTMCell:
    def __init__(self, batch_size, state_size, h=None, c=None, name=None):
        """
        Initializes an LSTM cell.

        Args:
            batch_size (int): Internal batch size is set to `batch_size`.
            state_size (int): Internal state size is set to `state_size`.
            h (~nnabla.Variable): Input N-D array with shape (batch_size, state_size). If not specified, it is initialized to zero by default.
            c (~nnabla.Variable): Input N-D array with shape (batch_size, state_size). If not specified, it is initialized to zero by default.
            name (str): Name for this LSTM Cell.
        """
        self.batch_size = batch_size
        self.state_size = state_size
        self.name = name
        if h:  # when user defines h
            self.h = h
        else:
            self.h = nn.Variable((self.batch_size, self.state_size))
            self.h.data.zero()
        if c:  # when user defines c
            self.c = c
        else:
            self.c = nn.Variable((self.batch_size, self.state_size))
            self.c.data.zero()

    def reset_state(self):
        """
        Resets states h and c to zero.
        """

        self.h.data.zero()
        self.c.data.zero()

    def __call__(self, x, w_init=None, b_init=None, fix_parameters=False):
        """
        Updates h and c by calling lstm function.

        Args:
            x (~nnabla.Variable): Input N-D array with shape (batch_size, input_size).
            w_init (:obj:`nnabla.initializer.BaseInitializer` or :obj:`numpy.ndarray`, optional): Initializer for weight. By default, it is initialized with :obj:`nnabla.initializer.UniformInitializer` within the range determined by :obj:`nnabla.initializer.calc_uniform_lim_glorot`.  
            b_init (:obj:`nnabla.initializer.BaseInitializer` or :obj:`numpy.ndarray`, optional): Initializer for bias. By default, it is initialized with zeros if `with_bias` is `True`.
            fix_parameters (bool): When set to `True`, the weights and biases will not be updated.

        """
        self.h, self.c = lstm(
            x, self.h, self.c, self.state_size,
            w_init, b_init,
            fix_parameters=fix_parameters, name=self.name)
        return self.h
