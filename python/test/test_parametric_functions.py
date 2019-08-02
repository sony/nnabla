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

import pytest
import numpy as np
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.initializer as I


@pytest.fixture(scope='function')
def g_rng(request):
    nn.clear_parameters()
    yield np.random.RandomState(1223)
    nn.clear_parameters()


def process_param_init(p_init, shape, rng):
    if p_init is True:
        p_init = np.asarray(rng.randn(*shape))
    return p_init


def process_rng(rng):
    if rng is True:
        rng = np.random.RandomState(1223)
    return rng


def insert_if_not_default(d, key, value, default):
    if value != default:
        d[key] = value


def insert_if_not_none(d, key, value):
    if value is not None:
        d[key] = value


def forward_backward_all(*vv):
    y = F.sink(*vv)
    y.forward()
    y.backward()


def check_none_arg(arg, val, none_case):
    if val is None:
        assert arg == none_case
        return
    assert arg == val


@pytest.mark.parametrize("inshape", [(8, 2, 2, 2), (16, 1, 8)])
@pytest.mark.parametrize("n_outmaps", [16, 32])
@pytest.mark.parametrize("base_axis", [1, 2])
@pytest.mark.parametrize("w_init", [None, I.NormalInitializer(), True])
@pytest.mark.parametrize("b_init", [None, I.ConstantInitializer(), True])
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("fix_parameters", [False, True])
@pytest.mark.parametrize("rng", [None, True])
def test_pf_affine_execution(g_rng, inshape, n_outmaps, base_axis, w_init, b_init, with_bias, fix_parameters, rng):

    w_shape = (int(np.prod(inshape[base_axis:])), n_outmaps)
    b_shape = (n_outmaps,)
    w_init = process_param_init(w_init, w_shape, g_rng)
    b_init = process_param_init(b_init, b_shape, g_rng)
    rng = process_rng(rng)

    kw = {}
    insert_if_not_none(kw, 'w_init', w_init)
    insert_if_not_none(kw, 'b_init', b_init)
    insert_if_not_none(kw, 'rng', rng)
    insert_if_not_default(kw, 'base_axis', base_axis, 1)
    insert_if_not_default(kw, 'fix_parameters', fix_parameters, False)
    insert_if_not_default(kw, 'with_bias', with_bias, True)

    x = nn.Variable.from_numpy_array(g_rng.randn(*inshape))

    # Check execution
    y = PF.affine(x, n_outmaps, **kw)
    y.forward()
    y.backward()

    # Check values
    # TODO

    # Check args
    assert y.parent.info.type_name == 'Affine'
    args = y.parent.info.args
    assert args['base_axis'] == base_axis

    # Check created parameters
    assert y.parent.inputs[0] == x
    assert len(y.parent.inputs) == 2 + int(with_bias)
    assert len(nn.get_parameters()) == 1 + int(with_bias)
    w = nn.get_parameters()['affine/W']
    assert w.shape == w_shape
    assert w.need_grad
    assert y.parent.inputs[1].need_grad == (not fix_parameters)
    if isinstance(w_init, np.ndarray):
        assert np.allclose(w_init, w.d)
    if with_bias:
        b = nn.get_parameters()['affine/b']
        assert b.shape == b_shape
        assert b.need_grad
        assert y.parent.inputs[2].need_grad == (not fix_parameters)
        if isinstance(b_init, np.ndarray):
            assert np.allclose(b_init, b.d)


@pytest.mark.parametrize("inshape, outmaps, kernel, pad, stride, dilation, group, base_axis", [
    ((1, 2, 1, 4, 4), 16, (3, 3), None, None, None, 1, 2),
    ((1, 2, 2, 2, 8), 8, (1, 1, 3), (0, 0, 1), (1, 1, 2), (1, 1, 2), 2, 1),
])
@pytest.mark.parametrize("w_init", [None, I.NormalInitializer(), True])
@pytest.mark.parametrize("b_init", [None, I.ConstantInitializer(), True])
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("fix_parameters", [False, True])
@pytest.mark.parametrize("rng", [None, True])
def test_pf_convolution_execution(g_rng, inshape, outmaps, kernel, pad, stride, dilation, group, base_axis, w_init, b_init, with_bias, fix_parameters, rng):

    w_shape = (outmaps, inshape[base_axis] // group,) + kernel
    b_shape = (outmaps,)
    w_init = process_param_init(w_init, w_shape, g_rng)
    b_init = process_param_init(b_init, b_shape, g_rng)
    rng = process_rng(rng)

    x = nn.Variable.from_numpy_array(g_rng.randn(*inshape))
    kw = {}
    insert_if_not_none(kw, 'pad', pad)
    insert_if_not_none(kw, 'stride', stride)
    insert_if_not_none(kw, 'dilation', dilation)
    insert_if_not_none(kw, 'w_init', w_init)
    insert_if_not_none(kw, 'b_init', b_init)
    insert_if_not_none(kw, 'rng', rng)
    insert_if_not_default(kw, 'group', group, 1)
    insert_if_not_default(kw, 'base_axis', base_axis, 1)
    insert_if_not_default(kw, 'fix_parameters', fix_parameters, False)
    insert_if_not_default(kw, 'with_bias', with_bias, True)

    # Check execution
    y = PF.convolution(x, outmaps, kernel, **kw)
    y.forward()
    y.backward()

    # Check values
    # TODO

    # Check args
    assert y.parent.info.type_name == 'Convolution'
    args = y.parent.info.args
    assert args['base_axis'] == base_axis
    assert args['group'] == group
    ndim = len(x.shape) - (base_axis + 1)
    check_none_arg(tuple(args['pad']), pad, (0,) * ndim)
    check_none_arg(tuple(args['stride']), stride, (1,) * ndim)
    check_none_arg(tuple(args['dilation']), dilation, (1,) * ndim)

    # Check created parameters
    assert y.parent.inputs[0] == x
    assert len(y.parent.inputs) == 2 + int(with_bias)
    assert len(nn.get_parameters()) == 1 + int(with_bias)
    w = nn.get_parameters()['conv/W']
    assert w.shape == w_shape
    assert w.need_grad
    assert y.parent.inputs[1].need_grad == (not fix_parameters)
    if isinstance(w_init, np.ndarray):
        assert np.allclose(w_init, w.d)
    if with_bias:
        b = nn.get_parameters()['conv/b']
        assert b.shape == b_shape
        assert b.need_grad
        assert y.parent.inputs[2].need_grad == (not fix_parameters)
        if isinstance(b_init, np.ndarray):
            assert np.allclose(b_init, b.d)


def _get_bn_parameter_shape(inshape, axes):
    '''
    Helper function which gets parameter shape of Batch Normalization.
    '''
    return tuple(size if i in axes else 1 for (i, size) in enumerate(inshape))


@pytest.mark.parametrize("inshape, decay_rate, eps, axes", [
    ((1, 2, 1, 4), 0.9, 1e-5, [3]),
    ((8, 8), 0.99, 1e-3, [1]),
])
@pytest.mark.parametrize('batch_stat, output_stat', [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize('param_init', [None, True])
@pytest.mark.parametrize("fix_parameters", [False, True])
@pytest.mark.parametrize("rng", [None, True])
def test_pf_batch_normalization_execution(
        g_rng, inshape, axes, decay_rate, eps, batch_stat, output_stat,
        param_init, fix_parameters, rng):

    p_shape = _get_bn_parameter_shape(inshape, axes)
    if param_init:
        beta_init = np.ones(p_shape)
        gamma_init = np.ones(p_shape) * 2
        mean_init = np.ones(p_shape) * 0.5
        var_init = np.ones(p_shape) * 1.5
        param_init = dict(
            beta=beta_init,
            gamma=gamma_init,
            mean=mean_init,
            var=var_init)
    rng = process_rng(rng)

    x = nn.Variable.from_numpy_array(g_rng.randn(*inshape))

    kw = {}
    insert_if_not_default(kw, 'axes', axes, [1])
    insert_if_not_default(kw, 'decay_rate', decay_rate, 0.9)
    insert_if_not_default(kw, 'eps', eps, 1e-5)
    insert_if_not_default(kw, 'batch_stat', batch_stat, True)
    insert_if_not_default(kw, 'output_stat', output_stat, False)
    insert_if_not_default(kw, 'fix_parameters', fix_parameters, False)
    insert_if_not_none(kw, 'param_init', param_init)

    # Check creation
    y = PF.batch_normalization(x, **kw)

    # Check parameter values before execution
    h = y[0] if output_stat else y
    _, b, g, m, v = h.parent.inputs
    if param_init:
        assert np.allclose(b.d, beta_init)
        assert np.allclose(g.d, gamma_init)
        assert np.allclose(m.d, mean_init)
        assert np.allclose(v.d, var_init)
    else:
        assert np.allclose(b.d, 0)
        assert np.allclose(g.d, 1)
        assert np.allclose(m.d, 0)
        assert np.allclose(v.d, 1)

    # Check execution
    if output_stat:
        forward_backward_all(*y)
    else:
        y.forward()
        # TODO: Enable when implemented
        if batch_stat:
            y.backward()

    # Check values
    # TODO

    # Check args
    assert h.parent.info.type_name == 'BatchNormalization'
    args = h.parent.info.args
    assert np.isclose(args['decay_rate'], decay_rate)
    assert np.isclose(args['eps'], eps)
    assert args['axes'] == axes
    assert args['batch_stat'] == batch_stat

    # Check created parameters
    assert h.parent.inputs[0] == x
    assert len(h.parent.inputs) == 5
    assert len(nn.get_parameters()) == 2
    assert len(nn.get_parameters(grad_only=False)) == 4
    beta, gamma, mean, var = [nn.get_parameters(grad_only=False)['bn/' + name]
                              for name in ['beta', 'gamma', 'mean', 'var']]
    assert beta.shape == p_shape
    assert gamma.shape == p_shape
    assert mean.shape == p_shape
    assert var.shape == p_shape

    assert beta.need_grad
    assert gamma.need_grad
    assert not mean.need_grad
    assert not var.need_grad

    _, b, g, m, v = h.parent.inputs
    assert b.need_grad == (not fix_parameters)
    assert g.need_grad == (not fix_parameters)
    assert not m.need_grad
    assert not v.need_grad


@pytest.mark.parametrize("inshape, decay_rate, eps, axes", [
    ((1, 2, 1, 4), 0.9, 1e-5, [3]),
    ((8, 8), 0.99, 1e-3, [1]),
])
@pytest.mark.parametrize('batch_stat, output_stat', [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("nonlinearity", ['relu'])
@pytest.mark.parametrize('param_init', [None, True])
@pytest.mark.parametrize("fix_parameters", [False, True])
@pytest.mark.parametrize("with_z", [False, True])
@pytest.mark.parametrize("rng", [None, True])
def test_pf_fused_batch_normalization_execution(
        g_rng, inshape, axes, decay_rate, eps, batch_stat, nonlinearity,
        output_stat, param_init, fix_parameters, with_z, rng):

    p_shape = _get_bn_parameter_shape(inshape, axes)

    if param_init:
        beta_init = np.ones(p_shape)
        gamma_init = np.ones(p_shape) * 2
        mean_init = np.ones(p_shape) * 0.5
        var_init = np.ones(p_shape) * 1.5
        param_init = dict(
            beta=beta_init,
            gamma=gamma_init,
            mean=mean_init,
            var=var_init)
    rng = process_rng(rng)

    x = nn.Variable.from_numpy_array(g_rng.randn(*inshape))
    z = None
    if with_z:
        z = nn.Variable.from_numpy_array(g_rng.randn(*inshape))

    kw = {}
    insert_if_not_none(kw, 'z', z)
    insert_if_not_default(kw, 'axes', axes, [1])
    insert_if_not_default(kw, 'decay_rate', decay_rate, 0.9)
    insert_if_not_default(kw, 'eps', eps, 1e-5)
    insert_if_not_default(kw, 'batch_stat', batch_stat, True)
    insert_if_not_default(kw, 'nonlinearity', nonlinearity, 'relu')
    insert_if_not_default(kw, 'output_stat', output_stat, False)
    insert_if_not_default(kw, 'fix_parameters', fix_parameters, False)
    insert_if_not_none(kw, 'param_init', param_init)

    # Check creation
    y = PF.fused_batch_normalization(x, **kw)

    # Check parameter values before execution
    h = y[0] if output_stat else y
    if with_z:
        _, b, g, m, v, _ = h.parent.inputs
    else:
        _, b, g, m, v = h.parent.inputs
    if param_init:
        assert np.allclose(b.d, beta_init)
        assert np.allclose(g.d, gamma_init)
        assert np.allclose(m.d, mean_init)
        assert np.allclose(v.d, var_init)
    else:
        assert np.allclose(b.d, 0)
        assert np.allclose(g.d, 1)
        assert np.allclose(m.d, 0)
        assert np.allclose(v.d, 1)

    # Check execution
    if output_stat:
        forward_backward_all(*y)
    else:
        y.forward()
        # TODO: Enable when implemented
        if batch_stat:
            y.backward()

    # Check values
    # TODO

    # Check args
    assert h.parent.info.type_name == 'FusedBatchNormalization'
    args = h.parent.info.args
    assert args['axes'] == axes
    assert np.isclose(args['decay_rate'], decay_rate)
    assert np.isclose(args['eps'], eps)
    assert args['batch_stat'] == batch_stat
    assert args['nonlinearity'] == nonlinearity

    # Check created parameters
    assert h.parent.inputs[0] == x
    num_inputs = 5
    if with_z:
        num_inputs = 6
        h.parent.inputs[5] == z
    assert len(h.parent.inputs) == num_inputs
    assert len(nn.get_parameters()) == 2
    assert len(nn.get_parameters(grad_only=False)) == 4
    beta, gamma, mean, var = [nn.get_parameters(grad_only=False)['bn/' + name]
                              for name in ['beta', 'gamma', 'mean', 'var']]
    assert beta.shape == p_shape
    assert gamma.shape == p_shape
    assert mean.shape == p_shape
    assert var.shape == p_shape

    assert beta.need_grad
    assert gamma.need_grad
    assert not mean.need_grad
    assert not var.need_grad

    _, b, g, m, v = h.parent.inputs[:5]
    assert b.need_grad == (not fix_parameters)
    assert g.need_grad == (not fix_parameters)
    assert not m.need_grad
    assert not v.need_grad


@pytest.mark.parametrize("w_shape, dim", [((32, 16, 3, 3), 0),  # convolution
                                          ((16, 1), 1),         # affine
                                          ((16, 32), 1),        # affine
                                          ((8, 8, 16), 2),      # affine
                                          ((8, 4, 16), 1),      # affine
                                          ])
@pytest.mark.parametrize("itr", [1, 2, 3])
@pytest.mark.parametrize("test", [True, False])
@pytest.mark.parametrize("u_init", [None, True])
def test_pf_spectral_norm_execution(g_rng, w_shape, dim, itr, test, u_init):
    # python implementation
    def spectral_norm_numpy(w, dim=0, itr=1, eps=1e-12, test=False, u_init_d=None):
        if test:
            return w
        w_shape = w.shape
        if dim != 0:
            dims_transpose = [dim] + \
                [i for i in range(len(w_shape)) if i != dim]
            w = w.transpose(*dims_transpose)
            w_shape = w.shape
        d0, d1 = w_shape[0], np.prod(w_shape[1:])  # [Out, In]
        w = w.reshape((d0, d1))
        u = u_init_d
        for i in range(itr):
            v = np.dot(w.T, u)
            v = v / np.sqrt(np.sum(v ** 2) + eps)
            u = np.dot(w, v)
            u = u / np.sqrt(np.sum(u ** 2) + eps)
        sigma = np.dot(u.T, np.dot(w, v))
        w_sn = w / sigma
        w_sn = w_sn.reshape(w_shape)
        if dim != 0:
            dims_transpose = [i for i in range(1, dim + 1)] \
                             + [0] + [i for i in range(dim + 1, len(w_shape))]
            w_sn = w_sn.transpose(*dims_transpose)
        return w_sn

    # Setting
    w = nn.Variable.from_numpy_array(g_rng.randn(*w_shape))
    u_init = process_param_init(u_init, [w_shape[dim]], g_rng)

    # Check execution
    w_sn = PF.spectral_norm(w, dim, itr, test=test, u_init=u_init)
    u_init_d = nn.get_parameters(grad_only=False)['spectral-norm/u'].d.copy() \
        if u_init is None else u_init
    if not test:
        w_sn.forward()
        w_sn.backward()
    else:
        w_sn = w

    # Check values
    w_sn_numpy = spectral_norm_numpy(
        w.d, dim, itr, test=test, u_init_d=u_init_d)
    assert np.allclose(w_sn_numpy, w_sn.d, atol=1e-2, rtol=1e-5)

    # Check args (cannot since this is the functions composite)

    # Check created parameters
    assert len(nn.get_parameters(grad_only=False)) == 2
    w_sn, u = [nn.get_parameters(grad_only=False)['spectral-norm/' + name]
               for name in ['W_sn', 'u']]


@pytest.mark.parametrize("inshape , batch_axis", [((4, 3, 8, 8), 0),
                                                  ((16, 1), 0),
                                                  # time-series (T, B, C) or (B, T, C)
                                                  ((3, 32, 4), 0),
                                                  ((10, 4, 16), [0, 1])
                                                  ])
@pytest.mark.parametrize('output_stat', [False, True])
@pytest.mark.parametrize("fix_parameters", [False, True])
@pytest.mark.parametrize('param_init', [None, True])
def test_pf_layer_normalization(g_rng, inshape, batch_axis, output_stat, fix_parameters, param_init):
    from nnabla.normalization_functions import _force_list, _get_axes_excluding

    def ref_layer_normalization(x, beta, gamma, batch_axis, eps, output_stat):
        batch_axis = _force_list(batch_axis)

        axes = tuple(_get_axes_excluding(len(x.shape), batch_axis))

        x_mean = x.mean(axis=axes, keepdims=True)
        x_std = x.std(axis=axes, keepdims=True)

        if output_stat:
            return (x - x_mean) / (x_std + eps) * gamma + beta, x_mean, x_std

        return (x - x_mean) / (x_std + eps) * gamma + beta

    eps = 1e-5

    p_shape = tuple([1 for _ in range(len(inshape))])

    x_npy = g_rng.randn(*inshape)

    if param_init:
        beta_init = np.ones(p_shape)
        gamma_init = np.ones(p_shape) * 2
        param_init = dict(beta=beta_init, gamma=gamma_init)
    else:
        beta_init = np.zeros(p_shape)
        gamma_init = np.ones(p_shape)

    x = nn.Variable.from_numpy_array(x_npy)

    kw = {}
    insert_if_not_default(kw, 'batch_axis', batch_axis, 0)
    insert_if_not_default(kw, 'eps', eps, 1e-5)
    insert_if_not_default(kw, 'output_stat', output_stat, False)
    insert_if_not_default(kw, 'fix_parameters', fix_parameters, False)
    insert_if_not_none(kw, 'param_init', param_init)

    # Check creation
    y = PF.layer_normalization(x, **kw)
    y = _force_list(y)  # just to simplify after execution

    # Check parameter values before execution
    h = y[0]
    b = h.parent.inputs[1]
    g = h.parent.inputs[0].parent.inputs[1]
    assert np.allclose(b.d, beta_init)
    assert np.allclose(g.d, gamma_init)

    # Check execution
    forward_backward_all(*y)

    # Check values
    ref = ref_layer_normalization(
        x_npy, beta_init, gamma_init, batch_axis, eps, output_stat)
    if not output_stat:
        ref = [ref]

    for i in range(len(ref)):
        assert np.allclose(y[i].d, ref[i], atol=1e-2, rtol=1e-5)

    # Check created parameters
    assert len(nn.get_parameters()) == 2
    assert len(nn.get_parameters(grad_only=False)) == 2
    beta, gamma = [nn.get_parameters()['layer_normalization/' + name]
                   for name in ['beta', 'gamma']]
    assert beta.shape == p_shape
    assert gamma.shape == p_shape

    assert beta.need_grad
    assert gamma.need_grad

    b = h.parent.inputs[1]
    g = h.parent.inputs[0].parent.inputs[1]
    assert b.need_grad == (not fix_parameters)
    assert g.need_grad == (not fix_parameters)


@pytest.mark.parametrize("inshape , batch_axis, channel_axis",
                         [((4, 32, 8, 8), 0, 1),  # convolution (NCHW)
                          ((4, 16, 16, 8), 0, 3),  # convolution (NHWC)
                          ((16, 4), 0, 1),  # affine
                          # time-series (T, B, C) or (B, T, C)
                          ((10, 4, 16), [0, 1], 2)
                          ])
@pytest.mark.parametrize('output_stat', [False, True])
@pytest.mark.parametrize("fix_parameters", [False, True])
@pytest.mark.parametrize('param_init', [None, True])
def test_pf_instance_normalization(g_rng, inshape, batch_axis, channel_axis, output_stat, fix_parameters, param_init):
    from nnabla.normalization_functions import _force_list, _get_axes_excluding

    def ref_instance_normalization(x, beta, gamma, channel_axis, batch_axis, eps, output_stat):

        ignore_axes = _force_list(batch_axis) + [channel_axis, ]

        axes = tuple(_get_axes_excluding(len(x.shape), ignore_axes))

        x_mean = x.mean(axis=axes, keepdims=True)
        x_std = x.std(axis=axes, keepdims=True)

        if output_stat:
            return (x - x_mean) / (x_std + eps) * gamma + beta, x_mean, x_std

        return (x - x_mean) / (x_std + eps) * gamma + beta

    eps = 1e-5

    p_shape = tuple([inshape[i] if i == channel_axis else 1
                     for i in range(len(inshape))])

    x_npy = g_rng.randn(*inshape)

    if param_init:
        beta_init = np.ones(p_shape)
        gamma_init = np.ones(p_shape) * 2
        param_init = dict(beta=beta_init, gamma=gamma_init)
    else:
        beta_init = np.zeros(p_shape)
        gamma_init = np.ones(p_shape)

    x = nn.Variable.from_numpy_array(x_npy)

    kw = {}
    insert_if_not_default(kw, 'channel_axis', channel_axis, 1)
    insert_if_not_default(kw, 'batch_axis', batch_axis, 0)
    insert_if_not_default(kw, 'eps', eps, 1e-5)
    insert_if_not_default(kw, 'output_stat', output_stat, False)
    insert_if_not_default(kw, 'fix_parameters', fix_parameters, False)
    insert_if_not_none(kw, 'param_init', param_init)

    # Check creation
    y = PF.instance_normalization(x, **kw)
    y = _force_list(y)  # just to simplify after execution

    # Check parameter values before execution
    h = y[0]
    b = h.parent.inputs[1]
    g = h.parent.inputs[0].parent.inputs[1]
    assert np.allclose(b.d, beta_init)
    assert np.allclose(g.d, gamma_init)

    # Check execution
    forward_backward_all(*y)

    # Check values
    ref = ref_instance_normalization(
        x_npy, beta_init, gamma_init, channel_axis, batch_axis, eps, output_stat)
    if not output_stat:
        ref = [ref]

    for i in range(len(ref)):
        assert np.allclose(y[i].d, ref[i], atol=1e-2, rtol=1e-5)

    # Check created parameters
    assert len(nn.get_parameters()) == 2
    assert len(nn.get_parameters(grad_only=False)) == 2
    beta, gamma = [nn.get_parameters()['instance_normalization/' + name]
                   for name in ['beta', 'gamma']]
    assert beta.shape == p_shape
    assert gamma.shape == p_shape

    assert beta.need_grad
    assert gamma.need_grad

    b = h.parent.inputs[1]
    g = h.parent.inputs[0].parent.inputs[1]
    assert b.need_grad == (not fix_parameters)
    assert g.need_grad == (not fix_parameters)


@pytest.mark.parametrize("num_groups", [2, 4])
@pytest.mark.parametrize("inshape , batch_axis, channel_axis",
                         [((4, 32, 8, 8), 0, 1),  # convolution (NCHW)
                          ((4, 16, 16, 8), 0, 3),  # convolution (NHWC)
                          ((16, 4), 0, 1),  # affine
                          # time-series (T, B, C) or (B, T, C)
                          ((10, 4, 16), [0, 1], 2)
                          ])
@pytest.mark.parametrize('output_stat', [False, True])
@pytest.mark.parametrize("fix_parameters", [False, True])
@pytest.mark.parametrize('param_init', [None, True])
def test_pf_group_normalization(g_rng, num_groups, inshape, batch_axis, channel_axis, output_stat, fix_parameters, param_init):
    from nnabla.normalization_functions import _force_list, _get_axes_excluding

    def ref_group_normalization(x, beta, gamma, num_groups, channel_axis, batch_axis, eps, output_stat):
        cdim = x.shape[channel_axis]

        if cdim % num_groups > 0:
            raise ValueError()

        shape = x.shape[:channel_axis] + (num_groups, int(cdim / num_groups))
        if channel_axis < len(x.shape) - 1:
            shape += x.shape[channel_axis + 1:]

        tmp = x.reshape(shape).copy()

        ignore_axes = _force_list(batch_axis) + [channel_axis, ]

        axes = tuple(_get_axes_excluding(len(shape), ignore_axes))

        x_mean = tmp.mean(axis=axes, keepdims=True)
        x_std = tmp.std(axis=axes, keepdims=True)

        if output_stat:
            return ((tmp - x_mean) / (x_std + eps) * gamma + beta).reshape(x.shape), x_mean, x_std

        return ((tmp - x_mean) / (x_std + eps) * gamma + beta).reshape(x.shape)

    eps = 1e-5

    p_shape = [1 for _ in range(len(inshape) + 1)]
    p_shape[channel_axis] = num_groups
    p_shape[channel_axis + 1] = int(inshape[channel_axis] / num_groups)
    p_shape = tuple(p_shape)

    x_npy = g_rng.randn(*inshape)

    if param_init:
        beta_init = np.ones(p_shape)
        gamma_init = np.ones(p_shape) * 2
        param_init = dict(beta=beta_init, gamma=gamma_init)
    else:
        beta_init = np.zeros(p_shape)
        gamma_init = np.ones(p_shape)

    x = nn.Variable.from_numpy_array(x_npy)

    kw = {}
    insert_if_not_default(kw, 'channel_axis', channel_axis, 1)
    insert_if_not_default(kw, 'batch_axis', batch_axis, 0)
    insert_if_not_default(kw, 'eps', eps, 1e-5)
    insert_if_not_default(kw, 'output_stat', output_stat, False)
    insert_if_not_default(kw, 'fix_parameters', fix_parameters, False)
    insert_if_not_none(kw, 'param_init', param_init)

    # Check creation
    y = PF.group_normalization(x, num_groups, **kw)
    y = _force_list(y)  # just to simplify after execution

    # Check parameter values before execution ( reshape(Add2(Mul2(h, g), b)) )
    h = y[0]
    b = h.parent.inputs[0].parent.inputs[1]
    g = h.parent.inputs[0].parent.inputs[0].parent.inputs[1]
    assert np.allclose(b.d, beta_init)
    assert np.allclose(g.d, gamma_init)

    # Check execution
    forward_backward_all(*y)

    # Check values
    ref = ref_group_normalization(
        x_npy, beta_init, gamma_init, num_groups, channel_axis, batch_axis, eps, output_stat)
    if not output_stat:
        ref = [ref]

    for i in range(len(ref)):
        assert np.allclose(y[i].d, ref[i], atol=1e-2, rtol=1e-5)

    # Check created parameters
    assert len(nn.get_parameters()) == 2
    assert len(nn.get_parameters(grad_only=False)) == 2
    beta, gamma = [nn.get_parameters()['group_normalization/' + name]
                   for name in ['beta', 'gamma']]
    assert beta.shape == p_shape
    assert gamma.shape == p_shape

    assert beta.need_grad
    assert gamma.need_grad

    b = h.parent.inputs[0].parent.inputs[1]
    g = h.parent.inputs[0].parent.inputs[0].parent.inputs[1]
    assert b.need_grad == (not fix_parameters)
    assert g.need_grad == (not fix_parameters)


from nbla_test_utils import list_context
ctxs = list_context('RNN')


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("inshape", [(5, 8, 16), (10, 16, 32)])
@pytest.mark.parametrize('w0_init, w_init, b_init', [
    (None, None, None),
    (I.ConstantInitializer(), I.ConstantInitializer(), I.ConstantInitializer()),
    (True, True, True), ])
@pytest.mark.parametrize("num_layers, nonlinearity, dropout, bidirectional, with_bias", [
    (1, 'tanh', 0.0, False, False),
    (2, 'relu', 0.5, True, True)])
@pytest.mark.parametrize("hidden_size", [5])
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("fix_parameters", [False, True])
@pytest.mark.parametrize("rng", [None, True])
def test_pf_rnn_execution(g_rng, inshape, w0_init, w_init, b_init, num_layers, nonlinearity, dropout, bidirectional, with_bias, hidden_size, training, fix_parameters, rng, ctx, func_name):

    with nn.context_scope(ctx):
        if func_name == "RNN":
            pytest.skip("Not implemented in CPU.")

        num_directions = 2 if bidirectional else 1
        w0_shape = (num_directions, hidden_size, inshape[2] + hidden_size)
        w_shape = (max(1, num_layers-1), num_directions, hidden_size,
                   num_directions * hidden_size + hidden_size)
        b_shape = (num_layers, num_directions, hidden_size)

        w0_init = process_param_init(w0_init, w0_shape, g_rng)
        w_init = process_param_init(w_init, w_shape, g_rng)
        b_init = process_param_init(b_init, b_shape, g_rng)
        rng = process_rng(rng)

        kw = {}
        insert_if_not_none(kw, 'w0_init', w0_init)
        insert_if_not_none(kw, 'w_init', w_init)
        insert_if_not_none(kw, 'b_init', b_init)
        insert_if_not_default(kw, 'num_layers', num_layers, 1)
        insert_if_not_default(kw, 'nonlinearity', nonlinearity, 'tanh')
        insert_if_not_default(kw, 'dropout', dropout, 0.0)
        insert_if_not_default(kw, 'bidirectional', bidirectional, False)
        insert_if_not_default(kw, 'training', training, True)
        insert_if_not_none(kw, 'rng', rng)
        insert_if_not_default(kw, 'with_bias', with_bias, True)
        insert_if_not_default(kw, 'fix_parameters', fix_parameters, False)

        x = nn.Variable.from_numpy_array(g_rng.randn(*inshape))
        h = nn.Variable.from_numpy_array(g_rng.randn(
            *(num_layers, num_directions, inshape[1], hidden_size)))

        # Check execution
        y, hn = PF.rnn(x, h, **kw)
        y.forward()
        if training:
            y.backward()

        # Check values
        # TODO

        # Check args
        assert y.parent.info.type_name == 'RNN'
        args = y.parent.info.args

        # Check created parameters
        assert y.parent.inputs[0] == x
        assert y.parent.inputs[1] == h
        w0 = nn.get_parameters()['rnn/weight_l0']
        assert w0.shape == w0_shape
        assert w0.need_grad
        assert y.parent.inputs[2].need_grad == (not fix_parameters)
        if isinstance(w0_init, np.ndarray):
            assert np.allclose(w0_init, w0.d)
        if num_layers > 1:
            w = nn.get_parameters()['rnn/weight']
            assert w.shape == w_shape
            assert w.need_grad
            assert y.parent.inputs[3].need_grad == (not fix_parameters)
            if isinstance(w_init, np.ndarray):
                assert np.allclose(w_init, w.d)
        if with_bias:
            b = nn.get_parameters()['rnn/bias']
            assert b.shape == b_shape
            assert b.need_grad
            if num_layers > 1:
                assert y.parent.inputs[4].need_grad == (not fix_parameters)
            else:
                assert y.parent.inputs[3].need_grad == (not fix_parameters)
            if isinstance(b_init, np.ndarray):
                assert np.allclose(b_init, b.d)


ctxs = list_context('GRU')


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("inshape", [(5, 8, 16), (10, 16, 32)])
@pytest.mark.parametrize('w0_init, w_init, b_init', [
    (None, None, None),
    (I.ConstantInitializer(), I.ConstantInitializer(), I.ConstantInitializer()),
    (True, True, True), ])
@pytest.mark.parametrize("num_layers, dropout, bidirectional, with_bias", [
    (1, 0.0, False, False),
    (2, 0.5, True, True)])
@pytest.mark.parametrize("hidden_size", [5])
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("fix_parameters", [False, True])
@pytest.mark.parametrize("rng", [None, True])
def test_pf_gru_execution(g_rng, inshape, w0_init, w_init, b_init, num_layers, dropout, bidirectional, with_bias, hidden_size, training, fix_parameters, rng, ctx, func_name):

    with nn.context_scope(ctx):
        if func_name == "GRU":
            pytest.skip("Not implemented in CPU.")

        num_directions = 2 if bidirectional else 1
        w0_shape = (num_directions, 3, hidden_size, inshape[2] + hidden_size)
        w_shape = (max(1, num_layers-1), num_directions, 3,
                   hidden_size, num_directions * hidden_size + hidden_size)
        b_shape = (num_layers, num_directions, 4, hidden_size)

        w0_init = process_param_init(w0_init, w0_shape, g_rng)
        w_init = process_param_init(w_init, w_shape, g_rng)
        b_init = process_param_init(b_init, b_shape, g_rng)
        rng = process_rng(rng)

        kw = {}
        insert_if_not_none(kw, 'w0_init', w0_init)
        insert_if_not_none(kw, 'w_init', w_init)
        insert_if_not_none(kw, 'b_init', b_init)
        insert_if_not_default(kw, 'num_layers', num_layers, 1)
        insert_if_not_default(kw, 'dropout', dropout, 0.0)
        insert_if_not_default(kw, 'bidirectional', bidirectional, False)
        insert_if_not_default(kw, 'training', training, True)
        insert_if_not_none(kw, 'rng', rng)
        insert_if_not_default(kw, 'with_bias', with_bias, True)
        insert_if_not_default(kw, 'fix_parameters', fix_parameters, False)

        x = nn.Variable.from_numpy_array(g_rng.randn(*inshape))
        h = nn.Variable.from_numpy_array(g_rng.randn(
            *(num_layers, num_directions, inshape[1], hidden_size)))

        # Check execution
        y, hn = PF.gru(x, h, **kw)
        y.forward()
        if training:
            y.backward()

        # Check values
        # TODO

        # Check args
        assert y.parent.info.type_name == 'GRU'
        args = y.parent.info.args

        # Check created parameters
        assert y.parent.inputs[0] == x
        assert y.parent.inputs[1] == h
        w0 = nn.get_parameters()['gru/weight_l0']
        assert w0.shape == w0_shape
        assert w0.need_grad
        assert y.parent.inputs[2].need_grad == (not fix_parameters)
        if isinstance(w0_init, np.ndarray):
            assert np.allclose(w0_init, w0.d)
        if num_layers > 1:
            w = nn.get_parameters()['gru/weight']
            assert w.shape == w_shape
            assert w.need_grad
            assert y.parent.inputs[3].need_grad == (not fix_parameters)
            if isinstance(w_init, np.ndarray):
                assert np.allclose(w_init, w.d)
        if with_bias:
            b = nn.get_parameters()['gru/bias']
            assert b.shape == b_shape
            assert b.need_grad
            if num_layers > 1:
                assert y.parent.inputs[4].need_grad == (not fix_parameters)
            else:
                assert y.parent.inputs[3].need_grad == (not fix_parameters)
            if isinstance(b_init, np.ndarray):
                assert np.allclose(b_init, b.d)


ctxs = list_context('LSTM')


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("inshape", [(5, 8, 16), (10, 16, 32)])
@pytest.mark.parametrize('w0_init, w_init, b_init', [
    (None, None, None),
    (I.ConstantInitializer(), I.ConstantInitializer(), I.ConstantInitializer()),
    (True, True, True), ])
@pytest.mark.parametrize("num_layers, dropout, bidirectional, with_bias", [
    (1, 0.0, False, False),
    (2, 0.5, True, True)])
@pytest.mark.parametrize("hidden_size", [5])
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("fix_parameters", [False, True])
@pytest.mark.parametrize("rng", [None, True])
def test_pf_lstm_execution(g_rng, inshape, w0_init, w_init, b_init, num_layers, dropout, bidirectional, with_bias, hidden_size, training, fix_parameters, rng, ctx, func_name):

    with nn.context_scope(ctx):
        if func_name == "LSTM":
            pytest.skip("Not implemented in CPU.")

        num_directions = 2 if bidirectional else 1
        w0_shape = (num_directions, 4, hidden_size, inshape[2] + hidden_size)
        w_shape = (max(1, num_layers-1), num_directions, 4,
                   hidden_size, num_directions * hidden_size + hidden_size)
        b_shape = (num_layers, num_directions, 4, hidden_size)

        w0_init = process_param_init(w0_init, w0_shape, g_rng)
        w_init = process_param_init(w_init, w_shape, g_rng)
        b_init = process_param_init(b_init, b_shape, g_rng)
        rng = process_rng(rng)

        kw = {}
        insert_if_not_none(kw, 'w0_init', w0_init)
        insert_if_not_none(kw, 'w_init', w_init)
        insert_if_not_none(kw, 'b_init', b_init)
        insert_if_not_default(kw, 'num_layers', num_layers, 1)
        insert_if_not_default(kw, 'dropout', dropout, 0.0)
        insert_if_not_default(kw, 'bidirectional', bidirectional, False)
        insert_if_not_default(kw, 'training', training, True)
        insert_if_not_none(kw, 'rng', rng)
        insert_if_not_default(kw, 'with_bias', with_bias, True)
        insert_if_not_default(kw, 'fix_parameters', fix_parameters, False)

        x = nn.Variable.from_numpy_array(g_rng.randn(*inshape))
        h = nn.Variable.from_numpy_array(g_rng.randn(
            *(num_layers, num_directions, inshape[1], hidden_size)))
        c = nn.Variable.from_numpy_array(g_rng.randn(
            *(num_layers, num_directions, inshape[1], hidden_size)))

        # Check execution
        y, hn, cn = PF.lstm(x, h, c, **kw)
        y.forward()
        if training:
            y.backward()

        # Check values
        # TODO

        # Check args
        assert y.parent.info.type_name == 'LSTM'
        args = y.parent.info.args

        # Check created parameters
        assert y.parent.inputs[0] == x
        assert y.parent.inputs[1] == h
        assert y.parent.inputs[2] == c
        w0 = nn.get_parameters()['lstm/weight_l0']
        assert w0.shape == w0_shape
        assert w0.need_grad
        assert y.parent.inputs[3].need_grad == (not fix_parameters)
        if isinstance(w0_init, np.ndarray):
            assert np.allclose(w0_init, w0.d)
        if num_layers > 1:
            w = nn.get_parameters()['lstm/weight']
            assert w.shape == w_shape
            assert w.need_grad
            assert y.parent.inputs[4].need_grad == (not fix_parameters)
            if isinstance(w_init, np.ndarray):
                assert np.allclose(w_init, w.d)
        if with_bias:
            b = nn.get_parameters()['lstm/bias']
            assert b.shape == b_shape
            assert b.need_grad
            if num_layers > 1:
                assert y.parent.inputs[5].need_grad == (not fix_parameters)
            else:
                assert y.parent.inputs[4].need_grad == (not fix_parameters)
            if isinstance(b_init, np.ndarray):
                assert np.allclose(b_init, b.d)


@pytest.mark.parametrize("inshape", [(8, 2, 2, 2), (16, 1, 8)])
@pytest.mark.parametrize("base_axis", [1, 2])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("slope_init", [None, I.NormalInitializer(), True])
@pytest.mark.parametrize("fix_parameters", [False, True])
def test_pf_prelu_execution(g_rng, inshape, base_axis, shared, slope_init, fix_parameters):

    slope_shape = tuple() if shared else (inshape[base_axis],)
    slope_init = process_param_init(slope_init, slope_shape, g_rng)

    kw = {}
    insert_if_not_none(kw, 'slope_init', slope_init)
    insert_if_not_default(kw, 'base_axis', base_axis, 1)
    insert_if_not_default(kw, 'shared', shared, True)
    insert_if_not_default(kw, 'fix_parameters', fix_parameters, False)

    x = nn.Variable.from_numpy_array(g_rng.randn(*inshape))

    # Check execution
    y = PF.prelu(x, **kw)
    y.forward()
    y.backward()

    # Check values
    # TODO

    # Check args
    assert y.parent.info.type_name == 'PReLU'
    args = y.parent.info.args
    assert args['base_axis'] == base_axis

    # Check created parameters
    assert y.parent.inputs[0] == x
    assert len(y.parent.inputs) == 2
    assert len(nn.get_parameters()) == 1
    slope = nn.get_parameters()['prelu/slope']
    assert slope.shape == slope_shape
    assert slope.need_grad
    assert y.parent.inputs[1].need_grad == (not fix_parameters)
    if isinstance(slope_init, np.ndarray):
        assert np.allclose(slope_init, slope.d)


# TODO: Test all parametric functions.
