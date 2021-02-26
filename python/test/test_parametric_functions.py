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
from nnabla.testing import assert_allclose


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
        assert_allclose(w_init, w.d)
    if with_bias:
        b = nn.get_parameters()['affine/b']
        assert b.shape == b_shape
        assert b.need_grad
        assert y.parent.inputs[2].need_grad == (not fix_parameters)
        if isinstance(b_init, np.ndarray):
            assert_allclose(b_init, b.d)


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
        assert_allclose(w_init, w.d)
    if with_bias:
        b = nn.get_parameters()['conv/b']
        assert b.shape == b_shape
        assert b.need_grad
        assert y.parent.inputs[2].need_grad == (not fix_parameters)
        if isinstance(b_init, np.ndarray):
            assert_allclose(b_init, b.d)


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
        assert_allclose(b.d, beta_init)
        assert_allclose(g.d, gamma_init)
        assert_allclose(m.d, mean_init)
        assert_allclose(v.d, var_init)
    else:
        assert_allclose(b.d, 0)
        assert_allclose(g.d, 1)
        assert_allclose(m.d, 0)
        assert_allclose(v.d, 1)

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
        assert_allclose(b.d, beta_init)
        assert_allclose(g.d, gamma_init)
        assert_allclose(m.d, mean_init)
        assert_allclose(v.d, var_init)
    else:
        assert_allclose(b.d, 0)
        assert_allclose(g.d, 1)
        assert_allclose(m.d, 0)
        assert_allclose(v.d, 1)

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


@pytest.mark.parametrize('no_scale, no_bias', [(False, False), (True, True)])
def test_pf_bn_no_scale_bias(no_scale, no_bias):
    x = nn.Variable(shape=(1, 3, 2, 2))
    y = PF.batch_normalization(
        x, batch_stat=True, no_scale=no_scale, no_bias=no_bias)

    params = nn.get_parameters()
    assert len(params) == 2 - int(no_scale) - int(no_bias)


@pytest.mark.parametrize('no_scale, no_bias', [(False, False), (True, True)])
def test_pf_fused_bn_no_scale_bias(no_scale, no_bias):
    x = nn.Variable(shape=(1, 3, 2, 2))
    y = PF.fused_batch_normalization(
        x, batch_stat=True, no_scale=no_scale, no_bias=no_bias)

    params = nn.get_parameters()
    assert len(params) == 2 - int(no_scale) - int(no_bias)


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
    def spectral_norm_numpy(w, dim=0, itr=1, eps=1e-12, u_init_d=None):
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

    w_sn.forward()
    w_sn.backward()

    # Check values
    w_sn_numpy = spectral_norm_numpy(
        w.d, dim, itr, u_init_d=u_init_d)
    assert_allclose(w_sn_numpy, w_sn.d, atol=1e-2, rtol=1e-5)
    if test:
        u = nn.get_parameters(grad_only=False)['spectral-norm/u']
        assert_allclose(u_init_d, u.d)

    # Check args (cannot since this is the functions composite)

    # Check created parameters
    assert len(nn.get_parameters(grad_only=False)) == 1
    u = nn.get_parameters(grad_only=False)['spectral-norm/u']


# util. for normalization tests
def check_normalization_params(f_name, p_shape, gamma_ref, beta_ref, no_scale, no_bias):
    n_params = int(not no_scale) + int(not no_bias)
    assert len(nn.get_parameters()) == n_params
    assert len(nn.get_parameters(grad_only=False)) == n_params

    if not no_scale:
        gamma = nn.get_parameters()['{}/gamma'.format(f_name)]
        assert gamma.shape == p_shape
        assert gamma.need_grad
        assert_allclose(gamma.d, gamma_ref)

    if not no_bias:
        beta = nn.get_parameters()['{}/beta'.format(f_name)]
        assert beta.shape == p_shape
        assert beta.need_grad
        assert_allclose(beta.d, beta_ref)


@pytest.mark.parametrize("inshape , batch_axis", [((4, 3, 8, 8), 0),
                                                  ((16, 1), 0),
                                                  # time-series (T, B, C) or (B, T, C)
                                                  ((3, 32, 4), 0),
                                                  ((10, 4, 16), [0, 1])
                                                  ])
@pytest.mark.parametrize('output_stat', [False, True])
@pytest.mark.parametrize("fix_parameters", [False, True])
@pytest.mark.parametrize('param_init', [None, True])
@pytest.mark.parametrize('no_scale', [False, True])
@pytest.mark.parametrize('no_bias', [False, True])
def test_pf_layer_normalization(g_rng, inshape, batch_axis, output_stat, fix_parameters, param_init, no_scale, no_bias):
    from nnabla.normalization_functions import _force_list, _get_axes_excluding, _apply_affine

    def ref_layer_normalization(x, beta, gamma, batch_axis, eps, output_stat):
        batch_axis = _force_list(batch_axis)

        axes = tuple(_get_axes_excluding(len(x.shape), batch_axis))

        x_mean = x.mean(axis=axes, keepdims=True)
        x_var = x.var(axis=axes, keepdims=True)

        norm = (x - x_mean) / (x_var + eps) ** 0.5

        if no_scale:
            gamma = None

        if no_bias:
            beta = None

        if output_stat:
            return _apply_affine(norm, scale=gamma, bias=beta), x_mean, x_var

        return _apply_affine(norm, scale=gamma, bias=beta)

    eps = 1e-5

    p_shape = list(inshape)
    for baxis in _force_list(batch_axis):
        p_shape[baxis] = 1
    p_shape = tuple(p_shape)

    x_npy = g_rng.randn(*inshape).astype(np.float32)

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
    kw.update({'no_scale': no_scale, 'no_bias': no_bias})

    # Check creation
    y = PF.layer_normalization(x, **kw)
    y = _force_list(y)  # just to simplify later processes

    # Check execution
    forward_backward_all(*y)

    # Check values
    ref = ref_layer_normalization(
        x_npy, beta_init, gamma_init, batch_axis, eps, output_stat)
    if not output_stat:
        ref = [ref]

    for i in range(len(ref)):
        assert_allclose(y[i].d, ref[i], atol=1e-2, rtol=1e-5)

    # Check created parameters
    check_normalization_params(
        "layer_normalization", p_shape, gamma_init, beta_init, no_scale, no_bias)


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
@pytest.mark.parametrize('no_scale', [False, True])
@pytest.mark.parametrize('no_bias', [False, True])
def test_pf_instance_normalization(g_rng, inshape, batch_axis, channel_axis, output_stat,
                                   fix_parameters, param_init, no_scale, no_bias):
    from nnabla.normalization_functions import _force_list, _get_axes_excluding, _apply_affine

    def ref_instance_normalization(x, beta, gamma, channel_axis, batch_axis, eps, output_stat):

        ignore_axes = _force_list(batch_axis) + [channel_axis, ]

        axes = tuple(_get_axes_excluding(len(x.shape), ignore_axes))

        x_mean = x.mean(axis=axes, keepdims=True)
        x_var = x.var(axis=axes, keepdims=True)

        norm = (x - x_mean) / (x_var + eps) ** 0.5

        if no_scale:
            gamma = None

        if no_bias:
            beta = None

        if output_stat:
            return _apply_affine(norm, scale=gamma, bias=beta), x_mean, x_var

        return _apply_affine(norm, scale=gamma, bias=beta)

    eps = 1e-5

    p_shape = [1 for _ in range(len(inshape))]
    p_shape[channel_axis] = inshape[channel_axis]
    p_shape = tuple(p_shape)

    x_npy = g_rng.randn(*inshape).astype(np.float32)

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
    kw.update({'no_scale': no_scale, 'no_bias': no_bias})

    # Check creation
    y = PF.instance_normalization(x, **kw)
    y = _force_list(y)  # just to simplify after execution

    # Check execution
    forward_backward_all(*y)

    # Check values
    ref = ref_instance_normalization(
        x_npy, beta_init, gamma_init, channel_axis, batch_axis, eps, output_stat)
    if not output_stat:
        ref = [ref]

    for i in range(len(ref)):
        assert_allclose(y[i].d, ref[i], atol=1e-2, rtol=1e-5)

    # Check created parameters
    check_normalization_params(
        "instance_normalization", p_shape, gamma_init, beta_init, no_scale, no_bias)


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
@pytest.mark.parametrize('no_scale', [False, True])
@pytest.mark.parametrize('no_bias', [False, True])
def test_pf_group_normalization(g_rng, num_groups, inshape, batch_axis, channel_axis, output_stat,
                                fix_parameters, param_init, no_scale, no_bias):
    from nnabla.normalization_functions import _force_list, _get_axes_excluding, _apply_affine

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
        x_var = tmp.var(axis=axes, keepdims=True)

        norm = (tmp - x_mean) / (x_var + eps) ** 0.5

        if no_scale:
            gamma = None

        if no_bias:
            beta = None

        if output_stat:
            return _apply_affine(norm.reshape(x.shape), scale=gamma, bias=beta), x_mean, x_var

        return _apply_affine(norm.reshape(x.shape), scale=gamma, bias=beta)

    eps = 1e-5

    p_shape = [1 for _ in range(len(inshape))]
    p_shape[channel_axis] = inshape[channel_axis]
    p_shape = tuple(p_shape)

    x_npy = g_rng.randn(*inshape).astype(np.float32)

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
    kw.update({'no_scale': no_scale, 'no_bias': no_bias})

    # Check creation
    y = PF.group_normalization(x, num_groups, **kw)
    y = _force_list(y)  # just to simplify after execution

    # Check execution
    forward_backward_all(*y)

    # Check values
    ref = ref_group_normalization(
        x_npy, beta_init, gamma_init, num_groups, channel_axis, batch_axis, eps, output_stat)
    if not output_stat:
        ref = [ref]

    for i in range(len(ref)):
        assert_allclose(y[i].d, ref[i], atol=1e-2, rtol=1e-5)

    # Check created parameters
    check_normalization_params(
        "group_normalization", p_shape, gamma_init, beta_init, no_scale, no_bias)


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
            assert_allclose(w0_init, w0.d)
        if num_layers > 1:
            w = nn.get_parameters()['rnn/weight']
            assert w.shape == w_shape
            assert w.need_grad
            assert y.parent.inputs[3].need_grad == (not fix_parameters)
            if isinstance(w_init, np.ndarray):
                assert_allclose(w_init, w.d)
        if with_bias:
            b = nn.get_parameters()['rnn/bias']
            assert b.shape == b_shape
            assert b.need_grad
            if num_layers > 1:
                assert y.parent.inputs[4].need_grad == (not fix_parameters)
            else:
                assert y.parent.inputs[3].need_grad == (not fix_parameters)
            if isinstance(b_init, np.ndarray):
                assert_allclose(b_init, b.d)


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
            assert_allclose(w0_init, w0.d)
        if num_layers > 1:
            w = nn.get_parameters()['gru/weight']
            assert w.shape == w_shape
            assert w.need_grad
            assert y.parent.inputs[3].need_grad == (not fix_parameters)
            if isinstance(w_init, np.ndarray):
                assert_allclose(w_init, w.d)
        if with_bias:
            b = nn.get_parameters()['gru/bias']
            assert b.shape == b_shape
            assert b.need_grad
            if num_layers > 1:
                assert y.parent.inputs[4].need_grad == (not fix_parameters)
            else:
                assert y.parent.inputs[3].need_grad == (not fix_parameters)
            if isinstance(b_init, np.ndarray):
                assert_allclose(b_init, b.d)


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
            assert_allclose(w0_init, w0.d)
        if num_layers > 1:
            w = nn.get_parameters()['lstm/weight']
            assert w.shape == w_shape
            assert w.need_grad
            assert y.parent.inputs[4].need_grad == (not fix_parameters)
            if isinstance(w_init, np.ndarray):
                assert_allclose(w_init, w.d)
        if with_bias:
            b = nn.get_parameters()['lstm/bias']
            assert b.shape == b_shape
            assert b.need_grad
            if num_layers > 1:
                assert y.parent.inputs[5].need_grad == (not fix_parameters)
            else:
                assert y.parent.inputs[4].need_grad == (not fix_parameters)
            if isinstance(b_init, np.ndarray):
                assert_allclose(b_init, b.d)


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
        assert_allclose(slope_init, slope.d)


# Quantized parametric functions
@pytest.mark.parametrize("inshape", [(4, 8, 32, 32), (4, 10)])
@pytest.mark.parametrize("q_min, q_max", [(0, 255), (-127, 127), (-128, 127)])
@pytest.mark.parametrize("decay", [0.999])
@pytest.mark.parametrize("x_min_max", [True, False])
@pytest.mark.parametrize("ema", [True, False])
@pytest.mark.parametrize("ste_fine_grained", [True])
@pytest.mark.parametrize("eps", [0.01])
@pytest.mark.parametrize("qr_min_init, qr_max_init",
                         [(None, None),
                          (I.ConstantInitializer(-1.0),
                           I.ConstantInitializer(1.0)),
                          (I.ConstantInitializer(-6.0), I.ConstantInitializer(6.0))])
@pytest.mark.parametrize("fix_parameters", [False, True])
@pytest.mark.parametrize("rng", [None, True])
def test_pf_min_max_quantize_execution(g_rng, inshape, q_min, q_max, decay, x_min_max, ema,
                                       ste_fine_grained, eps,
                                       qr_min_init, qr_max_init,
                                       fix_parameters, rng):
    rng = process_rng(rng)

    x = nn.Variable.from_numpy_array(g_rng.randn(*inshape))

    # Check execution
    y = PF.min_max_quantize(x, q_min, q_max, decay, x_min_max, ema,
                            ste_fine_grained=ste_fine_grained, eps=eps,
                            qr_min_init=qr_min_init, qr_max_init=qr_max_init,
                            fix_parameters=fix_parameters)
    y.forward()
    y.backward()

    # Check values
    # TODO

    # Check args
    assert y.parent.info.type_name == 'MinMaxQuantize'
    args = y.parent.info.args
    assert np.allclose(args['decay'], decay)
    assert args['x_min_max'] == x_min_max
    assert args['ema'] == ema

    # Check created parameters
    assert len(nn.get_parameters(grad_only=False)) == 4
    ema_min = nn.get_parameters(grad_only=False)['min_max_quantize/qr_min']
    ema_max = nn.get_parameters(grad_only=False)['min_max_quantize/qr_max']
    ql_min = nn.get_parameters(grad_only=False)['min_max_quantize/ql_min']
    ql_max = nn.get_parameters(grad_only=False)['min_max_quantize/ql_max']
    assert ema_min.shape == ema_max.shape and ema_max.shape == ql_min.shape and ql_min.shape == ql_max.shape


@pytest.mark.parametrize("inshape", [(8, 2, 2, 2), (16, 1, 8)])
@pytest.mark.parametrize("n_outmaps", [16, 32])
@pytest.mark.parametrize("base_axis", [1, 2])
@pytest.mark.parametrize("w_init", [None, I.NormalInitializer(), True])
@pytest.mark.parametrize("b_init", [None, I.ConstantInitializer(), True])
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("fix_parameters", [False, True])
@pytest.mark.parametrize("quantize_w", [True])
@pytest.mark.parametrize("ql_min_w", [0])
@pytest.mark.parametrize("ql_max_w", [255])
@pytest.mark.parametrize("w_min_max", [True, False])
@pytest.mark.parametrize("ste_fine_grained_w", [True, False])
@pytest.mark.parametrize("quantize_b", [True])
@pytest.mark.parametrize("ql_min_b", [0])
@pytest.mark.parametrize("ql_max_b", [255])
@pytest.mark.parametrize("b_min_max", [True, False])
@pytest.mark.parametrize("ste_fine_grained_b", [True, False])
@pytest.mark.parametrize("rng", [None, True])
def test_pf_min_max_quantized_affine_execution(g_rng, inshape, n_outmaps, base_axis,
                                               w_init, b_init, with_bias, fix_parameters,
                                               quantize_w, ql_min_w, ql_max_w, w_min_max, ste_fine_grained_w,
                                               quantize_b, ql_min_b, ql_max_b, b_min_max, ste_fine_grained_b,
                                               rng):
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
    insert_if_not_default(kw, 'quantize_w', quantize_w, True)
    insert_if_not_default(kw, 'ql_min_w', ql_min_w, True)
    insert_if_not_default(kw, 'ql_max_w', ql_max_w, True)
    insert_if_not_default(kw, 'w_min_max', w_min_max, False)
    insert_if_not_default(kw, 'ste_fine_grained_w', ste_fine_grained_w, True)
    insert_if_not_default(kw, 'quantize_b', quantize_b, True)
    insert_if_not_default(kw, 'ql_min_b', ql_min_b, True)
    insert_if_not_default(kw, 'ql_max_b', ql_max_b, True)
    insert_if_not_default(kw, 'b_min_max', b_min_max, False)
    insert_if_not_default(kw, 'ste_fine_grained_b', ste_fine_grained_b, True)

    x = nn.Variable.from_numpy_array(g_rng.randn(*inshape))

    # Check execution
    y = PF.min_max_quantized_affine(x, n_outmaps, **kw)
    y.forward()
    y.backward()

    # Check values
    # TODO

    # Check args
    assert y.parent.info.type_name == 'Affine'
    args = y.parent.info.args
    assert args['base_axis'] == base_axis
    args_qw = y.parent.inputs[1].parent.info.args
    assert args_qw['x_min_max'] == w_min_max
    if len(y.parent.inputs) == 3:
        args_qb = y.parent.inputs[2].parent.info.args
        assert args_qb['x_min_max'] == b_min_max

    # Check created parameters
    assert y.parent.inputs[0] == x
    assert len(y.parent.inputs) == 2 + int(with_bias)
    assert len(nn.get_parameters()) == 3 + 3 * \
        int(with_bias) if w_min_max or b_min_max else 1 + int(with_bias)
    w = nn.get_parameters()['min_max_quantized_affine/W']
    assert w.shape == w_shape
    assert w.need_grad
    assert y.parent.inputs[1].need_grad == (not fix_parameters)
    if isinstance(w_init, np.ndarray):
        assert np.allclose(w_init, w.d)
    if with_bias:
        b = nn.get_parameters()['min_max_quantized_affine/b']
        assert b.shape == b_shape
        assert b.need_grad
        assert y.parent.inputs[2].need_grad == (not fix_parameters)
        if isinstance(b_init, np.ndarray):
            assert np.allclose(b_init, b.d)
    # quantization-related parameters
    assert len(nn.get_parameters(grad_only=False)) == 6 + int(with_bias) * 6
    for k in nn.get_parameters(grad_only=False).keys():
        assert k in ['min_max_quantized_affine/W',
                     'min_max_quantized_affine/W_q',
                     'min_max_quantized_affine/min_max_quantize_w/min_max_quantize/qr_min',
                     'min_max_quantized_affine/min_max_quantize_w/min_max_quantize/qr_max',
                     'min_max_quantized_affine/min_max_quantize_w/min_max_quantize/ql_min',
                     'min_max_quantized_affine/min_max_quantize_w/min_max_quantize/ql_max',
                     'min_max_quantized_affine/b',
                     'min_max_quantized_affine/b_q',
                     'min_max_quantized_affine/min_max_quantize_b/min_max_quantize/qr_min',
                     'min_max_quantized_affine/min_max_quantize_b/min_max_quantize/qr_max',
                     'min_max_quantized_affine/min_max_quantize_b/min_max_quantize/ql_min',
                     'min_max_quantized_affine/min_max_quantize_b/min_max_quantize/ql_max']


@pytest.mark.parametrize("inshape, outmaps, kernel, pad, stride, dilation, group, base_axis", [
    ((1, 2, 1, 4, 4), 16, (3, 3), None, None, None, 1, 2),
    ((1, 2, 2, 2, 8), 8, (1, 1, 3), (0, 0, 1), (1, 1, 2), (1, 1, 2), 2, 1),
])
@pytest.mark.parametrize("w_init", [None, I.NormalInitializer(), True])
@pytest.mark.parametrize("b_init", [None, I.ConstantInitializer(), True])
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("fix_parameters", [False, True])
@pytest.mark.parametrize("quantize_w", [True])
@pytest.mark.parametrize("ql_min_w", [0])
@pytest.mark.parametrize("ql_max_w", [255])
@pytest.mark.parametrize("w_min_max", [True, False])
@pytest.mark.parametrize("ste_fine_grained_w", [True, False])
@pytest.mark.parametrize("quantize_b", [True])
@pytest.mark.parametrize("ql_min_b", [0])
@pytest.mark.parametrize("ql_max_b", [255])
@pytest.mark.parametrize("b_min_max", [True, False])
@pytest.mark.parametrize("ste_fine_grained_b", [True, False])
@pytest.mark.parametrize("rng", [None, True])
def test_pf_min_max_quantized_convolution_execution(g_rng, inshape, outmaps,
                                                    kernel, pad, stride, dilation, group, base_axis,
                                                    w_init, b_init, with_bias, fix_parameters,
                                                    quantize_w, ql_min_w, ql_max_w, w_min_max, ste_fine_grained_w,
                                                    quantize_b, ql_min_b, ql_max_b, b_min_max, ste_fine_grained_b,
                                                    rng):
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
    insert_if_not_default(kw, 'quantize_w', quantize_w, True)
    insert_if_not_default(kw, 'ql_min_w', ql_min_w, True)
    insert_if_not_default(kw, 'ql_max_w', ql_max_w, True)
    insert_if_not_default(kw, 'w_min_max', w_min_max, False)
    insert_if_not_default(kw, 'ste_fine_grained_w', ste_fine_grained_w, True)
    insert_if_not_default(kw, 'quantize_b', quantize_b, True)
    insert_if_not_default(kw, 'ql_min_b', ql_min_b, True)
    insert_if_not_default(kw, 'ql_max_b', ql_max_b, True)
    insert_if_not_default(kw, 'b_min_max', b_min_max, False)
    insert_if_not_default(kw, 'ste_fine_grained_b', ste_fine_grained_b, True)

    # Check execution
    y = PF.min_max_quantized_convolution(x, outmaps, kernel, **kw)
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
    args_qw = y.parent.inputs[1].parent.info.args
    assert args_qw['x_min_max'] == w_min_max
    if len(y.parent.inputs) == 3:
        args_qb = y.parent.inputs[2].parent.info.args
        assert args_qb['x_min_max'] == b_min_max

    # Check created parameters
    assert y.parent.inputs[0] == x
    assert len(y.parent.inputs) == 2 + int(with_bias)
    assert len(nn.get_parameters()) == 3 + 3 * \
        int(with_bias) if w_min_max or b_min_max else 1 + int(with_bias)
    w = nn.get_parameters()['min_max_quantized_conv/W']
    assert w.shape == w_shape
    assert w.need_grad
    assert y.parent.inputs[1].need_grad == (not fix_parameters)
    if isinstance(w_init, np.ndarray):
        assert np.allclose(w_init, w.d)
    if with_bias:
        b = nn.get_parameters()['min_max_quantized_conv/b']
        assert b.shape == b_shape
        assert b.need_grad
        assert y.parent.inputs[2].need_grad == (not fix_parameters)
        if isinstance(b_init, np.ndarray):
            assert np.allclose(b_init, b.d)
    # quantization-related parameters
    assert len(nn.get_parameters(grad_only=False)) == 6 + int(with_bias) * 6
    for k in nn.get_parameters(grad_only=False).keys():
        assert k in ['min_max_quantized_conv/W',
                     'min_max_quantized_conv/W_q',
                     'min_max_quantized_conv/min_max_quantize_w/min_max_quantize/qr_min',
                     'min_max_quantized_conv/min_max_quantize_w/min_max_quantize/qr_max',
                     'min_max_quantized_conv/min_max_quantize_w/min_max_quantize/ql_min',
                     'min_max_quantized_conv/min_max_quantize_w/min_max_quantize/ql_max',
                     'min_max_quantized_conv/b',
                     'min_max_quantized_conv/b_q',
                     'min_max_quantized_conv/min_max_quantize_b/min_max_quantize/qr_min',
                     'min_max_quantized_conv/min_max_quantize_b/min_max_quantize/qr_max',
                     'min_max_quantized_conv/min_max_quantize_b/min_max_quantize/ql_min',
                     'min_max_quantized_conv/min_max_quantize_b/min_max_quantize/ql_max']


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("src_len, tgt_len, batch_size", [
    (2, 3, 2)])
@pytest.mark.parametrize("q_input_dim, k_input_dim, v_input_dim, k_embed_dim, v_embed_dim, out_dim, num_heads, dropout", [
    (16, 16, 16, 12, 12, 12, 6, 0.0),
    (16, 15, 14, 12, 24, 24, 12, 0.0)])
@pytest.mark.parametrize("with_bias", [True, False])
@pytest.mark.parametrize("fix_parameters", [True, False])
@pytest.mark.parametrize("add_attn_bias", [True, False])
@pytest.mark.parametrize("rng", [None, True])
@pytest.mark.parametrize('param_init', [None, True])
def test_pf_multi_head_attention_execution(g_rng, src_len, tgt_len, batch_size, q_input_dim, k_input_dim, v_input_dim, k_embed_dim, v_embed_dim, out_dim, num_heads, dropout, rng, with_bias, add_attn_bias, fix_parameters, param_init, ctx, func_name):

    q_shape = (q_input_dim, k_embed_dim)
    k_shape = (k_input_dim, k_embed_dim)
    v_shape = (v_input_dim, v_embed_dim)
    o_shape = (v_embed_dim, out_dim)

    q_weight = process_param_init(I.NormalInitializer(), q_shape, g_rng)
    k_weight = process_param_init(I.NormalInitializer(), k_shape, g_rng)
    v_weight = process_param_init(I.NormalInitializer(), v_shape, g_rng)
    out_weight = process_param_init(I.NormalInitializer(), o_shape, g_rng)

    param_init = dict(
        q_weight=q_weight,
        k_weight=k_weight,
        v_weight=v_weight,
        out_weight=out_weight)

    if with_bias:
        qb_shape = (k_embed_dim, )
        kb_shape = (k_embed_dim, )
        vb_shape = (v_embed_dim, )
        ob_shape = (out_dim, )
        q_bias = process_param_init(I.ConstantInitializer(), qb_shape, g_rng)
        k_bias = process_param_init(I.ConstantInitializer(), kb_shape, g_rng)
        v_bias = process_param_init(I.ConstantInitializer(), vb_shape, g_rng)
        out_bias = process_param_init(I.ConstantInitializer(), ob_shape, g_rng)

        param_init['q_bias'] = q_bias
        param_init['k_bias'] = k_bias
        param_init['v_bias'] = v_bias
        param_init['out_bias'] = out_bias

    if add_attn_bias:
        attnk_shape = (1, 1, k_embed_dim)
        attnv_shape = (1, 1, v_embed_dim)
        attn_bias_k = process_param_init(
            I.NormalInitializer(), attnk_shape, g_rng)
        attn_bias_v = process_param_init(
            I.NormalInitializer(), attnv_shape, g_rng)

        param_init['attn_bias_k'] = attn_bias_k
        param_init['attn_bias_v'] = attn_bias_v

    rng = process_rng(rng)

    kw = {}
    insert_if_not_none(kw, 'num_heads', num_heads)
    insert_if_not_default(kw, 'dropout', dropout, 0.0)
    insert_if_not_none(kw, 'k_embed_dim', k_embed_dim)
    insert_if_not_none(kw, 'v_embed_dim', v_embed_dim)
    insert_if_not_none(kw, 'out_dim', out_dim)
    insert_if_not_none(kw, 'rng', rng)
    insert_if_not_default(kw, 'with_bias', with_bias, True)
    insert_if_not_default(kw, 'add_attn_bias', add_attn_bias, False)
    insert_if_not_default(kw, 'fix_parameters', fix_parameters, False)
    insert_if_not_none(kw, 'param_init', param_init)

    q = nn.Variable.from_numpy_array(
        g_rng.randn(tgt_len, batch_size, q_input_dim).astype(np.float32), need_grad=True)
    k = nn.Variable.from_numpy_array(
        g_rng.randn(src_len, batch_size, k_input_dim).astype(np.float32), need_grad=True)
    v = nn.Variable.from_numpy_array(
        g_rng.randn(src_len, batch_size, v_input_dim).astype(np.float32), need_grad=True)

    # Check execution
    y, w = PF.multi_head_attention(q, k, v, **kw)
    y.forward()
    y.backward()

    if with_bias:
        if add_attn_bias:
            assert len(nn.get_parameters()) == 10
        else:
            assert len(nn.get_parameters()) == 8
    else:
        if add_attn_bias:
            assert len(nn.get_parameters()) == 6
        else:
            assert len(nn.get_parameters()) == 4

    if with_bias:
        if add_attn_bias:
            qw, kw, vw, ow, qb, kb, vb, ob, abk, abv = [nn.get_parameters(grad_only=False)['multi_head_attention/' + name] for name in [
                                                                          'q_weight', 'k_weight', 'v_weight', 'out_weight', 'q_bias', 'k_bias', 'v_bias', 'out_bias', 'attn_bias_k', 'attn_bias_v']]
        else:
            qw, kw, vw, ow, qb, kb, vb, ob = [nn.get_parameters(grad_only=False)['multi_head_attention/' + name] for name in [
                                                                'q_weight', 'k_weight', 'v_weight', 'out_weight', 'q_bias', 'k_bias', 'v_bias', 'out_bias']]
    else:
        if add_attn_bias:
            qw, kw, vw, ow, abk, abv = [nn.get_parameters(grad_only=False)['multi_head_attention/' + name] for name in [
                                                          'q_weight', 'k_weight', 'v_weight', 'out_weight', 'attn_bias_k', 'attn_bias_v']]
        else:
            qw, kw, vw, ow = [nn.get_parameters(grad_only=False)[
                                                'multi_head_attention/' + name] for name in ['q_weight', 'k_weight', 'v_weight', 'out_weight']]

    assert qw.shape == q_shape
    assert kw.shape == k_shape
    assert vw.shape == v_shape
    assert ow.shape == o_shape
    assert qw.need_grad
    assert kw.need_grad
    assert vw.need_grad
    assert ow.need_grad

    if with_bias:
        assert qb.shape == qb_shape
        assert kb.shape == kb_shape
        assert vb.shape == vb_shape
        assert ob.shape == ob_shape
        assert qb.need_grad
        assert kb.need_grad
        assert vb.need_grad
        assert ob.need_grad

    if add_attn_bias:
        assert abk.shape == attnk_shape
        assert abv.shape == attnv_shape
        assert abk.need_grad
        assert abv.need_grad


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("src_len, tgt_len, batch_size", [
    (2, 3, 2)])
@pytest.mark.parametrize("embed_dim, num_heads, dropout, num_encoder_layers, num_decoder_layers", [
    (12, 6, 0.0, 6, 6),
    (24, 12, 0.0, 6, 3)])
@pytest.mark.parametrize("fix_parameters", [True, False])
@pytest.mark.parametrize("rng", [None, True])
@pytest.mark.parametrize("add_attn_bias", [True, False])
def test_pf_transformer_execution(g_rng, src_len, tgt_len, batch_size, embed_dim, num_heads, dropout, rng, add_attn_bias, num_encoder_layers, num_decoder_layers, fix_parameters, ctx, func_name):

    rng = process_rng(rng)

    kw = {}
    insert_if_not_none(kw, 'embed_dim', embed_dim)
    insert_if_not_none(kw, 'num_heads', num_heads)
    insert_if_not_none(kw, 'num_encoder_layers', num_encoder_layers)
    insert_if_not_none(kw, 'num_decoder_layers', num_decoder_layers)
    insert_if_not_default(kw, 'dropout', dropout, 0.0)
    insert_if_not_none(kw, 'rng', rng)
    insert_if_not_default(kw, 'add_attn_bias', add_attn_bias, False)
    insert_if_not_default(kw, 'fix_parameters', fix_parameters, False)

    src = nn.Variable.from_numpy_array(
        g_rng.randn(src_len, batch_size, embed_dim).astype(np.float32), need_grad=True)
    tgt = nn.Variable.from_numpy_array(
        g_rng.randn(tgt_len, batch_size, embed_dim).astype(np.float32), need_grad=True)

    # Check execution
    y = PF.transformer(src, tgt, **kw)
    y.forward()
    y.backward()

    if add_attn_bias:
        assert len(nn.get_parameters()) == 18 * \
                   num_encoder_layers + 30 * num_decoder_layers
    else:
        assert len(nn.get_parameters()) == 16 * \
                   num_encoder_layers + 26 * num_decoder_layers


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("src_len, batch_size", [
    (2, 3)])
@pytest.mark.parametrize("embed_dim, num_heads, dropout, dim_feedforward", [
    (24, 12, 0.0, 64)])
@pytest.mark.parametrize("fix_parameters", [True, False])
@pytest.mark.parametrize("rng", [None, True])
@pytest.mark.parametrize("add_attn_bias", [True, False])
def test_pf_transformer_encode_execution(g_rng, src_len, batch_size, embed_dim, num_heads, dropout, rng, add_attn_bias, dim_feedforward, fix_parameters, ctx, func_name):

    rng = process_rng(rng)

    kw = {}
    insert_if_not_none(kw, 'dim_feedforward', dim_feedforward)
    insert_if_not_default(kw, 'dropout', dropout, 0.0)
    insert_if_not_none(kw, 'rng', rng)
    insert_if_not_default(kw, 'add_attn_bias', add_attn_bias, False)
    insert_if_not_default(kw, 'fix_parameters', fix_parameters, False)

    src = nn.Variable.from_numpy_array(
        g_rng.randn(src_len, batch_size, embed_dim).astype(np.float32), need_grad=True)

    # Check execution
    y = PF.transformer_encode(src, embed_dim, num_heads, **kw)
    y.forward()
    y.backward()

    if add_attn_bias:
        assert len(nn.get_parameters()) == 18
    else:
        assert len(nn.get_parameters()) == 16


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("tgt_len, batch_size", [
    (2, 3)])
@pytest.mark.parametrize("embed_dim, num_heads, dropout, dim_feedforward", [
    (24, 12, 0.0, 64)])
@pytest.mark.parametrize("fix_parameters", [True, False])
@pytest.mark.parametrize("rng", [None, True])
@pytest.mark.parametrize("add_attn_bias", [True, False])
def test_pf_transformer_decode_execution(g_rng, tgt_len, batch_size, embed_dim, num_heads, dropout, rng, add_attn_bias, dim_feedforward, fix_parameters, ctx, func_name):

    rng = process_rng(rng)

    kw = {}
    insert_if_not_none(kw, 'dim_feedforward', dim_feedforward)
    insert_if_not_default(kw, 'dropout', dropout, 0.0)
    insert_if_not_none(kw, 'rng', rng)
    insert_if_not_default(kw, 'add_attn_bias', add_attn_bias, False)
    insert_if_not_default(kw, 'fix_parameters', fix_parameters, False)

    tgt = nn.Variable.from_numpy_array(
        g_rng.randn(tgt_len, batch_size, embed_dim).astype(np.float32), need_grad=True)
    memory = nn.Variable.from_numpy_array(
        g_rng.randn(tgt_len, batch_size, embed_dim).astype(np.float32), need_grad=True)

    # Check execution
    y = PF.transformer_decode(tgt, memory, embed_dim, num_heads, **kw)
    y.forward()
    y.backward()

    if add_attn_bias:
        assert len(nn.get_parameters()) == 30
    else:
        assert len(nn.get_parameters()) == 26


@pytest.mark.parametrize("func", ["conv", "affine"])
def test_pf_weight_norm_execution(g_rng, func):
    # python implementation
    def ref_weight_normalization(v, g, dim, eps=1e-12):
        axis = tuple([i for i in range(len(v.shape)) if i != dim])
        v_norm = np.sqrt(np.sum(v ** 2, axis=axis, keepdims=True) + eps)

        return g * v / v_norm

    dim = {"conv": 0, "affine": 1}[func]

    def wn_clbk(v): return PF.weight_normalization(v, dim=dim)

    x = nn.Variable.from_numpy_array(g_rng.randn(2, 4, 5, 5))
    if func == "conv":
        # assume channle first
        y = PF.convolution(x, 8, (3, 3), apply_w=wn_clbk)
    elif func == "affine":
        y = PF.affine(x, 8, apply_w=wn_clbk)
    else:
        raise ValueError("unexpected function name {}".format(func))

    # Setting
    y.forward()
    y.backward()

    params = nn.get_parameters()
    assert len(params) == 3  # w, b, g

    # Check values
    v = params["{}/W".format(func)]
    w = y.parent.inputs[1]

    v_np = v.d
    w_np = ref_weight_normalization(v_np, 1, dim)

    assert_allclose(w.d, w_np, atol=1e-2, rtol=1e-5)


@pytest.mark.parametrize("inshape, kernel, out_channels, pad, stride, dilation, group, deformable_group, base_axis", [
    ((2, 4, 8, 8), (3, 2), 4, None, None, None, 1, 2, 1),
    ((2, 4, 6, 6), (3, 2), 4, (0, 0), (1, 1), (1, 1), 2, 2, 1),
    ((2, 2, 5, 7), (3, 3), 2, None, None, None, 1, 1, 1),
    ((2, 2, 5, 7), (3, 3), 2, (1, 1), (1, 2), (2, 1), 1, 2, 1),
    ((2, 2, 5, 7), (3, 3), 2, (1, 1), (1, 2), (2, 1), 2, 1, 1),
 ])
@pytest.mark.parametrize("w_init", [None, I.NormalInitializer(), True])
@pytest.mark.parametrize("b_init", [None, I.ConstantInitializer(), True])
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("fix_parameters", [False, True])
@pytest.mark.parametrize("rng", [None, True])
def test_pf_deformable_convolution_2d_execution(g_rng, inshape, kernel, out_channels, pad, stride, dilation, base_axis, w_init, b_init, group,
                                                deformable_group, with_bias, fix_parameters, rng):
    import platform
    if platform.machine().startswith("arm"):
        pytest.skip('Skip the arm platform temporarily.')

    w_shape = (out_channels, inshape[base_axis] // group,) + kernel
    b_shape = (out_channels,)
    w_init = process_param_init(w_init, w_shape, g_rng)
    b_init = process_param_init(b_init, b_shape, g_rng)
    rng = process_rng(rng)

    x = nn.Variable.from_numpy_array(np.random.rand(*inshape))

    offset_channels = 2 * deformable_group * kernel[0] * kernel[1]
    offset_shape = inshape[0:base_axis] + \
        (offset_channels,) + inshape[base_axis + 1:]

    rng_off_mask = np.random.RandomState(1223)
    offset = (3.8 * rng_off_mask.rand(*offset_shape).astype(np.float32)) - 1.9
    offset += np.logical_or(np.abs(offset - np.floor(offset)) < 0.1,
                            np.abs(offset - np.ceil(offset)) < 0.1).astype(np.int)*0.5
    offset = nn.Variable.from_numpy_array(offset)

    mask_shape = inshape[0:base_axis] + \
        (deformable_group * kernel[0] * kernel[1],) + inshape[base_axis + 1:]
    mask = rng_off_mask.rand(*mask_shape).astype(np.float32)
    mask = nn.Variable.from_numpy_array(mask)

    kw = {}
    insert_if_not_none(kw, 'pad', pad)
    insert_if_not_none(kw, 'stride', stride)
    insert_if_not_none(kw, 'dilation', dilation)
    insert_if_not_none(kw, 'w_init', w_init)
    insert_if_not_none(kw, 'b_init', b_init)
    insert_if_not_none(kw, 'rng', rng)
    insert_if_not_default(kw, 'group', group, 1)
    insert_if_not_default(kw, 'deformable_group', deformable_group, 1)
    insert_if_not_default(kw, 'base_axis', base_axis, 1)
    insert_if_not_default(kw, 'fix_parameters', fix_parameters, False)
    insert_if_not_default(kw, 'with_bias', with_bias, True)

    # Check execution
    y = PF.deformable_convolution(
        x, out_channels, kernel, offset, mask, **kw)
    y.forward()
    y.backward()

    # Check values
    # Tested in test_forward_backward_2d in test_deformabl_convolution.py

    # Check args
    assert y.parent.info.type_name == 'DeformableConvolution'
    args = y.parent.info.args
    assert args['base_axis'] == base_axis
    assert args['group'] == group
    assert args['deformable_group'] == deformable_group
    ndim = len(x.shape) - (base_axis + 1)
    check_none_arg(tuple(args['pad']), pad, (0,) * ndim)
    check_none_arg(tuple(args['stride']), stride, (1,) * ndim)
    check_none_arg(tuple(args['dilation']), dilation, (1,) * ndim)

    # Check created parameters
    assert y.parent.inputs[0] == x
    assert len(y.parent.inputs) == 4 + int(with_bias)
    assert len(nn.get_parameters()) == 1 + int(with_bias)
    w = nn.get_parameters()['deformable_conv/W']
    assert w.shape == w_shape
    assert w.need_grad
    assert y.parent.inputs[1].need_grad == (not fix_parameters)
    if isinstance(w_init, np.ndarray):
        assert_allclose(w_init, w.d)
    assert y.parent.inputs[2] == offset
    assert y.parent.inputs[3] == mask
    if with_bias:
        b = nn.get_parameters()['deformable_conv/b']
        assert b.shape == b_shape
        assert b.need_grad
        assert y.parent.inputs[4].need_grad == (not fix_parameters)
        if isinstance(b_init, np.ndarray):
            assert_allclose(b_init, b.d)

# TODO: Test all parametric functions.
