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
        p_init = rng.randn(*shape)
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


@pytest.mark.parametrize("inshape, decay_rate, eps", [
    ((1, 2, 1, 4), 0.9, 1e-5),
    ((8, 8), 0.99, 1e-3),
])
@pytest.mark.parametrize('batch_stat, output_stat', [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize('param_init', [None, True])
@pytest.mark.parametrize("fix_parameters", [False, True])
@pytest.mark.parametrize("rng", [None, True])
def test_pf_batch_normalization_execution(g_rng, inshape, decay_rate, eps, batch_stat, output_stat, param_init, fix_parameters, rng):

    axis = 1  # Assume axes=[1]
    p_shape = [1] * len(inshape)
    p_shape[axis] = inshape[axis]
    p_shape = tuple(p_shape)

    if param_init:
        beta_init = np.ones(p_shape) * 1
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


# TODO: Test all parametric functions.
