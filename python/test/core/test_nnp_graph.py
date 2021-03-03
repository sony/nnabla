# Copyright (c) 2020 Sony Corporation. All Rights Reserved.
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
from nnabla.testing import assert_allclose

from helper import (get_saved_test_model, forward_variable_and_check_equal,
                    forward_variable, assert_topology, dump_network_topology)
import legacy_nnp_graph
from nnabla.utils import nnp_graph


def flat_module(x):
    def unit(i, prefix):
        c1 = PF.convolution(i, 4, (3, 3), pad=(1, 1), name=prefix + '-c1')
        c2 = PF.convolution(F.relu(c1), 4,
                            (3, 3), pad=(1, 1), name=prefix + '-c2')
        c = F.add2(c2, c1, inplace=True)
        return c
    c = unit(x, 'c1')
    c2 = unit(c, 'c2')
    y = PF.affine(c2, 5, name='fc')
    return y


def mlp_module(x0, x1):
    h1_0 = PF.affine(x0, 100, name='affine1_0')
    h1_1 = PF.affine(x1, 100, name='affine1_1')
    h1 = F.tanh(h1_0 + h1_1)
    h2 = F.tanh(PF.affine(h1, 50, name='affine2'))
    y0 = PF.affine(h2, 10, name='affiney_0')
    y1 = PF.affine(h2, 10, name='affiney_1')
    return y0, y1


def nnp_check(nnp, side, callback=None):
    yield len(nnp.get_network_names())
    for network_name in sorted(nnp.get_network_names()):
        yield network_name
        network = nnp.get_network(
            network_name, batch_size=32, callback=callback)
        for k in sorted(network.variables.keys()):
            print('variables', side, k, '{}, {}'.format(
                network.variables[k].shape, network.variables[k].d.shape))
            yield (k, network.variables[k])

        for k in sorted(network.inputs.keys()):
            print('inputs', side, k, '{}, {}'.format(
                network.inputs[k].shape, network.inputs[k].d.shape))
            yield network.inputs[k]

        for k in sorted(network.outputs.keys()):
            print('outputs', side, k, '{}, {}'.format(
                network.outputs[k].shape, network.outputs[k].d.shape))
            yield network.outputs[k]
        yield network


def compare_nn_variable_metadata(ref_v, v):
    assert ref_v.d.shape == v.d.shape
    assert ref_v.need_grad == v.need_grad


def compare_nn_variable_with_name(ref_v, v):
    ref_v_name, ref_var = ref_v
    v_name, var = v
    assert ref_v_name == v_name
    assert ref_var.d.shape == var.d.shape
    assert ref_var.need_grad == var.need_grad
    if ref_v_name[-2:] in ['/b', '/W']:
        print("compare {} <--> {}".format(ref_var.d.shape, var.d.shape))
        assert np.allclose(ref_var.d, var.d)


def verify_equivalence(ref_v, v):
    if isinstance(ref_v, legacy_nnp_graph.NnpNetwork):
        ref_inputs = list(ref_v.inputs.values())
        inputs = list(v.inputs.values())

        ref_outputs = list(ref_v.outputs.values())
        outputs = list(v.outputs.values())
        for ref_d, d in zip(forward_variable(ref_inputs, ref_outputs, 'left'),
                            forward_variable(inputs, outputs, 'right')):
            assert_allclose(ref_d, d)
        assert_topology(ref_outputs, outputs)

    elif isinstance(ref_v, nn.Variable):
        compare_nn_variable_metadata(ref_v, v)
    elif isinstance(ref_v, tuple):
        compare_nn_variable_with_name(ref_v, v)
    else:
        print("compare {} <==> {}".format(ref_v, v))
        assert ref_v == v


def assert_parameter_scope_empty():
    params = nn.get_parameters()
    assert len(params) == 0


def assert_parameters_equal(ref_params, params):
    def iter_all_param(parameters):
        for k in sorted(parameters.keys()):
            param_dict = parameters[k]
            yield len(param_dict)
            for param_name in sorted(param_dict.keys()):
                yield param_name, param_dict[param_name]

    for a, b in zip(iter_all_param(ref_params),
                    iter_all_param(params)):
        if isinstance(a, tuple):
            ref_n, ref_v = a
            n, v = b
            assert ref_n == n
            assert ref_v.need_grad == v.need_grad
            assert ref_v.d.shape == v.d.shape
            assert_allclose(ref_v.d, v.d)
        else:
            print("{} <--> {}".format(a, b))
            assert a == b


# MODULES = [
#     (flat_module, [(4, 3, 32, 32)]),
#     (mlp_module, [(16, 100), (16, 100)])
# ]

MODULES = [
    (flat_module, [('Convolution_in', (4, 3, 32, 32))]),
    (mlp_module, [('Affine_in', (16, 100)),
                  ('Affine_in_1', (16, 100))])
]


@pytest.mark.parametrize("module", MODULES)
def test_nnp_graph_simple_load(module):
    '''This test tests whether equivalency between new or old implementation of nnp_graph
    '''
    with get_saved_test_model(module) as nnp_file:
        ref_nnp = legacy_nnp_graph.NnpLoader(nnp_file)
        nnp = nnp_graph.NnpLoader(nnp_file)
        for ref_v, v in zip(nnp_check(ref_nnp, 'left'), nnp_check(nnp, 'right')):
            verify_equivalence(ref_v, v)


# - Test parameter_scope
# - Test batch_size
# - Test NetworkPass
#     - on_function_pass_by_name
#     - on_function_pass_by_type
#     - on_generate_variable
#     - on_generate_function_by_name
#     - on_generate_function_by_type
#     - remove_and_rewire
#     - drop_function
#     - fix_parameters
#     - use_up_to
#     - set_variable
#     - fix_parameters
#     - force_average_pooling_global
#     - check_average_pooling_global
#     - set_batch_normalization_batch_stat_all


@pytest.mark.parametrize("module", MODULES)
def test_networkpass_set_variable(module):
    '''This test tests whether equivalency between new or old implementation of nnp_graph
    '''
    _, inputs = module
    verbose = 1
    callback = nnp_graph.NnpNetworkPass(verbose)
    ref_callback = legacy_nnp_graph.NnpNetworkPass(verbose)
    for inp_name, inp_shape in inputs:
        inp_shape = (1, *inp_shape[1:])  # change shape
        callback.set_variable(inp_name, nn.Variable(inp_shape))
        ref_callback.set_variable(inp_name, nn.Variable(inp_shape))
    with get_saved_test_model(module) as nnp_file:
        ref_nnp = legacy_nnp_graph.NnpLoader(nnp_file)
        nnp = nnp_graph.NnpLoader(nnp_file)
        for ref_v, v in zip(nnp_check(ref_nnp, 'left', ref_callback),
                            nnp_check(nnp, 'right', callback)):
            verify_equivalence(ref_v, v)


@pytest.mark.parametrize("module", MODULES)
def test_networkpass_on_generate_function(module):
    '''This test tests whether equivalency between new or old implementation of nnp_graph
    '''
    _, inputs = module
    verbose = 1
    callback = nnp_graph.NnpNetworkPass(verbose)

    @callback.on_generate_function_by_name('Convolution')
    def change_convolution_param(f):
        print('{}'.format(f.proto.convolution_param.pad.dim[:]))
        f.proto.convolution_param.pad.dim[:] = [1, 1]
        return f

    @callback.on_function_pass_by_type('Affine')
    def change_affine_param(f, variables, param_scope):
        param_name = f.inputs[1].proto.name
        input_shape = f.inputs[0].proto.shape.dim[:]
        w_shape = f.inputs[1].proto.shape.dim[:]
        rng = np.random.RandomState(388)
        with nn.parameter_scope('', param_scope):
            W = nn.Variable.from_numpy_array(
                rng.randn(np.prod(input_shape[1:]), w_shape[1]))
            nn.parameter.set_parameter(param_name, W)
            W.need_grad = True

    with get_saved_test_model(module) as nnp_file:
        ref_nnp = legacy_nnp_graph.NnpLoader(nnp_file)
        nnp = nnp_graph.NnpLoader(nnp_file)
        for ref_v, v in zip(nnp_check(ref_nnp, 'left', callback),
                            nnp_check(nnp, 'right', callback)):
            verify_equivalence(ref_v, v)


@pytest.mark.parametrize("module", MODULES)
def test_nnp_load_parameter_scope(module):
    '''This test tests whether equivalency between new or old implementation of nnp_graph
    '''
    _, inputs = module
    verbose = 1
    callback = nnp_graph.NnpNetworkPass(verbose)

    @callback.on_generate_function_by_name('Convolution')
    def change_convolution_param(f):
        print('{}'.format(f.proto.convolution_param.pad.dim[:]))
        f.proto.convolution_param.pad.dim[:] = [1, 1]
        return f

    @callback.on_function_pass_by_type('Affine')
    def change_affine_param(f, variables, param_scope):
        param_name = f.inputs[1].proto.name
        input_shape = f.inputs[0].proto.shape.dim[:]
        w_shape = f.inputs[1].proto.shape.dim[:]
        rng = np.random.RandomState(388)
        with nn.parameter_scope('', param_scope):
            W = nn.Variable.from_numpy_array(
                rng.randn(np.prod(input_shape[1:]), w_shape[1]))
            W.need_grad = True
            nn.parameter.set_parameter(param_name, W)
    ref_params = {}
    with get_saved_test_model(module) as nnp_file:
        nnp = legacy_nnp_graph.NnpLoader(nnp_file)
        for network_name in sorted(nnp.get_network_names()):
            network = nnp.get_network(
                network_name, batch_size=32, callback=callback)
            ref_params[network_name] = nn.get_parameters().copy()
        nn.clear_parameters()

        params = {}
        nnp = nnp_graph.NnpLoader(nnp_file)
        assert_parameter_scope_empty()
        for network_name in sorted(nnp.get_network_names()):
            network = nnp.get_network(
                network_name, batch_size=32, callback=callback)
            params[network_name] = nn.get_parameters()

    assert_parameters_equal(ref_params, params)


@pytest.mark.parametrize("module", MODULES)
def test_networkpass_fix_parameter(module):
    '''This test tests whether equivalency between new or old implementation of nnp_graph
    '''
    _, inputs = module
    verbose = 1
    callback = nnp_graph.NnpNetworkPass(verbose)
    callback.fix_parameters()
    with get_saved_test_model(module) as nnp_file:
        nnp = nnp_graph.NnpLoader(nnp_file)
        assert_parameter_scope_empty()
        for network_name in sorted(nnp.get_network_names()):
            network = nnp.get_network(
                network_name, batch_size=32, callback=callback)
            assert_parameter_scope_empty()


@pytest.mark.parametrize("module", MODULES)
def test_networkpass_remove_and_rewire(module):
    '''This test tests whether equivalency between new or old implementation of nnp_graph
    '''
    _, inputs = module
    verbose = 1
    callback = nnp_graph.NnpNetworkPass(verbose)
    callback.remove_and_rewire('affine1_1')
    callback.remove_and_rewire('c1-c1')
    with get_saved_test_model(module) as nnp_file:
        ref_nnp = legacy_nnp_graph.NnpLoader(nnp_file)
        nnp = nnp_graph.NnpLoader(nnp_file)
        for ref_v, v in zip(nnp_check(ref_nnp, 'left', callback),
                            nnp_check(nnp, 'right', callback)):
            verify_equivalence(ref_v, v)


@pytest.mark.parametrize("module", MODULES)
def test_networkpass_use_up_to(module):
    '''This test tests whether equivalency between new or old implementation of nnp_graph
    '''
    _, inputs = module
    verbose = 1
    callback = nnp_graph.NnpNetworkPass(verbose)
    callback.use_up_to('Tanh_out_1')
    callback.use_up_to('Convolution_out_3')
    with get_saved_test_model(module) as nnp_file:
        ref_nnp = legacy_nnp_graph.NnpLoader(nnp_file)
        nnp = nnp_graph.NnpLoader(nnp_file)
        for ref_v, v in zip(nnp_check(ref_nnp, 'left', callback),
                            nnp_check(nnp, 'right', callback)):
            verify_equivalence(ref_v, v)


# @pytest.mark.parametrize("module", MODULES)
# def test_nnp_graph_topology_with_stop(module):
#     '''This test tests whether equivalency between new or old implementation of nnp_graph
#     '''
#     callback = nnp_graph.NnpNetworkPass(1)
#     callback.use_up_to('Tanh_out_1')
#     with get_saved_test_model(module) as nnp_file:
#         ref_nnp = legacy_nnp_graph.NnpLoader(nnp_file)
#         network = ref_nnp.get_network('model', batch_size=32, callback=callback)
#         # nnp = nnp_graph.NnpLoader(nnp_file)
#         # n = nnp.g.default_graph()
#         # n.promote(callback)
#         for k in sorted(network.variables.keys()):
#             print('variables', k, '{}, {}'.format(network.variables[k].shape, network.variables[k].d.shape))

def get_nnp(contents, tmpdir, need_file_object, file_type):
    import io
    from nnabla.utils.save import save
    from nnabla.utils import nnp_graph

    if file_type == '.nntxt' or file_type == '.prototxt':
        include_params = True
    else:
        include_params = False

    if need_file_object:
        nnp_object = io.BytesIO() if file_type == '.nnp' else io.StringIO()
        save(nnp_object, contents, extension=file_type,
             include_params=include_params)
        nnp_object.seek(0)
        nnp = nnp_graph.NnpLoader(nnp_object, extension=file_type)
    else:
        tmpdir.ensure(dir=True)
        nnp_file = tmpdir.join('tmp'+file_type).strpath
        save(nnp_file, contents, include_params=include_params)
        nnp = nnp_graph.NnpLoader(nnp_file)
    return nnp


@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("need_file_object", [True, False])
@pytest.mark.parametrize("file_type", ['.nnp', '.nntxt', '.prototxt'])
def test_nnp_graph_save_type(seed, tmpdir, need_file_object, file_type):

    rng = np.random.RandomState(seed)

    def unit(i, prefix):
        c1 = PF.convolution(i, 4, (3, 3), pad=(1, 1), name=prefix + '-c1')
        c2 = PF.convolution(F.relu(c1), 4,
                            (3, 3), pad=(1, 1), name=prefix + '-c2')
        c = F.add2(c2, c1, inplace=True)
        return c
    x = nn.Variable([2, 3, 4, 4])
    c1 = unit(x, 'c1')
    c2 = unit(x, 'c2')
    y = PF.affine(c2, 5, name='fc')

    runtime_contents = {
        'networks': [
            {'name': 'graph',
             'batch_size': 2,
             'outputs': {'y': y},
             'names': {'x': x}}],
    }
    nnp = get_nnp(runtime_contents, tmpdir, need_file_object, file_type)

    graph = nnp.get_network('graph')
    x2 = graph.inputs['x']
    y2 = graph.outputs['y']

    d = rng.randn(*x.shape).astype(np.float32)
    x.d = d
    x2.d = d
    y.forward(clear_buffer=True)
    y2.forward(clear_buffer=True)
    assert_allclose(y.d, y2.d)


def check_nnp_graph_save_load(tmpdir, x, y, batch_size, variable_batch_size):

    # Save
    contents = {
        'networks': [
            {'name': 'graph',
             'batch_size': 1,
             'outputs': {'y': y},
             'names': {'x': x}}]}
    from nnabla.utils.save import save
    tmpdir.ensure(dir=True)
    tmppath = tmpdir.join('tmp.nnp')
    nnp_file = tmppath.strpath
    save(nnp_file, contents,
         variable_batch_size=variable_batch_size)

    # Load
    from nnabla.utils import nnp_graph
    nnp = nnp_graph.NnpLoader(nnp_file)
    graph = nnp.get_network('graph', batch_size=batch_size)
    x2 = graph.inputs['x']
    y2 = graph.outputs['y']
    if not variable_batch_size:
        assert x2.shape == x.shape
        assert y2.shape == y.shape
        return x2, y2

    assert x2.shape[0] == batch_size
    assert y2.shape[0] == batch_size
    return x2, y2


@pytest.mark.parametrize('variable_batch_size', [False, True])
@pytest.mark.parametrize('batch_size', [1, 4])
@pytest.mark.parametrize("shape", [(10, 32, -1), (-1, 32, 8)])
def test_nnp_graph_reshape(tmpdir, variable_batch_size, batch_size, shape):
    x = nn.Variable([10, 2, 10, 10])
    h = PF.convolution(x, 4, kernel=(3, 3), stride=(1, 1))
    y = F.reshape(h, shape=shape)
    x2, y2 = check_nnp_graph_save_load(
        tmpdir, x, y, batch_size, variable_batch_size)
    if not variable_batch_size:
        return
    shape2 = list(y.shape)
    shape2[0] = batch_size
    x2.d = np.random.randn(*x2.shape)
    y2.forward()


@pytest.mark.parametrize('variable_batch_size', [False, True])
@pytest.mark.parametrize('batch_size', [1, 4])
def test_nnp_graph_broadcast(tmpdir, variable_batch_size, batch_size):
    x = nn.Variable([10, 2, 5, 5])
    h = PF.convolution(x, 4, kernel=(3, 3), stride=(2, 3))
    y = F.broadcast(h, shape=[10, 4, 2, 5])
    x2, y2 = check_nnp_graph_save_load(
        tmpdir, x, y, batch_size, variable_batch_size)
