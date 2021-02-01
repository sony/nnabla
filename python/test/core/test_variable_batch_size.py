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
from nnabla.utils import load
from nnabla.utils import save
from nnabla.utils import nnp_graph
from helper import forward_variable


def base_axis_0_reshape_with_neg_1(x):
    h = PF.convolution(x, 3, (3, 3), pad=(0, 0), name='c1', base_axis=0)
    y = F.reshape(h, shape=(3, -1))
    return y


def base_axis_0_reshape_without_neg_1(x):
    h = PF.convolution(x, 3, (3, 3), pad=(0, 0), name='c1', base_axis=0)
    y = F.reshape(h, shape=(3, 36))
    return y


def base_axis_0_broadcast(x):
    h = PF.convolution(x, 3, (2, 2), pad=(0, 0), name='c1', base_axis=0)
    y = F.broadcast(h, shape=(3, 3, 3))
    return y


def base_axis_1_reshape_with_neg_1(x):
    h = PF.convolution(x, 3, (3, 3), pad=(0, 0), name='c1', base_axis=1)
    y = F.reshape(h, shape=(1, 18, -1))
    return y


def base_axis_1_reshape_without_neg_1(x):
    h = PF.convolution(x, 3, (3, 3), pad=(0, 0), name='c1', base_axis=1)
    y = F.reshape(h, shape=(1, 18, 6))
    return y


def base_axis_1_broadcast(x):
    h = PF.convolution(x, 3, (2, 2), pad=(0, 0), name='c1', base_axis=1)
    y = F.broadcast(h, shape=(2, 3, 3, 3))
    return y


def base_axis_2_reshape_with_neg_1(x):
    h = PF.convolution(x, 3, (3, 3), pad=(0, 0), name='c1', base_axis=2)
    y = F.reshape(h, shape=(2, 18, -1))
    return y


def base_axis_2_reshape_without_neg_1(x):
    h = PF.convolution(x, 3, (3, 3), pad=(0, 0), name='c1', base_axis=2)
    y = F.reshape(h, shape=(2, 18, 6))
    return y


def base_axis_2_broadcast(x):
    h = PF.convolution(x, 3, (2, 2), pad=(0, 0), name='c1', base_axis=2)
    y = F.broadcast(h, shape=(1, 2, 3, 3, 3))
    return y


def save_model_from_utils_save(nnp_file, model_def, input_shape, variable_batch_size):
    x = nn.Variable(input_shape)
    y = model_def(x)
    contents = {
        'networks':
            [{'name': 'model',
              'batch_size': 1,
              'outputs': {'y': y},
              'names': {'x': x}}],
        'executors':
            [{'name': 'runtime',
              'network': 'model',
              'data': ['x'],
              'output': ['y']}]}
    save.save(nnp_file, contents, variable_batch_size=variable_batch_size)


def save_model_from_graph_def_save(nnp_file, model_def, input_shape, variable_batch_size):
    with nn.graph_def.graph() as g:
        x = nn.ProtoVariable(input_shape)
        y = model_def(x)
    g.save(nnp_file, variable_batch_size=variable_batch_size)


def load_model_from_utils_load_and_forward(nnp_file, batch_size):
    g = load.load(nnp_file, batch_size=batch_size).proto_graph
    inputs = [g.default_graph(
    ).variables[i].variable_instance for i in g.default_graph().inputs]
    for i in inputs:
        i.d = np.random.random(i.d.shape)
    outputs = [g.default_graph(
    ).variables[i].variable_instance for i in g.default_graph().outputs]
    y = F.sink(*outputs)
    y.forward()
    out = outputs[0].d
    return out


def load_model_from_graph_def_load_and_forward(nnp_file, batch_size):
    graph = nn.graph_def.load(nnp_file, batch_size=batch_size)
    y = graph()
    inputs = [graph.default_graph(
    ).variables[i].variable_instance for i in graph.default_graph().inputs]
    for i in inputs:
        i.d = np.random.random(i.d.shape)
    y.forward()
    return y.d


def load_model_from_nnp_graph_and_forward(nnp_file, batch_size):
    nnp = nnp_graph.NnpLoader(nnp_file)
    network = nnp.get_network(nnp.get_network_names()[
                              0], batch_size=batch_size)
    inputs = list(network.inputs.values())
    outputs = list(network.outputs.values())
    out = []
    for d in forward_variable(inputs, outputs, 'left'):
        out.append(d)
    return d


@pytest.mark.parametrize("batch_size", [4, 8, 16])
@pytest.mark.parametrize("variable_batch_size, model_def, input_shape, expect_batch_size", [
    (True, base_axis_0_reshape_with_neg_1, (3, 8, 8), 3),
    (True, base_axis_0_reshape_without_neg_1, (3, 8, 8), 3),
    (True, base_axis_0_broadcast, (3, 2, 2), 3),
    (True, base_axis_1_reshape_with_neg_1, (1, 3, 8, 8), -1),
    (True, base_axis_1_reshape_without_neg_1, (1, 3, 8, 8), -1),
    (True, base_axis_1_broadcast, (2, 3, 2, 2), -1),
    (True, base_axis_2_reshape_with_neg_1, (1, 2, 3, 8, 8), -1),
    (True, base_axis_2_reshape_without_neg_1, (1, 2, 3, 8, 8), -1),
    (True, base_axis_2_broadcast, (1, 2, 3, 2, 2), -1),
    (False, base_axis_0_reshape_with_neg_1, (3, 8, 8), 3),
    (False, base_axis_0_reshape_without_neg_1, (3, 8, 8), 3),
    (False, base_axis_0_broadcast, (3, 2, 2), 3),
])
@pytest.mark.parametrize("save_model", [
    save_model_from_utils_save,
    save_model_from_graph_def_save,
])
@pytest.mark.parametrize("load_model_and_forward", [
    load_model_from_utils_load_and_forward,
    load_model_from_graph_def_load_and_forward,
    load_model_from_nnp_graph_and_forward,
])
def test_variable_batch_size(tmpdir, batch_size, variable_batch_size, model_def, input_shape, expect_batch_size, save_model, load_model_and_forward):
    nn.clear_parameters()
    # step1: Create and save the network model
    tmpdir.ensure(dir=True)
    nnp_file = tmpdir.join('tmp.nnp').strpath
    save_model(nnp_file, model_def, input_shape, variable_batch_size)
    # step2: Load network model and forward.
    out = load_model_and_forward(nnp_file, batch_size)
    # step3: Varify batch_size
    if expect_batch_size != -1:
        batch_size = expect_batch_size
    assert (batch_size == out.shape[0])
