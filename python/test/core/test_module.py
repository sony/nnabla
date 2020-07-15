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

import os
import pytest
import numpy as np
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
from nnabla.testing import assert_allclose
from nnabla.core.modules import ConvBn, ResUnit

nnp_file = "t.nnp"


@pytest.fixture(scope="function", autouse=True)
def clear_temporary_file():
    yield
    if os.path.exists(nnp_file):
        os.remove(nnp_file)


class Shared(nn.Module):
    def __init__(self, module):
        self.module = module

    def call(self, x):
        return self.module(x)


class Example(nn.Module):
    def __init__(self):

        self.cb = ConvBn(2)
        self.cb2 = ConvBn(2)
        self.shared1 = Shared(self.cb2)
        self.shared2 = Shared(self.cb2)

    def call(self, x):
        h = self.cb(x)
        h = self.cb2(h)
        h = self.shared1(h)
        h = self.shared2(h)
        return h


class TSTNetAbnormal(nn.Module):
    def __init__(self):
        self.conv_bn_1 = ConvBn(1)
        self.conv_bn_2 = ConvBn(1)

    def call(self, x1, x2):
        y1 = self.conv_bn_1(x1)
        y2 = self.conv_bn_2(x2)
        y = F.concatenate(y1, y2, axis=1)
        # ConvBn() will be destroyed when leave this scope.
        # Thus, the parameters owned by `cb` object will be released too.
        cb = ConvBn(1)
        y = F.concatenate(y, cb(x1), axis=1)
        return y


class TSTNetNormal(nn.Module):
    def __init__(self):
        self.conv_bn_1 = ConvBn(1)
        self.conv_bn_2 = ConvBn(1)

    def call(self, x1, x2):
        y1 = self.conv_bn_1(x1)
        y2 = self.conv_bn_2(x2)
        y = F.concatenate(y1, y2, axis=1)
        return y


class NestedTestNet(nn.Module):
    def __init__(self):
        self.tnd1 = TSTNetNormal()
        self.tnd2 = TSTNetNormal()

    def call(self, x1, x2):
        y1 = self.tnd1(x1, x2)
        y2 = self.tnd2(x1, x2)
        y = F.concatenate(y1, y2, axis=1)
        return y


def test_module_parameter_sharing():
    x = nn.Variable((4, 3, 32, 32))

    # Define a module and create a graph with it
    e = Example()
    h = e(x)

    # Create another graph with the same architecture
    e2 = Example()
    assert not e2.get_parameters(), "It doesn't have any parameters so far."

    # Setting parameters from an existing model)
    e2.set_parameters(e.get_parameters())
    assert e.get_parameters() == e2.get_parameters(), "They have the same parameters."

    #  Executing both
    h2 = e2(x)
    x.d = np.random.randn(*x.shape)
    h.forward(clear_buffer=True)
    h2.forward(clear_buffer=True)
    assert np.allclose(h.d, h2.d)


def test_module_parameter_path():
    x = nn.Variable((4, 3, 32, 32))

    # Define a module and create a graph with it
    e = Example()
    h = e(x)

    # Create another graph with the same architecture
    e2 = Example()
    assert not e2.get_parameters(), "It doesn't have any parameters so far."

    # Setting parameters from an existing model)
    e2.set_parameters(e.get_parameters())
    assert e.get_parameters() == e2.get_parameters(), "They have the same parameters."

    assert '@cb/conv/W' in e.get_parameters()
    assert '@cb/conv/W' in e2.get_parameters()


class ModuleCreator:
    def __init__(self, module, input_shape):
        self.module = module
        self.input_shape = input_shape
        self.variable_inputs = None
        self.proto_variable_inputs = None

    def get_variable_inputs(self):
        if self.variable_inputs is None:
            variable_inputs = [nn.Variable(shape)
                               for shape in self.input_shape]
            variable_values = [np.random.random(
                shape) for shape in self.input_shape]
            for v, d in zip(variable_inputs, variable_values):
                v.d = d
            self.variable_inputs = variable_inputs
            return variable_inputs
        return self.variable_inputs

    def get_proto_variable_inputs(self):
        if self.proto_variable_inputs is None:
            proto_variable_inputs = [nn.ProtoVariable(
                shape) for shape in self.input_shape]
            self.proto_variable_inputs = proto_variable_inputs
            return proto_variable_inputs
        return self.proto_variable_inputs


def forward_variable_and_check_equal(variable_a, variable_b):
    def forward_output(variable):
        if isinstance(variable, nn.Variable):
            variable.forward()
        else:
            y = F.sink(*variable)
            for v in variable:
                v.persistent = True
            y.forward()

    for v in [variable_a, variable_b]:
        forward_output(v)
    if isinstance(variable_a, nn.Variable):
        assert_allclose(variable_a.d, variable_b.d)
        return
    for a, b in zip(variable_a, variable_b):
        assert_allclose(a.d, b.d, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("module_creator", [ModuleCreator(TSTNetAbnormal(), [(4, 3, 32, 32), (4, 3, 32, 32)])])
def test_unsupported_module_definition(module_creator):
    # Since we use global graphdef, we should reset it beforehand
    # This is already done in test fixture.
    # nn.graph_def.reset_default_graph()

    # get module from test parameters
    module = module_creator.module

    # create variable inputs and initialized by random value
    variable_inputs = module_creator.get_variable_inputs()

    # create reference network by passing in variable inputs
    ref_outputs = module(*variable_inputs)

    # create proto variable inputs
    proto_variable_inputs = module_creator.get_proto_variable_inputs()
    #
    # generate proto graph
    proto_outputs = module(*proto_variable_inputs)
    #
    # find out graph_def in global default graph
    g = nn.graph_def.get_default_graph()

    outputs = g(*variable_inputs)

    # Here, the result is impossible to equal.
    # Since the variable instance of parameters in local module instance
    # will be re-created in another time's instantiated.
    #     1. cb = ConvBn(1), cb ownered a few of parameters
    #     In the following, cb ownered parameters will release after leave the scope.
    #          ref_outputs = module(*variable_inputs)
    #     2. proto_outputs = module(*proto_variable_inputs)
    #     The new parameters will be created during generating output.
    #     This new parameters are different from above module.
    #     3. result is different.
    # Conclusion:
    #     Module-based graph definition does not allow a local submodule instance.
    #     it must be referred as current module's member variable, otherwise,
    #     the parameters will fly away when leave module.call().
    with pytest.raises(AssertionError) as excinfo:
        forward_variable_and_check_equal(outputs, ref_outputs)
    print(excinfo.value)


@pytest.mark.parametrize("module_creator", [ModuleCreator(TSTNetNormal(), [(4, 3, 32, 32), (4, 3, 32, 32)]),
                                            ModuleCreator(
                                                ResUnit(16), [(4, 3, 32, 32)]),
                                            ModuleCreator(NestedTestNet(), [(4, 3, 32, 32), (4, 3, 32, 32)])])
def test_simple_create_graph_def(module_creator):
    # get module from test parameters
    module = module_creator.module

    # create proto variables as inputs
    proto_variable_inputs = [nn.ProtoVariable(
        shape) for shape in module_creator.input_shape]

    # create graph_def by passing proto_variables as inputs
    outputs = module(*proto_variable_inputs)

    # find out graph_def in global default graph
    graph_def = nn.graph_def.get_default_graph()

    # create variable inputs and initialized by random value
    variable_inputs = module_creator.get_variable_inputs()

    # create network by module-like graph_def
    outputs = graph_def(*variable_inputs)

    # create reference network by passing in variable inputs
    ref_outputs = module(*variable_inputs)

    # check if outputs are equal
    forward_variable_and_check_equal(outputs, ref_outputs)


@pytest.mark.parametrize("module_creator", [ModuleCreator(TSTNetNormal(), [(4, 3, 32, 32), (4, 3, 32, 32)]),
                                            ModuleCreator(
                                                ResUnit(16), [(4, 3, 32, 32)]),
                                            ModuleCreator(NestedTestNet(), [(4, 3, 32, 32), (4, 3, 32, 32)])])
def test_get_graph_def_by_variable(module_creator):
    # get module from test parameters
    module = module_creator.module

    # create proto variables as inputs
    proto_variable_inputs = [nn.ProtoVariable(
        shape) for shape in module_creator.input_shape]

    # create graph_def by passing proto_variables as inputs
    outputs = module(*proto_variable_inputs)

    # find out graph_def in global default graph
    graph_def = nn.graph_def.get_default_graph_by_variable(outputs)

    # create variable inputs and initialized by random value
    variable_inputs = module_creator.get_variable_inputs()

    # create network by module-like graph_def
    outputs = graph_def(*variable_inputs)

    # create reference network by passing in variable inputs
    ref_outputs = module(*variable_inputs)

    # check if outputs are equal
    forward_variable_and_check_equal(outputs, ref_outputs)


@pytest.mark.parametrize("module_creator", [ModuleCreator(TSTNetNormal(), [(4, 3, 32, 32), (4, 3, 32, 32)]),
                                            ModuleCreator(
                                                ResUnit(16), [(4, 3, 32, 32)]),
                                            ModuleCreator(NestedTestNet(), [(4, 3, 32, 32), (4, 3, 32, 32)])])
def test_with_statement_graph_def(module_creator):
    # get module from test parameters
    module = module_creator.module

    # create proto variables as inputs
    proto_variable_inputs = [nn.ProtoVariable(
        shape) for shape in module_creator.input_shape]

    # create graph_def by passing proto_variables as inputs
    with nn.graph_def.graph() as g:
        outputs = module(*proto_variable_inputs)

    # create variable inputs and initialized by random value
    variable_inputs = module_creator.get_variable_inputs()

    # create network by module-like graph_def
    outputs = g(*variable_inputs)

    # create reference network by passing in variable inputs
    ref_outputs = module(*variable_inputs)

    # check if outputs are equal
    forward_variable_and_check_equal(outputs, ref_outputs)


@pytest.mark.parametrize("module_creator", [ModuleCreator(TSTNetNormal(), [(4, 3, 32, 32), (4, 3, 32, 32)]),
                                            ModuleCreator(
                                                ResUnit(16), [(4, 3, 32, 32)]),
                                            ModuleCreator(NestedTestNet(), [(4, 3, 32, 32), (4, 3, 32, 32)])])
@pytest.mark.parametrize("network_name", ['TestNet', "network"])
def test_with_statement_graph_def_name(module_creator, network_name):
    module = module_creator.module

    # create proto variables as inputs
    proto_variable_inputs = [nn.ProtoVariable(
        shape) for shape in module_creator.input_shape]

    # create graph_def by passing proto_variables as inputs
    with nn.graph_def.graph(name=network_name) as g:
        outputs = module(*proto_variable_inputs)

    # create variable inputs and initialized by random value
    variable_inputs = module_creator.get_variable_inputs()

    # create network by module-like graph_def
    outputs = g[network_name](*variable_inputs)

    # create reference network by passing in variable inputs
    ref_outputs = module(*variable_inputs)

    # check if outputs are equal
    forward_variable_and_check_equal(outputs, ref_outputs)


@pytest.mark.parametrize("module_creator", [ModuleCreator(TSTNetNormal(), [(4, 3, 32, 32), (4, 3, 32, 32)]),
                                            ModuleCreator(
                                                ResUnit(16), [(4, 3, 32, 32)]),
                                            ModuleCreator(NestedTestNet(), [(4, 3, 32, 32), (4, 3, 32, 32)])])
def test_with_statement_graph_def_test_name(module_creator):
    """ This case tests specifying the name of a graph.
    """
    module = module_creator.module

    # create proto variables as inputs
    proto_variable_inputs = [nn.ProtoVariable(
        shape) for shape in module_creator.input_shape]

    # create graph_def by passing proto_variables as inputs
    with nn.graph_def.graph(name="test_net") as g:
        outputs = module(*proto_variable_inputs)

    # create variable inputs and initialized by random value
    variable_inputs = module_creator.get_variable_inputs()

    # create network by module-like graph_def
    # Access the network directly by network name
    outputs = g.test_net(*variable_inputs)

    # create reference network by passing in variable inputs
    ref_outputs = module(*variable_inputs)

    # check if outputs are equal
    forward_variable_and_check_equal(outputs, ref_outputs)


def test_get_graph_def_by_name():
    """This cases assume that user creates multiple networks in a ProtoGraph.
    User may specify the name of network(graph) they created by graph_name().
    """
    module_creators = [ModuleCreator(TSTNetNormal(), [(4, 3, 32, 32), (4, 3, 32, 32)]),
                       ModuleCreator(ResUnit(16), [(4, 3, 32, 32)]),
                       ModuleCreator(NestedTestNet(), [(4, 3, 32, 32), (4, 3, 32, 32)])]
    network_names = [
        'network1',
        'network2',
        'network3'
    ]

    # create graph_def by passing proto_variables as inputs
    with nn.graph_def.graph() as g:
        for module_creator, network_name in zip(module_creators, network_names):
            module = module_creator.module

            # create proto variables as inputs
            proto_variable_inputs = [nn.ProtoVariable(
                shape) for shape in module_creator.input_shape]

            with nn.graph_def.graph_name(network_name):
                # generate graph
                outputs = module(*proto_variable_inputs)

    for module_creator, network_name in zip(module_creators, network_names):
        # create variable inputs and initialized by random value
        variable_inputs = module_creator.get_variable_inputs()

        # create network by module-like graph_def
        outputs = g[network_name](*variable_inputs)

        # create reference network by passing in variable inputs
        ref_outputs = module_creator.module(*variable_inputs)

        # check if outputs are equal
        forward_variable_and_check_equal(outputs, ref_outputs)


def test_get_default_graph_def_by_name():
    """This case tests retrieving graph using nn.graph_def.get_default_graph() by
    specifying the name of graph.
    """
    module_creators = [ModuleCreator(TSTNetNormal(), [(4, 3, 32, 32), (4, 3, 32, 32)]),
                       ModuleCreator(ResUnit(16), [(4, 3, 32, 32)]),
                       ModuleCreator(NestedTestNet(), [(4, 3, 32, 32), (4, 3, 32, 32)])]
    network_names = [
        'network1',
        'network2',
        'network3'
    ]

    for module_creator, network_name in zip(module_creators, network_names):
        module = module_creator.module

        # create proto variables as inputs
        proto_variable_inputs = [nn.ProtoVariable(
            shape) for shape in module_creator.input_shape]

        with nn.graph_def.graph_name(network_name):
            # generate graph
            outputs = module(*proto_variable_inputs)

    for module_creator, network_name in zip(module_creators, network_names):
        # create variable inputs and initialized by random value
        variable_inputs = module_creator.get_variable_inputs()

        # get graph from default by name
        g = nn.graph_def.get_default_graph(network_name)

        # create network by module-like graph_def
        outputs = g(*variable_inputs)

        # create reference network by passing in variable inputs
        ref_outputs = module_creator.module(*variable_inputs)

        # check if outputs are equal
        forward_variable_and_check_equal(outputs, ref_outputs)


@pytest.mark.parametrize("module_creator", [ModuleCreator(TSTNetNormal(), [(4, 3, 32, 32), (4, 3, 32, 32)]),
                                            ModuleCreator(
                                                ResUnit(16), [(4, 3, 32, 32)]),
                                            ModuleCreator(NestedTestNet(), [(4, 3, 32, 32), (4, 3, 32, 32)])])
def test_save_load_consistency(module_creator):
    module = module_creator.module

    # create proto variables as inputs
    proto_variable_inputs = [nn.ProtoVariable(
        shape) for shape in module_creator.input_shape]

    # create graph_def by passing proto_variables as inputs
    with nn.graph_def.graph() as g:
        outputs = module(*proto_variable_inputs)

    g.save(nnp_file)

    g = nn.graph_def.load(nnp_file)

    # create variable inputs and initialized by random value
    variable_inputs = module_creator.get_variable_inputs()

    # create network by module-like graph_def
    outputs = g(*variable_inputs)

    # create reference network by passing in variable inputs
    ref_outputs = module(*variable_inputs)

    # check if outputs are equal
    forward_variable_and_check_equal(outputs, ref_outputs)


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
    h1_1 = PF.affine(x1, 100, name='affine1_0')
    h1 = F.tanh(h1_0 + h1_1)
    h2 = F.tanh(PF.affine(h1, 50, name='affine2'))
    y0 = PF.affine(h2, 10, name='affiney_0')
    y1 = PF.affine(h2, 10, name='affiney_1')
    return y0, y1


@pytest.mark.parametrize("module_func", [(flat_module, [(4, 3, 32, 32)]),
                                         (mlp_module, [(16, 100), (16, 100)])])
def test_flat_module_support(module_func):
    '''This case is used to test creating graph_def by
    passing nn.ProtoVariable() to a flat-style model.
    '''
    func, in_shapes = module_func

    inputs = [nn.ProtoVariable(shape) for shape in in_shapes]
    outputs = func(*inputs)
    if not isinstance(outputs, tuple):
        outputs = (outputs, )
    g = nn.graph_def.get_default_graph_by_variable(outputs[0])
    g.save(nnp_file)

    g = nn.graph_def.load(nnp_file)

    inputs = [nn.Variable(shape) for shape in in_shapes]
    for i in inputs:
        i.d = np.random.random(i.shape)
    outputs = g(*inputs)
    outputs_ref = func(*inputs)

    forward_variable_and_check_equal(outputs, outputs_ref)


@pytest.mark.parametrize("module_func", [(flat_module, [(4, 3, 32, 32)]),
                                         (mlp_module, [(16, 100), (16, 100)])])
def test_flat_module_with_statement(module_func):
    '''This case is used to test creating graph_def by
    passing nn.ProtoVariable() to a flat-style model.
    '''
    func, in_shapes = module_func
    with nn.graph_def.graph() as g:
        inputs = [nn.ProtoVariable(shape) for shape in in_shapes]
        outputs = func(*inputs)
    g.save(nnp_file)

    inputs = [nn.Variable(shape) for shape in in_shapes]
    for i in inputs:
        i.d = np.random.random(i.shape)
    outputs = g(*inputs)
    outputs_ref = func(*inputs)

    forward_variable_and_check_equal(outputs, outputs_ref)


# TODO:
#   - context priority hierarchy testing
#   - context setting change testing
#   - batch size change testing (Not implemented yet)
#   - binary-operator of ProtoVariable testing
