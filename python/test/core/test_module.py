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
from nnabla.core.modules import ConvBn, ResUnit

from helper import ModuleCreator, forward_variable_and_check_equal

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


class TSTPureConv(nn.Module):
    def __init__(self):
        self.conv_bn = ConvBn(3)

    def call(self, x1):
        h1 = self.conv_bn(x1)
        h2 = self.conv_bn(h1)
        y = self.conv_bn(h2)
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

    def no_call(self, x1, x2):
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

    def no_call(self, x1, x2):
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
    #     1. cb = ConvBn(1), cb owned a few of parameters
    #     In the following, cb owned parameters will release after leave the scope.
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


@pytest.mark.parametrize("module_creator", [ModuleCreator(TSTNetNormal(), [(4, 3, 32, 32), (4, 3, 32, 32)])])
def test_create_graph_def_without_using_call(module_creator):
    # get module from test parameters
    module = module_creator.module

    # create proto variables as inputs
    proto_variable_inputs = [nn.ProtoVariable(
        shape) for shape in module_creator.input_shape]

    # create graph_def by passing proto_variables as inputs
    with nn.parameter_scope('', module.parameter_scope):
        # Here, if user does not call module(), instead, they call a self-defined
        # function, we should not produce strange result.
        outputs = module.no_call(*proto_variable_inputs)

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


@pytest.mark.parametrize("module_creator", [ModuleCreator(TSTNetNormal(), [(4, 3, 32, 32), (4, 3, 32, 32)])])
def test_with_statement_graph_def_without_call(module_creator):
    # get module from test parameters
    module = module_creator.module

    # create proto variables as inputs
    proto_variable_inputs = [nn.ProtoVariable(
        shape) for shape in module_creator.input_shape]

    # create graph_def by passing proto_variables as inputs
    with nn.graph_def.graph() as g:
        outputs = module.no_call(*proto_variable_inputs)

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
def test_simple_create_graph_def(module_creator):
    # get module from test parameters
    module = module_creator.module

    # create proto variables as inputs
    proto_variable_inputs = [nn.ProtoVariable(
        shape) for shape in module_creator.input_shape]

    # create graph_def by passing proto_variables as inputs
    outputs = module(*proto_variable_inputs)

    # find out graph_def in global default graph
    g = nn.graph_def.get_default_graph()

    # create variable inputs and initialized by random value
    variable_inputs = module_creator.get_variable_inputs()

    # create network by module-like graph_def
    outputs = g(*variable_inputs)

    # create reference network by passing in variable inputs
    ref_outputs = module(*variable_inputs)

    # check if outputs are equal
    forward_variable_and_check_equal(outputs, ref_outputs)


@pytest.mark.parametrize("module_creator", [ModuleCreator(TSTPureConv(), [(4, 3, 32, 32)])])
@pytest.mark.parametrize("another_input_shape", [(1, 3, 64, 64), (1, 3, 80, 80)])
def test_another_shape_input(module_creator, another_input_shape):
    # get module from test parameters
    module = module_creator.module

    # create proto variables as inputs
    proto_variable_inputs = [nn.ProtoVariable(
        shape) for shape in module_creator.input_shape]

    # create graph_def by passing proto_variables as inputs
    outputs = module(*proto_variable_inputs)

    # find out graph_def in global default graph
    g = nn.graph_def.get_default_graph()

    input = nn.Variable(another_input_shape)

    output = g(input)

    output.forward()


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
    # get module from test module
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


def test_with_statement_graph_def_multiple_network_no_call():
    """This cases assume that user create network multiple times in a ProtoGraph.
    Because no graph_name() is called to name each network, all computation graph
    operators are collected into a network, it looks like multiple networks
    are merged into a network. In testing time, we only need to feed concatenated
    inputs to this network and check concatenate outputs.
    """
    module_creators = [ModuleCreator(TSTNetNormal(), [(4, 3, 32, 32), (4, 3, 32, 32)]),
                       ModuleCreator(NestedTestNet(), [(4, 3, 32, 32), (4, 3, 32, 32)])]

    # create graph_def by passing proto_variables as inputs
    with nn.graph_def.graph() as g:
        for module_creator in module_creators:
            module = module_creator.module

            # create proto variables as inputs
            proto_variable_inputs = [nn.ProtoVariable(
                shape) for shape in module_creator.input_shape]

            # generate graph
            outputs = module.no_call(*proto_variable_inputs)

    for module_creator, network in zip(module_creators, g.networks.values()):
        # create variable inputs and initialized by random value
        variable_inputs = module_creator.get_variable_inputs()

        # create network by module-like graph_def
        outputs = network(*variable_inputs)

        # create reference network by passing in variable inputs
        ref_outputs = module_creator.module(*variable_inputs)

        # check if outputs are equal
        forward_variable_and_check_equal(outputs, ref_outputs)


def test_multiple_network():
    """This cases assume that user create network multiple times in a ProtoGraph.
    Because no graph_name() is called to name each network, all computation graph
    operators are collected into a network, it looks like multiple networks
    are merged into a network. In testing time, we only need to feed concatenated
    inputs to this network and check concatenate outputs.
    """
    module_creators = [ModuleCreator(TSTNetNormal(), [(4, 3, 32, 32), (4, 3, 32, 32)]),
                       ModuleCreator(ResUnit(16), [(4, 3, 32, 32)]),
                       ModuleCreator(NestedTestNet(), [(4, 3, 32, 32), (4, 3, 32, 32)])]

    # create graph_def by passing proto_variables as inputs
    with nn.graph_def.graph() as g:
        for module_creator in module_creators:
            module = module_creator.module

            # create proto variables as inputs
            proto_variable_inputs = [nn.ProtoVariable(
                shape) for shape in module_creator.input_shape]

            # generate graph
            outputs = module(*proto_variable_inputs)

    for module_creator, network in zip(module_creators, g.networks.values()):
        # create variable inputs and initialized by random value
        variable_inputs = module_creator.get_variable_inputs()

        # create network by module-like graph_def
        outputs = network(*variable_inputs)

        # create reference network by passing in variable inputs
        ref_outputs = module_creator.module(*variable_inputs)

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


@pytest.mark.parametrize("module_func", [(flat_module, [(4, 3, 32, 32)]),
                                         (mlp_module, [(16, 100), (16, 100)])])
def test_iterator_through_forward_sequence(module_func):
    func, in_shapes = module_func
    with nn.graph_def.graph() as g:
        inputs = [nn.ProtoVariable(shape) for shape in in_shapes]
        outputs = func(*inputs)

    inputs = [nn.Variable(shape) for shape in in_shapes]
    for i in inputs:
        i.d = np.random.random(i.shape)
    outputs_ref = func(*inputs)
    if not isinstance(outputs_ref, tuple):
        outputs_ref = (outputs_ref, )

    output = F.sink(*outputs_ref)
    forward_sequence = []

    def visit_func(f):
        if f.name != 'Sink':
            forward_sequence.append(f.name)

    output.visit(visit_func)

    for a, b in zip(g.default_graph().forward_sequence(), forward_sequence):
        assert a.type == b


@pytest.mark.parametrize("module_func", [(flat_module, [(4, 3, 32, 32)]),
                                         (mlp_module, [(16, 100), (16, 100)])])
@pytest.mark.parametrize("backend", ['cpu', 'cuda', 'cudnn'])
def test_context_priority_hierarchy_testing(module_func, backend):
    context_backend = ''
    func, in_shapes = module_func
    with nn.graph_def.graph() as g:
        inputs = [nn.ProtoVariable(shape) for shape in in_shapes]
        outputs = func(*inputs)

    import nnabla_ext.cpu
    nn.set_default_context(nnabla_ext.cpu.context())

    if backend == 'cpu':
        ctx = nnabla_ext.cpu.context()
    elif backend == 'cuda':
        try:
            import nnabla_ext.cuda
            ctx = nnabla_ext.cuda.context()
            context_backend = 'Cuda'
        except ImportError:
            ctx = nnabla_ext.cpu.context()
    elif backend == 'cudnn':
        try:
            import nnabla_ext.cudnn
            ctx = nnabla_ext.cudnn.context()
            context_backend = 'Cudnn'
        except ImportError:
            ctx = nnabla_ext.cpu.context()

    g.current_context = ctx
    g()
    for f in g.default_graph().forward_sequence():
        if context_backend:
            assert ('Cuda' in f.function_instance.name)


def test_training_property():
    import nnabla.parametric_functions as PF

    class ConvBn(nn.Module):
        def call(self, x):
            h = PF.convolution(x, 3, (3, 2))
            return h

    model = ConvBn()
    x = nn.Variable((64, 3, 32, 32))
    y = model(x)
    model.training = True
    assert model.training == True

# TODO:
#   - binary-operator of ProtoVariable testing
