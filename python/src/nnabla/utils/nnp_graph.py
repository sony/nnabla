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

import os
import weakref
import numpy as np

import nnabla as nn
import nnabla.function as F
from nnabla.utils import nnabla_pb2
from nnabla.parameter import get_parameter
from nnabla.utils.load_function import _create_function_instance
from nnabla.utils.load import (
    resolve_reshape_params,
    resolve_broadcast_params)


def _load_nnp_to_proto(nnp_path):
    import google.protobuf.text_format as text_format
    import tempfile
    import zipfile
    import shutil
    proto = nnabla_pb2.NNablaProtoBuf()

    tmpdir = tempfile.mkdtemp()
    try:
        with zipfile.ZipFile(nnp_path, "r") as nnp:
            for name in nnp.namelist():
                _, ext = os.path.splitext(name)
                if name == "nnp_version.txt":
                    pass  # Currently nnp_version.txt is ignored
                elif ext in [".nntxt", ".prototxt"]:
                    nnp.extract(name, tmpdir)
                    with open(os.path.join(tmpdir, name), "rt") as f:
                        text_format.Merge(f.read(), proto)
                elif ext in [".protobuf", ".h5"]:
                    nnp.extract(name, tmpdir)
                    nn.load_parameters(os.path.join(tmpdir, name))
    finally:
        shutil.rmtree(tmpdir)

    return proto


def _create_function(inputs, f, batch_size):
    ctx = nn.get_current_context()
    function_proto = f

    # todo: arrange weight name for NNC

    if function_proto.type == "Reshape":  # if batch_size = -1, something wrong?
        reshape_shape = resolve_reshape_params(
            inputs, function_proto, batch_size)
        function_instance = F.Reshape(
            ctx, shape=reshape_shape, inplace=function_proto.reshape_param.inplace)
    elif function_proto.type == 'Broadcast':
        shape = resolve_broadcast_params(inputs, function_proto, batch_size)
        function_instance = F.Broadcast(ctx, shape=shape)
    elif function_proto.type == "RepeatStart":
        raise NotImplementedError("Repeat not supported.")
        function_instance = F.Identity(ctx)
    elif function_proto.type == "RepeatEnd":
        raise NotImplementedError("Repeat not supported.")
        function_instance = F.Identity(ctx)
    elif function_proto.type == "RecurrentOutput":
        raise NotImplementedError("Recurrent not supported.")
        function_instance = F.Stack(
            ctx, axis=function_proto.recurrent_param.axis)
    elif function_proto.type == "RecurrentInput":
        raise NotImplementedError("Recurrent not supported.")
        function_instance = F.Split(
            ctx, axis=function_proto.recurrent_param.axis)
    elif function_proto.type == "Delay":
        raise NotImplementedError("Recurrent not supported.")
        function_instance = F.Identity(ctx)
    else:
        function_instance = _create_function_instance(ctx, function_proto)

    return function_instance


class VariableProto:

    def __init__(self, v):
        self.proto = v
        self.parent = None
        self._referrers = []
        self.variable = None

    def add_referrer(self, f):
        assert isinstance(f, FunctionProto)
        self._referrers.append(weakref.ref(f))

    @property
    def referrers(self):
        referrers = [r() for r in self.referrers]
        assert all([r is not None for r in referrers])
        return referrers

    @property
    def num_referrers(self):
        return len(self._referrers)


class FunctionProto:
    def __init__(self, proto):
        self.proto = proto
        self._inputs = []
        self._outputs = []
        self.function = None

    @property
    def inputs(self):
        return self._inputs

    @inputs.setter
    def inputs(self, inputs):
        for i in inputs:
            assert isinstance(i, VariableProto)
            i.add_referrer(self)
        self._inputs = list(inputs)

    @property
    def outputs(self):
        outputs = [o() for o in self._outputs]
        assert all([os is not None for o in outputs])
        return outputs

    @outputs.setter
    def outputs(self, outputs):
        for o in outputs:
            assert isinstance(o, VariableProto)
        self._outputs = [weakref.ref(o) for o in outputs]
        for o in outputs:
            o.parent = self


def visit_forward(variables, callback, fclosed=None):
    if fclosed is None:
        fclosed = set()
    for v in variables:
        f = v.parent
        if f is None:
            continue
        if f in fclosed:
            continue
        fclosed.add(f)
        visit_forward(f.inputs, callback, fclosed)
        callback(f)


class NnpNetwork(object):
    '''A graph object which is read from nnp file.


    An instance of NnpNetwork is usually created by an NnpLoader instance.
    See an example usage described in :obj:`NnpLoader`.


    Attributes:
        variables (dict): A dict of all variables in a created graph
            with a variable name as a key, and a nnabla.Variable as a value.

        inputs (dict): All input variables.

        outputs (dict): All output variables.

    '''

    def _get_variable_or_create(self, v):

        if v.variable is not None:
            return v.variable

        pvar = v.proto
        name = pvar.name
        shape = list(pvar.shape.dim)
        if shape[0] < 0:
            shape[0] = self.batch_size
        shape = tuple(shape)
        assert np.all(np.array(shape) >
                      0), "Shape must be positive. Given {}.".format(shape)

        # The variable is a parameter, then get from parameter registry.
        if pvar.type == 'Parameter':
            try:
                param = get_parameter(name)
                assert param is not None, \
                    "A parameter `{}` is not found.".format(name)
            except:
                import sys
                import traceback
                raise ValueError(
                    'An error occurs during creation of a variable `{}` as a'
                    ' parameter variable. The error was:\n----\n{}\n----\n'
                    'The parameters registered was {}'.format(
                        name, traceback.format_exc(),
                        '\n'.join(
                            list(nn.get_parameters(grad_only=False).keys()))))
            assert shape == param.shape
            v.variable = param
            return param

        # Create a new one and returns.
        var = nn.Variable(shape)
        v.variable = var
        return var

    def _create_inputs(self, inputs):
        input_vars = []
        for i in inputs:
            input_vars.append(self._get_variable_or_create(i))
        return input_vars

    def _create_function(self, f):
        inputs = self._create_inputs(f.inputs)

        function_instance = _create_function(inputs, f.proto, self.batch_size)

        outputs = function_instance(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)

        for o, ovar in zip(f.outputs, outputs):
            o.variable = ovar

    def __init__(self, network_proto, batch_size=None, callbacks=None):
        if batch_size is None:
            batch_size = network_proto.batch_size
        self.batch_size = batch_size

        # Variable proto messages as a dictionary with name as a key
        variables = {v.name: VariableProto(v) for v in network_proto.variable}
        functions = [FunctionProto(f) for f in network_proto.function]

        for f in functions:
            inputs = [variables[name] for name in f.proto.input]
            outputs = [variables[name] for name in f.proto.output]
            f.inputs = inputs
            f.outputs = outputs

        # Filter isolated variables
        variables = {k: v for k, v in variables.items(
        ) if v.parent is not None or v.num_referrers > 0}

        # Get outputs
        outputs = [v for v in variables.values() if v.num_referrers == 0]

        # Build computation graph
        visit_forward(outputs, self._create_function)

        # Get input variables
        self.variables = {v.proto.name: v.variable for v in variables.values()}
        inputs = [v for v in variables.values(
        ) if v.parent is None and v.proto.type != "Parameter"]
        self.inputs = {i.proto.name: i.variable for i in inputs}
        self.outputs = {o.proto.name: o.variable for o in outputs}


class NnpLoader(object):
    '''An NNP file loader.

    Example:

        .. code-block:: python

            from nnabla.utils.nnp_graph import NnpLoader

            # Read a .nnp file.
            nnp = NnpLoader('/path/to/nnp.nnp')
            # Assume a graph `graph_a` is in the nnp file.
            net = nnp.get_network(network_name, batch_size=1)
            # `x` is an input of the graph.
            x = net.inputs['x']
            # 'y' is an outputs of the graph.
            y = net.outputs['y']
            # Set random data as input and perform forward prop.
            x.d = np.random.randn(x.shape)
            y.forward(clear_buffer=True)
            print('output:', y.d)

    '''

    def __init__(self, filepath):
        _, ext = os.path.splitext(filepath)

        if ext == ".nnp":
            proto = _load_nnp_to_proto(filepath)
        else:
            raise NotImplementedError(
                "Currently extension of file for loading must be ['.nnp', ]")
        self.proto = proto
        self.network_dict = {
            network.name: network for network in proto.network}

    def get_network_names(self):
        '''Returns network names available.
        '''
        return list(self.network_dict.keys())

    def get_network(self, name, batch_size=None, callbacks=None):
        '''Create a variable graph given  network by name

        Returns: NnpNetwork

        '''
        return NnpNetwork(self.network_dict[name], batch_size, callbacks)
