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


def _create_function(ctx, inputs, funtion_proto, batch_size):
    # todo: arrange weight name for NNC

    if funtion_proto.type == "Reshape":  # if batch_size = -1, something wrong?
        reshape_shape = resolve_reshape_params(
            inputs, funtion_proto, batch_size)
        function_instance = F.Reshape(
            ctx, shape=reshape_shape, inplace=funtion_proto.reshape_param.inplace)
    elif funtion_proto.type == 'Broadcast':
        shape = resolve_broadcast_params(inputs, funtion_proto, batch_size)
        function_instance = F.Broadcast(ctx, shape=shape)
    elif funtion_proto.type == "RepeatStart":
        raise NotImplementedError("Repeat not supported.")
        function_instance = F.Identity(ctx)
    elif funtion_proto.type == "RepeatEnd":
        raise NotImplementedError("Repeat not supported.")
        function_instance = F.Identity(ctx)
    elif funtion_proto.type == "RecurrentOutput":
        raise NotImplementedError("Recurrent not supported.")
        function_instance = F.Stack(
            ctx, axis=funtion_proto.recurrent_param.axis)
    elif funtion_proto.type == "RecurrentInput":
        raise NotImplementedError("Recurrent not supported.")
        function_instance = F.Split(
            ctx, axis=funtion_proto.recurrent_param.axis)
    elif funtion_proto.type == "Delay":
        raise NotImplementedError("Recurrent not supported.")
        function_instance = F.Identity(ctx)
    else:
        function_instance = _create_function_instance(ctx, funtion_proto)

    return function_instance


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

    def _get_variable_or_create(self, name, shape, var_type):

        # The variable is a parameter, then get from parameter registry.
        if var_type == 'Parameter':
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
            return param

        # Returns if variable is already created.
        try:
            var, count = self.vseen[name]
            count[0] += 1
        except:
            # Create a new one and returns.
            var = nn.Variable(shape)
            self.vseen[name] = (var, [1])
            return var

        # Found already created variable.
        assert var.shape == shape
        return var

    def _create_inputs(self, input_names):
        inputs = []
        for name in input_names:
            pvar = self.variable_proto[name]
            shape = list(pvar.shape.dim)
            if shape[0] < 0:
                shape[0] = self.batch_size
            shape = tuple(shape)
            assert np.all(np.array(shape) >
                          0), "Shape must be positive. Given {}.".format(shape)
            var = self._get_variable_or_create(name, shape, pvar.type)
            inputs.append(var)
        return inputs

    def _create_function(self, function_proto):
        inputs = self._create_inputs(function_proto.input)

        function_instance = _create_function(
            nn.get_current_context(), inputs, function_proto, self.batch_size)

        outputs = function_instance(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)

        for i, name in enumerate(function_proto.output):
            try:
                var, _ = self.vseen[name]
            except:
                self.vseen[name] = (outputs[i], [0])
                continue
            var.rewire_on(outputs[i])

    def __init__(self, network_proto, batch_size=None):
        if batch_size is None:
            batch_size = network_proto.batch_size
        self.batch_size = batch_size

        # Variable proto messages as a dictionary with name as a key
        self.variable_proto = {v.name: v for v in network_proto.variable}
        self.vseen = {}

        # Create function graph
        for function_proto in network_proto.function:
            self._create_function(function_proto)

        # Get input variables
        self.variables = {name: v for name, (v, c) in self.vseen.items()}
        self.inputs = {name: v for name,
                       (v, c) in self.vseen.items() if v.parent is None}
        self.outputs = {name: v for name,
                        (v, c) in self.vseen.items() if c[0] == 0}


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
        '''Returns network names availble.
        '''
        return list(self.network_dict.keys())

    def get_network(self, name, batch_size=None):
        '''Create a variable graph given  network by name

        Returns: NnpNetwork

        '''
        return NnpNetwork(self.network_dict[name], batch_size)
