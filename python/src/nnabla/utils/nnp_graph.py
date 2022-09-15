# Copyright 2018,2019,2020,2021 Sony Corporation.
# Copyright 2021 Sony Group Corporation.
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

from __future__ import print_function

import itertools
from collections import OrderedDict

import nnabla as nn
import numpy as np


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

    def __init__(self, proto_network, batch_size, callback):
        proto_network = proto_network.expand_loop_control()
        self.proto_network = proto_network.promote(callback)
        self.proto_network(batch_size=batch_size)
        for k, v in itertools.chain(
                self.proto_network.variables.items(), self.proto_network.parameters.items()):
            if v.variable_instance is not None:
                v.variable_instance.name = k
        self._inputs = {
            i: self.proto_network.variables[i].variable_instance
            for i in self.proto_network.inputs
        }
        self._outputs = {
            i: self.proto_network.variables[i].variable_instance
            for i in self.proto_network.outputs
        }
        self._variables = {
            k: v.variable_instance
            for k, v in itertools.chain(
                self.proto_network.variables.items(), self.proto_network.parameters.items())
        }

        # publish network's parameters to current parameter scope
        # like original implementation.
        with nn.parameter_scope('', nn.get_current_parameter_scope()):
            for k, v in self.proto_network.parameters.items():
                nn.parameter.set_parameter(k, v.variable_instance)

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    @property
    def variables(self):
        return self._variables


class NnpLoader(object):
    '''An NNP file loader.

    Args:
        filepath : file-like object or filepath.
        extension: if filepath is file-like object, extension is one of ".nnp", ".nntxt", ".prototxt".

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
            x.d = np.random.randn(*x.shape)
            y.forward(clear_buffer=True)
            print('output:', y.d)

    '''

    def __init__(self, filepath, scope=None, extension=".nntxt"):
        # OrderedDict maintains loaded parameters from nnp files.
        # The loaded parameters will be copied to the current
        # scope when get_network is called.
        self._params = scope if scope else OrderedDict()
        self.g = nn.graph_def.load(
            filepath, parameter_scope=self._params, rng=np.random.RandomState(1223), extension=extension)
        self.network_dict = {
            name: pn for name, pn in self.g.networks.items()
        }

    def get_network_names(self):
        '''Returns network names available.
        '''
        return list(self.network_dict.keys())

    def get_network(self, name, batch_size=None, callback=None):
        '''Create a variable graph given  network by name

        Returns: NnpNetwork

        '''
        return NnpNetwork(self.network_dict[name], batch_size, callback=callback)


class NnpNetworkPass(object):

    def _no_verbose(self, *a, **kw):
        pass

    def _verbose(self, *a, **kw):
        print(*a, **kw)

    def __init__(self, verbose=0):
        self._variable_callbacks = {}
        self._function_callbacks_by_name = {}
        self._function_callbacks_by_type = {}
        self._passes_by_name = {}
        self._passes_by_type = {}
        self._fix_parameters = False
        self._use_up_to_variables = set()

        self.verbose = self._no_verbose
        self.verbose2 = self._no_verbose
        if verbose:
            self.verbose = self._verbose
        if verbose > 1:
            self.verbose2 = self._verbose

    def on_function_pass_by_name(self, name):
        def _on_function_pass_by_name(callback):
            def _callback(f, variables, param_scope):
                return callback(f, variables, param_scope)
            self._passes_by_name[name] = _callback
            return _callback
        return _on_function_pass_by_name

    def on_function_pass_by_type(self, name):
        def _on_function_pass_by_type(callback):
            def _callback(f, variables, param_scope):
                return callback(f, variables, param_scope)
            self._passes_by_name[name] = _callback
            return _callback
        return _on_function_pass_by_type

    def on_generate_variable(self, name):
        def _on_generate_variable(callback):
            def _callback(v):
                return callback(v)
            self._variable_callbacks[name] = _callback
            return _callback
        return _on_generate_variable

    def on_generate_function_by_name(self, name):
        def _on_generate_function_by_name(callback):
            def _callback(v):
                return callback(v)
            self._function_callbacks_by_name[name] = _callback
            return _callback
        return _on_generate_function_by_name

    def on_generate_function_by_type(self, name):
        def _on_generate_function_by_type(callback):
            def _callback(v):
                return callback(v)
            self._function_callbacks_by_type[name] = _callback
            return _callback
        return _on_generate_function_by_type

    def drop_function(self, *names):
        def callback(f, variables, param_scope):
            self.verbose('Pass: Deleting {}.'.format(f.name))
            f.disable()

        for name in names:
            self.on_function_pass_by_name(name)(callback)

    def fix_parameters(self):
        self._fix_parameters = True

    def use_up_to(self, *names):
        self._use_up_to_variables.update(set(names))

    def remove_and_rewire(self, name, i=0, o=0):
        @self.on_function_pass_by_name(name)
        def on_dr(f, variables, param_scope):
            fi = f.inputs[i]
            fo = f.outputs[o]
            self.verbose('Removing {} and rewire input={} and output={}.'.format(
                f.name, fi.name, fo.name))
            fo.rewire_on(fi)
            # Use input name
            fo.proto.name = fi.name

    def set_variable(self, name, input_var):
        @self.on_generate_variable(name)
        def on_input_x(v):
            self.verbose('Replace {} by {}.'.format(name, input_var))
            v.proto.shape.dim[:] = input_var.shape
            v.variable = input_var
            input_var.name = v.name
            return v

    def force_average_pooling_global(self, name, by_type=False):
        dec = self.on_generate_function_by_name
        if by_type:
            dec = self.on_generate_function_by_type

        @dec(name)
        def on_avgpool(f):
            pool_shape = f.inputs[0].proto.shape.dim[2:]
            self.verbose('Change strides of {} to {}.'.format(
                f.name, pool_shape))
            p = f.proto.average_pooling_param
            p.kernel.dim[:] = pool_shape
            p.stride.dim[:] = pool_shape
            return f

    def check_average_pooling_global(self, name, by_type=False):
        dec = self.on_generate_function_by_name
        if by_type:
            dec = self.on_generate_function_by_type

        @dec(name)
        def on_avgpool_check(f):
            pool_shape = f.inputs[0].proto.shape.dim[2:]
            p = f.proto.average_pooling_param
            if p.kernel.dim[:] != pool_shape or p.stride.dim[:] != pool_shape:
                raise ValueError(
                    'Stride configuration of average pooling is not for global pooling.'
                    ' Given Image shape is {}, whereas pooling window size is {} and its stride is {}.'
                    ' Consider using force_global_pooling=True'.format(
                        pool_shape, p.kernel.dim[:], p.stride.dim[:]))
            return f

    def set_batch_normalization_batch_stat_all(self, batch_stat):
        @self.on_generate_function_by_type('BatchNormalization')
        def on_bn(f):
            self.verbose('Setting batch_stat={} at {}.'.format(
                batch_stat, f.name))
            p = f.proto.batch_normalization_param
            p.batch_stat = batch_stat
            return f

    def _apply_function_pass_by_name(self, f, variables, param_scope):
        if f.name not in self._passes_by_name:
            return f
        return self._passes_by_name[f.name](f, variables, param_scope)

    def _apply_function_pass_by_type(self, f, variables, param_scope):
        if f.proto.type not in self._passes_by_type:
            return f
        return self._passes_by_type[f.proto.type](f, variables, param_scope)

    def _apply_generate_variable(self, v):
        if v.name in self._variable_callbacks:
            v = self._variable_callbacks[v.name](v)
        if self._fix_parameters:
            v.need_grad = False
        return v

    def _apply_generate_function_by_name(self, f):
        if f.name not in self._function_callbacks_by_name:
            return f
        return self._function_callbacks_by_name[f.name](f)

    def _apply_generate_function_by_type(self, f):
        if f.proto.type not in self._function_callbacks_by_type:
            return f
        return self._function_callbacks_by_type[f.proto.type](f)

    def _apply_use_up_to(self, variables):
        for v in variables:
            if v.name in self._use_up_to_variables:
                self.verbose('Stopping at {}.'.format(v.name))
                v.stop = True
