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

from __future__ import print_function

from collections import OrderedDict
import os
import weakref
import numpy as np

import nnabla as nn
import nnabla.function as F
from nnabla.utils import nnabla_pb2
from nnabla.parameter import get_parameter, set_parameter
from nnabla.utils.load_function import _create_function_instance
from nnabla.initializer import (
    NormalInitializer, UniformInitializer, ConstantInitializer, RangeInitializer,
    calc_normal_std_he_forward, calc_normal_std_he_backward, calc_normal_std_glorot, calc_uniform_lim_glorot)
from nnabla import logger
from nnabla.parameter import get_parameter_or_create


def _create_variable(v, name, shape, rng):
    # Create and initialize variables
    class Variable:
        pass

    parameter = v.type == "Parameter"
    variable_instance = None
    if parameter:
        if v.initializer.type == 'Normal':
            initializer = NormalInitializer(v.initializer.multiplier, rng=rng)
        elif v.initializer.type == 'NormalAffineHe' or v.initializer.type == 'NormalAffineHeForward':
            initializer = (lambda shape: NormalInitializer(calc_normal_std_he_forward(
                shape[0], np.prod(shape[1:])), rng=rng)(shape) * v.initializer.multiplier)
        elif v.initializer.type == 'NormalAffineHeBackward':
            initializer = (lambda shape: NormalInitializer(calc_normal_std_he_backward(
                shape[0], np.prod(shape[1:])), rng=rng)(shape) * v.initializer.multiplier)
        elif v.initializer.type == 'NormalAffineGlorot':
            initializer = (lambda shape: NormalInitializer(calc_normal_std_glorot(
                shape[0], np.prod(shape[1:])), rng=rng)(shape) * v.initializer.multiplier)
        elif v.initializer.type == 'NormalConvolutionHe' or v.initializer.type == 'NormalConvolutionHeForward':
            initializer = (lambda shape: NormalInitializer(calc_normal_std_he_forward(
                shape[-3], shape[0], kernel=shape[-2:]), rng=rng)(shape) * v.initializer.multiplier)
        elif v.initializer.type == 'NormalConvolutionHeBackward':
            initializer = (lambda shape: NormalInitializer(calc_normal_std_he_backward(
                shape[-3], shape[0], kernel=shape[-2:]), rng=rng)(shape) * v.initializer.multiplier)
        elif v.initializer.type == 'NormalConvolutionGlorot':
            initializer = (lambda shape: NormalInitializer(calc_normal_std_glorot(
                shape[-3], shape[0], kernel=shape[-2:]), rng=rng)(shape) * v.initializer.multiplier)
        elif v.initializer.type == 'NormalCLConvHe' or v.initializer.type == 'NormalCLConvHeForward':
            initializer = (lambda shape: NormalInitializer(calc_normal_std_he_forward(
                shape[-1], shape[0], kernel=shape[1:3]), rng=rng)(shape) * v.initializer.multiplier)
        elif v.initializer.type == 'NormalCLConvHeBackward':
            initializer = (lambda shape: NormalInitializer(calc_normal_std_he_backward(
                shape[-1], shape[0], kernel=shape[1:3]), rng=rng)(shape) * v.initializer.multiplier)
        elif v.initializer.type == 'NormalCLConvGlorot':
            initializer = (lambda shape: NormalInitializer(calc_normal_std_glorot(
                shape[-1], shape[0], kernel=shape[1:3]), rng=rng)(shape) * v.initializer.multiplier)
        elif v.initializer.type == 'Uniform':
            initializer = UniformInitializer(
                lim=[-v.initializer.multiplier, v.initializer.multiplier], rng=rng)
        elif v.initializer.type == 'UniformAffineGlorot':
            initializer = (lambda shape: UniformInitializer(calc_uniform_lim_glorot(
                shape[0], np.prod(shape[1:])), rng=rng)(shape) * v.initializer.multiplier)
        elif v.initializer.type == 'UniformConvolutionGlorot':
            initializer = (lambda shape: UniformInitializer(calc_uniform_lim_glorot(
                shape[-3], shape[0], kernel=shape[-2:]), rng=rng)(shape) * v.initializer.multiplier)
        elif v.initializer.type == 'UniformCLConvGlorot':
            initializer = (lambda shape: UniformInitializer(calc_uniform_lim_glorot(
                shape[-1], shape[0], kernel=shape[1:3]), rng=rng)(shape) * v.initializer.multiplier)
        elif v.initializer.type == 'Range':
            initializer = (lambda shape: RangeInitializer(0, 1)
                           (shape) * v.initializer.multiplier)
        elif v.initializer.type == 'Constant':
            initializer = ConstantInitializer(value=v.initializer.multiplier)
        else:
            initializer = None
        print("create parameter: {}".format(name))
        variable_instance = get_parameter_or_create(name, shape, initializer)
    else:
        # create empty variable, memory will be allocated in network.setup()
        # after network optimization
        variable_instance = nn.Variable()

    variable = Variable()
    variable.name = name
    variable.parameter = parameter
    variable.shape = shape
    variable.variable_instance = variable_instance

    return variable


def resolve_reshape_params(inputs, function_proto, batch_size):
    '''Resolve shape parameter and returns shape.
    '''
    f = function_proto  # alias

    # There are 2 exceptional cases.
    # A. Negative dimension is batch dimension
    # A-1. Detect multiple negative dimensions (not allowed).
    negative_count = 0
    for d in f.reshape_param.shape.dim:
        if d < 0:
            negative_count += 1
    if negative_count > 1:
        raise ValueError('Reshape: shape has multiple negative number.')

    # A-2. Fill negative dimensions with batch size.
    shape = tuple(
        [d if d >= 0 else batch_size for d in f.reshape_param.shape.dim])

    # B. Console omits batch dimensions (the first dimension) during saving.
    # B-1. Fill with batch size if shapes don't match.
    if np.prod(shape) != np.prod(inputs[0].shape):
        shape = (batch_size,) + shape
        if np.prod(shape) != np.prod(inputs[0].shape):
            raise ValueError('Shape after filling batch dimension does not match the input shape. prod({}) != prod({})'.format(
                shape, inputs[0].shape))
    return shape


def resolve_broadcast_params(inputs, function_proto, batch_size):
    '''Resolve shape parameter and returns shape.
    '''
    f = function_proto  # alias

    # A. Detect multiple negative dimensions (not allowed).
    negative_count = 0
    for d in f.broadcast_param.shape.dim:
        if d < 0:
            negative_count += 1
    if negative_count > 1:
        raise ValueError('Reshape: shape has multiple negative number.')

    # B. Fill negative dimensions with batch size.
    shape = tuple(
        [d if d >= 0 else batch_size for d in f.broadcast_param.shape.dim])
    return shape


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


def _load_nntxt_to_proto(nntxt_path):
    import google.protobuf.text_format as text_format
    proto = nnabla_pb2.NNablaProtoBuf()

    with open(nntxt_path, "rt") as f:
        text_format.Merge(f.read(), proto)

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


class VariableProto(object):

    def __init__(self, v):
        self.proto = v
        self.parent = None
        self._referrers = {}
        self.variable = None
        self.need_grad = None
        self.stop = False

    @property
    def name(self):
        return self.proto.name

    def add_referrer(self, f):
        assert isinstance(f, FunctionProto)
        self._referrers[f.name] = weakref.ref(f)

    @property
    def referrers(self):
        referrers = {k: r()
                     for k, r in self._referrers.items() if r() is not None}
        return referrers

    @property
    def num_referrers(self):
        return len(self._referrers)

    def delete_referrer(self, f):
        del self._referrers[f.name]

    def rewire_on(self, var):
        parent = var.parent
        if self.parent is not None:
            self.parent.disable()
        self.parent = parent
        var.parent = None

        # Replace var with self for var.parent.outputs
        if parent is None:
            return
        new_outputs = []
        for o in parent.outputs:
            new = o
            if o is var:
                new = self
            new_outputs.append(new)
        self.parent.outputs = new_outputs


class FunctionProto(object):
    def __init__(self, proto):
        self.proto = proto
        self._inputs = []
        self._outputs = []
        self.function = None
        self._disabled = False

    @property
    def name(self):
        return self.proto.name

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

    def disable(self):
        self._disabled = True
        for i in self.inputs:
            i.delete_referrer(self)  # Forget me.
        for o in self.outputs:
            o.parent = None  # Forget me.
        self._inputs = []
        self._outputs = []

    @property
    def disabled(self):
        return self._disabled


def visit_forward(variables, callback, fclosed=None):
    if fclosed is None:
        fclosed = set()
    stop = False
    for v in variables:
        stop |= v.stop
        f = v.parent
        if f is None:
            continue
        if f in fclosed:
            continue
        fclosed.add(f)
        stop_f = visit_forward(f.inputs, callback, fclosed)
        # Send stop signal to child function if any of predecessors of
        # `variables` has the stop attribute.
        stop |= stop_f
        if stop_f:
            print('Skip {} by stop signal.'.format(f.name))
            f.disable()
            continue
        callback(f)
        print(f.name)
    return stop


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

    def _get_variable_or_create(self, v, callback, current_scope):

        if v.variable is not None:
            return v.variable

        v = callback._apply_generate_variable(v)

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

        if pvar.type != 'Parameter':
            # Create a new variable and returns.
            var = nn.Variable(shape)
            v.variable = var
            var.name = name
            return var

        # Trying to load the parameter from the global scope.
        try:
            with nn.parameter_scope('', current_scope):
                param = get_parameter(name)

            if param is not None:
                assert shape == param.shape
                param = param.get_unlinked_variable(need_grad=v.need_grad)
                v.variable = param
                param.name = name
                return param

            # Parameter does not exist in the global scope.
            # Then try to load the parameter from .nnp file.
            callback.verbose(
                'Loading parameter `{}` from .nnp.'.format(name))
            param = get_parameter(name)

            if param is None:
                logger.info(
                    'Parameter `{}` is not found. Initializing.'.format(name))
                tmp = _create_variable(pvar, name, shape, self.rng)
                param = tmp.variable_instance

            # Register the parameter to the current (global) scope.
            with nn.parameter_scope('', current_scope):
                set_parameter(name, param)

        except:
            import traceback
            raise ValueError(
                'An error occurs during creation of a variable `{}` as a'
                ' parameter variable. The error was:\n----\n{}\n----\n'
                'The parameters registered was {}'.format(
                    name, traceback.format_exc(),
                    '\n'.join(
                        list(nn.get_parameters(grad_only=False).keys()))))

        assert shape == param.shape
        param = param.get_unlinked_variable(need_grad=v.need_grad)
        v.variable = param
        param.name = name
        return param

    def _create_inputs(self, inputs, callback, current_scope):
        input_vars = []
        for i in inputs:
            input_vars.append(self._get_variable_or_create(
                i, callback, current_scope))
        return input_vars

    def _create_function(self, f, callback, current_scope):
        callback.verbose2('Creating function {}: {} --> {}.'.format(f.name,
                                                                    [i.name for i in f.inputs], [i.name for i in f.outputs]))

        f = callback._apply_generate_function_by_type(f)
        f = callback._apply_generate_function_by_name(f)
        inputs = self._create_inputs(f.inputs, callback, current_scope)
        function_instance = _create_function(inputs, f.proto, self.batch_size)

        outputs = function_instance(
            *inputs, n_outputs=len(f.outputs), auto_forward=nn.get_auto_forward())
        if not isinstance(outputs, tuple):
            outputs = (outputs,)

        for o, ovar in zip(f.outputs, outputs):
            o.variable = ovar
            ovar.name = o.name

    def _filter_variables(self, variables):
        # Filter isolated variables
        variables = {k: v for k, v in variables.items(
        ) if v.parent is not None or v.num_referrers > 0}
        return variables

    def _get_inputs(self, variables):
        inputs = [v for v in variables.values(
        ) if v.parent is None and v.proto.type != "Parameter"]
        return inputs

    def _get_outputs(self, variables):
        # Get outputs
        outputs = [v for v in variables.values() if v.num_referrers == 0]
        return outputs

    def _functions_in_forward_order(self, variables):
        variables = self._filter_variables(variables)
        outputs = self._get_outputs(variables)
        function_order = []

        def _function_order(f):
            function_order.append(f)
        visit_forward(outputs, _function_order)
        for f in function_order:
            yield f

    def __init__(self, network_proto, scope, batch_size=None, rng=None, callback=None):

        if batch_size is None:
            batch_size = network_proto.batch_size
        self.batch_size = batch_size
        if rng is None:
            rng = np.random.RandomState(1223)
        self.rng = rng

        if callback is None:
            callback = NnpNetworkPass()  # No pass

        # Variable proto messages as a dictionary with name as a key
        variables = {v.name: VariableProto(v) for v in network_proto.variable}
        functions = [FunctionProto(f) for f in network_proto.function]

        for f in functions:
            inputs = [variables[name] for name in f.proto.input]
            outputs = [variables[name] for name in f.proto.output]
            f.inputs = inputs
            f.outputs = outputs

        # Apply function passes
        for f in self._functions_in_forward_order(variables):
            if f.disabled:
                continue
            callback._apply_function_pass_by_type(f, variables, scope)
            callback._apply_function_pass_by_name(f, variables, scope)

        # Apply stop-at.
        for f in self._functions_in_forward_order(variables):
            # callback.verbose2('Applying stop-at for inputs of {}.'.format(f.name))
            callback._apply_use_up_to(f.inputs)

        # Build computation graph
        num_ops = 0
        current_scope = nn.get_current_parameter_scope()
        with nn.parameter_scope('', scope):
            for f in self._functions_in_forward_order(variables):
                self._create_function(f, callback, current_scope)
                # print(f.name)
                num_ops += 1
        callback.verbose2('Created {} functions.'.format(num_ops))

        variables = self._filter_variables(variables)
        inputs = self._get_inputs(variables)
        outputs = self._get_outputs(variables)

        # Get input variables
        self.variables = {v.name: v.variable for v in variables.values()}
        self.inputs = {i.name: i.variable for i in inputs}
        self.outputs = {o.name: o.variable for o in outputs}


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

    def __init__(self, filepath, scope=None):
        # OrderedDict maintains loaded parameters from nnp files.
        # The loaded parameters will be copied to the current
        # scope when get_network is called.
        if scope is None:
            scope = OrderedDict()
        self._params = scope

        _, ext = os.path.splitext(filepath)

        if ext == ".nnp":
            # Load parameters to self._params rather than
            # loading to global current scope.
            with nn.parameter_scope('', self._params):
                proto = _load_nnp_to_proto(filepath)
        elif ext in ('.nntxt', '.prototxt'):
            proto = _load_nntxt_to_proto(filepath)
        else:
            raise NotImplementedError(
                "Currently extension of file for loading must be ['.nnp', '.nntxt']")
        self.proto = proto
        self.network_dict = {
            network.name: network for network in proto.network}

    def get_network_names(self):
        '''Returns network names available.
        '''
        return list(self.network_dict.keys())

    def get_network(self, name, batch_size=None, callback=None):
        '''Create a variable graph given  network by name

        Returns: NnpNetwork

        '''
        network_proto = nnabla_pb2.Network()
        network_proto.CopyFrom(self.network_dict[name])
        return NnpNetwork(network_proto, self._params, batch_size, callback=callback)


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
            pool_shape = f.inputs[0].variable.shape[2:]
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
            pool_shape = f.inputs[0].variable.shape[2:]
            p = f.proto.average_pooling_param
            if tuple(p.kernel.dim[:]) != pool_shape or tuple(p.stride.dim[:]) != pool_shape:
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
