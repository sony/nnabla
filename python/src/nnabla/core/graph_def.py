# Copyright 2020,2021 Sony Corporation.
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

"""
In NNabla, `graph_def` represents a kind of representation of a computation graph which is special designed for
storage optimization and format converter.

A computation graph can be defined by the call of NNabla functions. Such computation graph has instantiated the input
and output variables of the functions, inherent topology has been established for forward or backward computation.
But for persistence of such graph, another abstract representation, so-called protobuf graph(or network), as abbreviation
- proto graph is used normally. In this graph, only the information being necessary for persistence are kept, the
information only used for computation will be dropped.

`graph_def` provides a group of functions and classes, tends to facilitate user creates protobuf network from their
computation graph, and saving and restoring their neural network from a persistent protobuf network representation.

"""

import weakref
from collections import OrderedDict
from contextlib import contextmanager
from functools import reduce
from itertools import chain

import nnabla as nn
import numpy as np

__graph_members__ = ['inputs', 'outputs',
                     'parameters', 'functions', 'variables']
__recurrent_functions__ = ['RecurrentInput', 'RecurrentOutput']
__repeat_functions__ = ['RepeatStart', 'RepeatEnd']
__delay_function__ = ['Delay']
__branch_function__ = ["RepeatStart"] + __delay_function__
__loop_control_functions__ = __recurrent_functions__ + \
    __repeat_functions__ + __delay_function__


class RepeatInfo:
    def __init__(self, id, times):
        self.id = id
        self.times = times


class LoopControlFunction:
    def __init__(self):
        self.arguments = {}


def load(filename, batch_size=None, exclude_parameter=False, parameter_only=False,
         extension='.nntxt', parameter_scope=None, rng=None):
    """load
    Load network from files

    Args:
        filename (str or list or file-like object): Filename string ,list of filenames or file-like object.
        batch_size (int): The batch size expected to be set.
        exclude_parameter (bool): If True, only load model, not load parameters of this model. Default is False.
        parameter_only (bool): If True, only load model parameters. Default is False.
        extension (str): This parameter is needed when filename is a file-like object. Default is `.nntxt`.
        parameter_scope (OrderedDict): User may provide a user owned parameter scope. If this parameter is not provided,
                                       loaded parameters will be created in created proto_graph's parameter_scope. This
                                       parameter_scope is default initialized with empty dictionary.
        rng (random state): User may specify random state, which cause parameters are initialized by determined random seed.

    Returns:
        ProtoGraph:
            A ProtoGraph object, in which, there are one or multiple ProtoNetwork objects.

    Example:
        The following example loads a model and generate the output variable
        through this model:

        .. code-block:: python

            import nnabla as nn
            import nnabla.functions as F
            import nnabla.parametric_functions as PF

            def fusion_net(x):
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

            x = nn.ProtoVariable((64, 3, 32, 32))
            y = fusion_net(x)
            g = nn.graph_def.get_default_graph()  # Get generated graph_def
            g.save("fusion_net.nnp")
            ...
            g = nn.graph_def.load("fusion_net.nnp")
            x = nn.Variable((64, 3, 32, 32))
            x.d = ... # user provided input data for this graph
            y = g(x) # create computation graph by passing in nn.Variable()
            y.forward() # calculate output by this graph
            ...

            # You may use your special context(e.g. cuda context)
            with context_scope(ctx):
               y = g(x) # create computation graph representation with specified backend context.
               y.forward() # forward using specified backend

    """
    from nnabla.utils import nnabla_pb2
    from nnabla.utils.get_file_handle import get_initial_file_loader, load_files, FileHandlerContext

    ctx = FileHandlerContext()

    ctx.exclude_parameter = exclude_parameter
    ctx.parameter_only = parameter_only
    ctx.proto = nnabla_pb2.NNablaProtoBuf()
    if parameter_scope is None:
        ctx.parameter_scope = OrderedDict()
    else:
        ctx.parameter_scope = parameter_scope
    file_loaders = get_initial_file_loader()

    if rng is None:
        rng = np.random.RandomState(0)

    load_files(ctx, file_loaders, filename, extension)
    g = ProtoGraph.from_proto(ctx.proto,
                              batch_size=batch_size,
                              param_scope=ctx.parameter_scope,
                              rng=rng)
    return g


def save(filename, content, include_parameters=False, variable_batch_size=True, extension='.nnp'):
    """Save network

    Args:
        filename (str or file object): Filename to store information. The file
            extension is used to determine the saving file format.
            ``.nnp``: (Recommended) Creating a zip archive with nntxt (network
            definition etc.) and h5 (parameters).
            ``.nntxt``: Protobuf in text format.
            ``.protobuf``: Protobuf in binary format (unsafe in terms of
             backward compatibility).
        content (list): Currently only ProtoGraph or PhotoNetwork objects are
                       supported.
        include_parameters (bool): Includes parameter into single file. This is
            ignored when the extension of filename is nnp.
        variable_batch_size (bool): Whether or not convert batch size of computation graph to a special
                                    value, so that user may use any other batch_size value when using it.

    Example:
        The following example creates a two inputs and two
        outputs MLP, and save the network structure and the initialized
        parameters.

        .. code-block:: python

            import nnabla as nn
            import nnabla.functions as F
            import nnabla.parametric_functions as PF

            def mlp_module(x0, x1):
                h1_0 = PF.affine(x0, 100, name='affine1_0')
                h1_1 = PF.affine(x1, 100, name='affine1_0')
                h1 = F.tanh(h1_0 + h1_1)
                h2 = F.tanh(PF.affine(h1, 50, name='affine2'))
                y0 = PF.affine(h2, 10, name='affiney_0')
                y1 = PF.affine(h2, 10, name='affiney_1')
                return y0, y1

            with nn.graph_def.graph() as g:
                x0 = nn.ProtoVariable((64, 100))
                x1 = nn.ProtoVariable((64, 100))
                y0, y1 = mlp_module(x0, x1)

            nn.graph_def.save("mlp_net.nnp", [g])

    """
    from nnabla.logger import logger
    from nnabla.utils.get_file_handle import FileHandlerContext, get_default_file_savers, save_files

    def _create_proto(contents, include_params, variable_batch_size):
        params = None
        for g in contents:
            if isinstance(g, ProtoGraph):
                proto = g.as_proto(
                    include_parameter=include_params, variable_batch_size=variable_batch_size)
                params = g.get_parameters()
                break
            if isinstance(g, ProtoNetwork):
                proto = g.owner().as_proto(
                    include_parameter=include_params, networks=[g], variable_batch_size=variable_batch_size)
                params = g.owner().get_parameters()
                break
        return proto, params

    ctx = FileHandlerContext()
    ctx.proto, ctx.parameters = _create_proto(
        content, include_parameters, variable_batch_size)
    file_savers = get_default_file_savers()
    supported = save_files(ctx, file_savers, filename, extension)
    assert supported, "No supported format."
    logger.info("Model file is saved as {}.".format(filename))


def _get_unique_name(names, prefix):
    if prefix in names:
        name = "{}_{}".format(prefix, names[prefix])
        names[prefix] += 1
    else:
        name = prefix
        names[prefix] = 1
    return name


def _create_initializer(v, rng):
    if v.initializer is None:
        return None
    from nnabla.initializer import (
        NormalInitializer, UniformInitializer, ConstantInitializer, RangeInitializer,
        calc_normal_std_he_forward, calc_normal_std_he_backward, calc_normal_std_glorot, calc_uniform_lim_glorot)

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
    return initializer


class ProtoExecutor:
    def __init__(self, proto):
        self.proto = proto

    @staticmethod
    def from_proto(proto):
        executor = ProtoExecutor(proto)
        return executor

    def as_proto(self):
        return self.proto


class ProtoNetwork:
    """
    This class represents a protobuf network, which comes from a corresponding computation graph or restored from
    a saved protobuf network(e.g. `.nnp` file).

    This class describes a neural network by the following members:

     * functions: An OrderedDict of name-value pairs, the value is ProtoFunction object.
     * variables: An OrderedDict of name-value pairs, the value is ProtoVariable object.
     * parameters: An OrderedDict of name-value pairs, the value is ProtoVariable object.
     * inputs:    A string list, which contains the name of input variables of this network.
     * outputs:   A string list, which contains the name of output variables of this network.

     ``variables`` represents activations in networks, ``parameters`` mainly includes weights and all learnable
     parameters. ``functions`` represents functions in networks, the sequence of functions might not equal forward
     sequence. Please use `forward_sequence` to obtain exactly forward function sequence.
    """

    def __init__(self, owner, name=None, batch_size=None):
        members = __graph_members__
        for key in members:
            self.__dict__[key] = OrderedDict()

        self.arguments = {}
        self.func_names = {}
        self.var_names = {}
        self.name = name
        self.owner = weakref.ref(owner)
        self.batch_size = batch_size
        self.repeat_info = {}

    @property
    def current_context(self):
        return self.owner().current_context

    def __getitem__(self, item):
        return self.__dict__[item]

    def resolve_function_parameters(self, function, params):
        inputs = []
        input_index = 0
        for k in function.inputs:
            v = self.variables[k] if k in self.variables else self.parameters[k]
            if isinstance(v, nn.Variable):
                if v.data in params:
                    v_name = params[v.data]
                else:
                    # some module(parameters) are created as local variable
                    v_name = _get_unique_name(
                        self.var_names, "{}_{}".format(function.name, input_index))
                yield k
                p = self.parameters[v_name] = ProtoVariable(
                    v.shape, v_name, v.need_grad, 'Parameter')
                if v.info:
                    p.info = v.info  # No used
                    p.initializer = v.info.initializer
                p.required.append(function.name)
                p.variable_instance = v
                inputs.append(v_name)
            else:
                inputs.append(k)
            input_index += 1
        function.inputs = inputs

    def commit(self, name, networks, params):
        networks[name] = self
        unused_hash_keys = []
        for f in self.functions.values():
            unused_hash_keys += list(
                self.resolve_function_parameters(f, params))
        for k in set(unused_hash_keys):
            del self.parameters[k]
        with nn.parameter_scope('', self.owner().parameter_scope):
            for k, v in self.parameters.items():
                nn.parameter.set_parameter(k, v.variable_instance)
        for k, v in self.variables.items():
            if v.parent not in self.functions:
                v.parent = None
        self.inputs = [k for k, v in self.variables.items()
                       if not v.parent and v.required]
        self.outputs = [k for k, v in self.variables.items()
                        if not v.required and v.parent]
        self.name = name

    def update_actual_shape_of_reshape(self):
        for pf in self.forward_sequence():
            if pf.type == 'Reshape':
                arg_shape = pf.args['shape']
                input = self.variables.get(pf.inputs[0], None)
                if input is None:
                    input = self.parameters.get(pf.inputs[0], None)
                input_shape = input.shape
                shape_infer_index = -1
                rest_size = 1
                for i, s in enumerate(arg_shape):
                    if s < 0:
                        if shape_infer_index >= 0:
                            raise ValueError(
                                'Reshape: shape has multiple negative value.')
                        shape_infer_index = i
                    else:
                        rest_size *= s
                if shape_infer_index >= 0:
                    arg_shape[shape_infer_index] = int(
                        np.prod(input_shape) / rest_size)
                    pf.args['shape'] = arg_shape

    def as_proto(self, **kwargs):
        """This function returns a protobuf data structure, which can be directly
        accessed by the functions in `nnabla.utils.nnabla_pb2`. Thus,
        it allows user further manipulates this protobuf representation, for example,
        performing format converting, or network structure optimization.

        Args:
            variable_batch_size (bool, optional):
                If true, the batch size of the network will be replaced with
                an abstract representation, so that it can be replaced with
                other value in restoring computation graph.

        Returns:
            protobuf:
                A protobuf object.
        """
        from nnabla.utils import nnabla_pb2
        from nnabla.utils.save_function import _create_function_nntxt
        if kwargs:
            self.arguments.update(kwargs)
        variable_batch_size = self.arguments.get('variable_batch_size', True)
        self.update_actual_shape_of_reshape()
        if variable_batch_size:
            from nnabla.core.variable_batch_size import variable_batch_size
            variable_batch_size(self)
        n = nnabla_pb2.Network()
        if self.batch_size:
            n.batch_size = self.batch_size
        n.name = self.name
        variables = OrderedDict(self.variables)
        variables.update(self.parameters)
        functions = self.functions
        for name, variable in variables.items():
            v = n.variable.add()
            v.CopyFrom(variable.proto)

        for name, function in functions.items():
            f = n.function.add()
            _create_function_nntxt(f, name, function)

        return n

    def __call__(self, *args, **kwargs):
        """Generate a computation graph of this protonetwork.

        Args:
            args (tuple of nn.Variables or None)
                 The inputs of network, which can be different from the inputs of original computation graph as long as the network allows.

                 For example,

                 .. code-block:: python

                     import nnabla as nn
                     import numpy as np

                     resnet = nn.graph_def.load("resnet.nnp")
                     x.d = np.random.random(input_shape)
                     y = resnet(x)

                 The variable y corresponding to a computation graph, user may perform forward like:

                 .. code-block:: python

                     y.forward()

                 If user does not provide inputs for this function, because proto network has the memory
                 of network inputs, this function will create corresponding nn.Variable objects as the
                 inputs of this network. This input variables actually are placeholder of input, hence,
                 users need to find these input variables and fill actual value to these placeholders,
                 so that this computation graph is ready for forward or backward.

                 For example,

                 .. code-block:: python

                     g = nn.graph_def.load("resnet.nnp")
                     y = g() # Not provide input variables

                 To feed training or evaluation data to this network, user needs to locate input variable,
                 for example:

                 .. code-block:: python

                    input = g.networks[network_name].variables[input_name].variable_instance
                    input.d = np.random.random(input_shape)

             batch_size (int, optional, default=None):
                 If provided, batch_size will be applied for newly created computation graph. For example,

                 .. code-block:: python

                     g = nn.graph_def.load("my_model.nnp")
                     y = g(batch_size=32)

                In this sample, `batch_size` will be used to create a computation graph with specified batch size.
                Supposed `x` is the input of network, its original shape is (1, 3, 32, 32), then the actual computation
                graph will be (32, 3, 32, 32).
        """
        ctx = kwargs.get("ctx", None)
        if ctx:
            self.owner().current_context = ctx
        with nn.parameter_scope('', self.owner().parameter_scope):
            input_proto_variables = [self.variables[k]
                                     if k in self.variables
                                     else self.parameters[k]
                                     for k in self.inputs]
            batch_size = kwargs.get('batch_size', None)
            batch_size = batch_size if batch_size is not None else self.batch_size
            if len(args) == 0:
                for pv in input_proto_variables:
                    pv_shape = tuple(
                        [d if d >= 1 else batch_size for d in pv.shape])
                    if callable(pv.initializer):
                        pv_shared = self.owner().global_variables.get(pv.name, None)
                        if pv_shared:
                            pv.variable_instance = pv_shared
                        else:
                            pv.variable_instance = nn.Variable.from_numpy_array(
                                pv.initializer(shape=pv_shape), need_grad=True)
                            self.owner(
                            ).global_variables[pv.name] = pv.variable_instance
                    elif pv.variable_instance is None:
                        pv.variable_instance = nn.Variable(pv_shape)
            else:
                for pv, v in zip(input_proto_variables, args):
                    # pv_shape = tuple(
                    #     [d if d >= 1 else batch_size for d in pv.shape])
                    # We loose this restrict so that user may try another shape inputs.
                    # if pv_shape != tuple(v.shape):
                    #     raise ValueError("variable {} {} != {}.".format(
                    #         pv.name, pv_shape, v.shape))
                    pv.variable_instance = v

            self.execute_on_proto(
                lambda pf: pf.graph_call(batch_size=batch_size))
            outputs = [self.variables[k].variable_instance
                       for k in self.outputs]
            if len(outputs) == 1:
                return outputs[0]
            return tuple(outputs)

    def expand_loop_control(self):
        """ This function expand ``loop control`` statement and generate a new
        proto network object without ``loop control`` statement. ``loop control``
        statement cannot be created by python code, it can be only created by interactive
        neural network design tool. The following briefly introduce its specification:

        * As for variable,
           In nntxt, if the variable includes a field repeat_id, it means that
           this variable is in surround with a loop control structure.
           A renaming rule is applied if expanding this network.
           The variable name will be added a postfix, like:

           * For old style, e.g.:

           .. code-block:: none

                BatchNormalization_6/bn/mean --> BatchNormalization_6/bn/mean_RepeatStart[0]
                                                                                 ^        ^  repeat_time
                                                                              repeat_id[index]

                original_name --> original_name + << _%repeat_id%[%repeat_time%],  for each in repeat_id >>


           * For new style, e.g.:

           .. code-block:: none

                BatchNormalization_6{RepeatStart}/bn/mean --> BatchNormalization_6[0]/bn/mean_RepeatStart
                                                                                   ^
                                                                              repeat_time
                original_name --> original_name + << [%repeat_time%],  for each in repeat_id >>


        * As for RepeatStart, RepeatEnd
            The functions or variables nodes between these 2 layers will be repeated.
            Expanding will create times of functions and variables, and connected them each other.

        * As for RecurrentInput,
            Axis of RecurrentParam points out which axis will be split-ed. And each branch will duplicated
            the functions and variables with this repeat_id. This layer works like a split function.

        * As for RecurrentOutput,
            RecurrentOutput merge multiple branches into one output, looks like a stack function.

        * As for Delay
            First time, the output is its input[1], after that, the output is its input[0]
        """
        import itertools
        if len(self.repeat_info) == 0:
            return self.clone()
        proto_network = ProtoNetwork(self.owner(), self.name, self.batch_size)
        # Detach from owner by using a hard-reference.
        proto_network.owner_ = self.owner()
        variable_names = set()
        for pv in itertools.chain(self.variables.values(), self.parameters.values()):
            for variable_index in itertools.product(*map(tuple, map(range, [self.repeat_info[rid] for rid in pv.repeat_id]))):
                name = pv.name
                for index, i in enumerate(variable_index):
                    r_id = pv.repeat_id[index]
                    pv_repeat_id = '{' + r_id + '}'
                    if pv_repeat_id in name:
                        name = name.replace(pv_repeat_id, '[' + str(i) + ']')
                    else:
                        name += '_{}[{}]'.format(r_id, i)
                n_pv = pv.clone()
                n_pv.name = name
                n_pv.parent = None
                n_pv.required = []
                if pv.type == 'Parameter':
                    proto_network.parameters[n_pv.name] = n_pv
                else:
                    proto_network.variables[n_pv.name] = n_pv
                variable_names.add(n_pv.name)

        for pf in self.functions.values():
            for variable_index in itertools.product(*map(tuple, map(range, [self.repeat_info[rid] for rid in pf.repeat_id]))):
                variable_index_name = ''.join(
                    ['_{}[{}]'.format(pf.repeat_id[index], i) for index, i in enumerate(variable_index)])
                variable_low_level_name = ''.join(
                    ['_{}[{}]'.format(pf.repeat_id[index], i) for index, i in enumerate(variable_index[:-1])])
                func_name = pf.name + variable_index_name
                inputs = pf.inputs
                outputs = pf.outputs
                if pf.type == "RepeatStart":
                    assert (len(pf.inputs) == 2)
                    if variable_index[-1] == 0:
                        input_names = [
                            inputs[0] if inputs[0] in variable_names else inputs[0] + variable_low_level_name]
                    else:
                        input_names = [inputs[1] + variable_low_level_name + '_{}[{}]'.format(
                            pf.repeat_param.repeat_id, variable_index[-1] - 1)]
                elif pf.type == "RepeatEnd":
                    assert (len(pf.inputs) == 1)
                    times = self.repeat_info[pf.repeat_param.repeat_id]
                    input_names = [inputs[0] + variable_index_name +
                                   '_{}[{}]'.format(pf.repeat_param.repeat_id, times - 1)]
                elif pf.type == "RecurrentInput":
                    if variable_index[-1] > 0:
                        continue
                    func_name = pf.name + variable_low_level_name
                    input_names = [v_name if v_name in variable_names else
                                   v_name + variable_low_level_name for v_name in pf.inputs]
                elif pf.type == "RecurrentOutput":
                    assert len(pf.inputs) == 1, "RecurrentOutput only has one input, but actual has {}.".format(
                        len(pf.inputs))
                    input_names = [inputs[0] + variable_index_name + '_{}[{}]'.format(pf.recurrent_param.repeat_id,
                                                                                      v_index) for v_index in range(pf.recurrent_param.length)]
                elif pf.type == 'Delay':
                    assert len(pf.inputs) == 2, "Delay requires 2 inputs."
                    if variable_index[-1] == 0:
                        input_names = [
                            inputs[1] if inputs[1] in variable_names else inputs[1] + variable_low_level_name]
                    else:
                        input_names = [inputs[0] + variable_low_level_name + '_{}[{}]'.format(pf.recurrent_param.repeat_id,
                                                                                              variable_index[-1] - 1)]
                else:
                    v_names = []
                    for v_name in inputs:
                        for index, i in enumerate(variable_index):
                            v_name = v_name.replace(
                                '{' + pf.repeat_id[index] + '}', '[{}]'.format(i))
                        v_names.append(v_name)
                    input_names = [v_name if v_name in variable_names else
                                   v_name + variable_index_name if v_name + variable_index_name in variable_names else
                                   v_name + variable_low_level_name for v_name in v_names]

                if pf.type == "RecurrentInput":
                    pv = pf.owner().variables[inputs[0]]
                    output_names = [outputs[0] + variable_low_level_name
                                    + '_{}[{}]'.format(pf.recurrent_param.repeat_id, v_index)
                                    for v_index in range(pv.shape[pf.recurrent_param.axis])]
                else:
                    output_names = [v_name + variable_index_name if v_name + variable_index_name
                                    in variable_names else v_name for v_name in outputs]

                if pf.type == "RepeatStart":
                    n_pf = ProtoFunction(
                        None, "Identity", pf.args, '$EC$/' + func_name, proto_network)
                elif pf.type == "RepeatEnd":
                    n_pf = ProtoFunction(
                        None, "Identity", pf.args, '$EC$/' + func_name, proto_network)
                elif pf.type == "RecurrentOutput":
                    n_pf = ProtoFunction(
                        None, "Stack", {"axis": pf.recurrent_param.axis}, func_name, proto_network)
                elif pf.type == "RecurrentInput":
                    n_pf = ProtoFunction(
                        None, "Split", {"axis": pf.recurrent_param.axis}, func_name, proto_network)
                elif pf.type == "Delay":
                    n_pf = ProtoFunction(
                        None, "Identity", pf.args, '$EC$/' + func_name, proto_network)
                else:
                    n_pf = ProtoFunction(
                        None, pf.type, pf.args, func_name, proto_network)

                n_pf.inputs = input_names
                n_pf.outputs = output_names
                for pv in [proto_network.variables[k] for k in n_pf.outputs]:
                    pv.parent = n_pf.name
                for pv in [proto_network.variables[k] if k in proto_network.variables
                           else proto_network.parameters[k] for k in n_pf.inputs]:
                    pv.required.append(n_pf.name)

                proto_network.functions[n_pf.name] = n_pf

        proto_network.inputs = [k for k, v in proto_network.variables.items()
                                if not v.parent and v.required]
        proto_network.outputs = [k for k, v in proto_network.variables.items()
                                 if not v.required and v.parent]

        return proto_network

    def execute_on_proto(self, execute):
        """ This function performs a virtual forward, following the sequence from inputs to output. This function does
        not use recursive call to perform graph-travel, instead, a non-recursive algorithm is used to graph-travel.
        For each function, `execute` is called when meet a function, a ProtoFunction object is passed in for further
        operation with this function.

        Args:
            execute (callable):
                A callback function (or callable object), which is called when each ProtoFunction is met in travelling
                graph.

                ``execute`` should look like:

                .. code-block:: python

                    def execute(pf: ProtoFunction):
                        # Do what you want to do with pf
                        pass

                Or:

                .. code-block:: python

                    class MyCallback:
                        def __call__(pf: ProtoFunction):
                            # Do what you want to do with pf
                            pass
        """
        from nnabla.logger import logger

        evaluated = {}
        stack = []
        for inp in self.inputs:
            evaluated[inp] = 0

        for out in self.outputs:
            pv = self.variables[out]
            if pv.parent:
                pf = self.functions[pv.parent]
                stack.append(pf)

        while True:
            if stack:
                pf = stack.pop()
                if pf.name not in self.functions:
                    continue
            else:
                break
            if len(pf.inputs) != 0:
                pf_inputs = [self.variables[k] if k in self.variables else self.parameters[k]
                             for k in pf.inputs]
                stop = reduce(lambda v0, v1: v0 | v1, map(
                         lambda x: x.stop, pf_inputs))
                pf_outputs = [self.variables[k] if k in self.variables else self.parameters[k]
                              for k in pf.outputs]
                for pv in pf_outputs:
                    pv.stop |= stop
                unresolved = list(
                    filter(lambda pv: pv.parent and pv.name not in evaluated, pf_inputs))
            else:
                unresolved = []
                stop = False
                pf_outputs = [self.variables[k] if k in self.variables else self.parameters[k]
                              for k in pf.outputs]
            if not unresolved:
                no_need = all(map(lambda pv: pv.name in evaluated, pf_outputs))
                need = all(
                    map(lambda pv: pv.name not in evaluated, pf_outputs))
                assert no_need or need, "Invalid graph if different functions output to same variable."
                if not no_need:
                    if not stop:
                        execute(pf)
                        logger.debug("execute: {}, {}".format(
                            pf.name, [n for n in pf.outputs]))
                    else:
                        logger.debug("skip {}".format(pf.name))
                        if pf.delegate is not None:
                            pf.delegate().disable()
                    evaluated.update({pv_name: 0 for pv_name in pf.outputs})
            else:
                stack.append(pf)
                for pv in unresolved:
                    stack.append(self.functions[pv.parent])

    def forward_sequence(self):
        """ This function creates an iteratable for iterating functions in
        network with the sequence of actually forward.

        For example,

            .. code-block:: python

                for pf in proto_network.forward_sequence():
                    print(pf.name)

        """
        functions = []

        def visit_func(pf):
            functions.append(pf)
        self.execute_on_proto(visit_func)
        for pf in functions:
            yield pf

    @staticmethod
    def create_function_instance(ctx, f, g, create_func):
        class Param:
            pass
        if f.type not in __loop_control_functions__:
            func = create_func(ctx, f)
            arguments = func.arguments
        else:
            arguments = None
        proto_function = ProtoFunction(None,
                                       f.type,
                                       arguments,
                                       f.name,
                                       g)
        if f.HasField('repeat_param'):
            proto_function.repeat_param = Param()
            proto_function.repeat_param.repeat_id = f.repeat_param.repeat_id
            proto_function.repeat_param.times = f.repeat_param.times
        elif f.HasField('recurrent_param'):
            proto_function.recurrent_param = Param()
            proto_function.recurrent_param.repeat_id = f.recurrent_param.repeat_id
            proto_function.recurrent_param.length = f.recurrent_param.length
            proto_function.recurrent_param.axis = f.recurrent_param.axis

        proto_function.repeat_id = [repeat_id for repeat_id in f.repeat_id]
        return proto_function

    @staticmethod
    def from_proto(owner, proto, rng, batch_size):
        from nnabla.utils.load_function import _create_function_instance

        g = ProtoNetwork(owner)
        g.repeat_info = {r.id: r.times for r in proto.repeat_info}
        g.batch_size = batch_size if batch_size else proto.batch_size
        g.name = proto.name
        for v in proto.variable:
            if v.type == 'Buffer':
                pv = g.variables[v.name] = ProtoVariable(
                    list(v.shape.dim), v.name, False, v.type)
                pv.initializer = _create_initializer(v, rng)
            elif v.type == 'Parameter':
                pv = g.parameters[v.name] = ProtoVariable(
                    list(v.shape.dim), v.name, True, v.type)
                pv.initializer = _create_initializer(v, rng)
            pv.repeat_id = [repeat_id for repeat_id in v.repeat_id]
        for f in proto.function:
            # Here, we temporarily created a function instance for converting
            # arguments from proto-oriented representation to general-purpose representation
            pf = ProtoNetwork.create_function_instance(
                nn.get_current_context(), f, g, _create_function_instance)
            g.functions[f.name] = pf
            for n in f.input:
                pv = g.variables[n] if n in g.variables else g.parameters[n]
                pf.inputs.append(pv.name)
                pv.required.append(pf.name)
            for n in f.output:
                pv = g.variables[n] if n in g.variables else g.parameters[n]
                pf.outputs.append(pv.name)
                pv.parent = pf.name
        g.inputs = [k for k, v in g.variables.items()
                    if not v.parent and v.required]
        g.outputs = [k for k, v in g.variables.items()
                     if not v.required and v.parent]
        return g

    def save(self, filename, include_parameter=False, variable_batch_size=True):
        """This function saves current proto network to a file, which is specified by
        filename, normally, e.g. a .nnp file.

        Args:
            filename (str):
                string filename, its extension name is used to determine the file format.
                The extension name normally is .nnp.
            include_parameter (bool, optional, default=False):
                Whether saving parameters to protobuf tree.
            variable_batch_size (bool, optional, default=True):
                Whether replace current network's batch size dimension with an abstract
                representation. If it is true, it is possible to use another batch size
                when this network is reused.
        """
        save(filename, [self], include_parameters=include_parameter,
             variable_batch_size=variable_batch_size)

    def clone(self):
        proto_network = ProtoNetwork(self.owner(), self.name, self.batch_size)
        for pv in self.variables.values():
            proto_network.variables[pv.name] = pv.clone()

        for pv in self.parameters.values():
            proto_network.parameters[pv.name] = pv.clone()

        for pf in self.functions.values():
            proto_network.functions[pf.name] = pf.clone(proto_network)

        proto_network.inputs = self.inputs.copy()
        proto_network.outputs = self.outputs.copy()

        # Detach from ProtoGraph, we need to hard-refer to its parent
        proto_network.owner_ = self.owner()

        return proto_network

    def _update_network_ports(self):
        self.inputs = [k for k, v in self.variables.items()
                       if not v.parent and v.required]
        self.outputs = [k for k, v in self.variables.items()
                        if not v.required and v.parent]
        self.variables = OrderedDict([(pv.name, pv) for pv in
                                      filter(lambda v: v.required or v.parent, self.variables.values())])
        self.parameters = OrderedDict([(pv.name, pv) for pv in
                                       filter(lambda v: v.required or v.parent, self.parameters.values())])
        self.functions = OrderedDict([(pf.name, pf) for pf in
                                      filter(lambda pf: pf.inputs or pf.outputs, self.functions.values())])

    def _patch_by_network_pass(self, callback):
        class VariableDelegate:
            def __init__(self, pv, pn):
                self.pv = pv
                self.pn = pn
                pv.delegate = weakref.ref(self)

            @property
            def proto(self):
                return self.pv.proto

            @proto.setter
            def proto(self, proto):
                self.pv.proto = proto

            @property
            def parent(self):
                if self.pv.parent is None:
                    return None
                return self.pn.functions[self.pv.parent].delegate()

            @parent.setter
            def parent(self, f):
                if f is None:
                    self.pv.parent = None
                else:
                    self.pv.parent = f.name

            @property
            def name(self):
                return self.pv.name

            def add_referrer(self, f):
                assert isinstance(f, VariableDelegate)
                self.pv.required.append(f.name)

            @property
            def referrers(self):
                referrers = {f_name: self.pn.functions[f_name].delegate()
                             for f_name in self.pv.required
                             if self.pn.functions[f_name].delegate() is not None}
                return referrers

            @property
            def num_referrers(self):
                return len(self.pv.required)

            def delete_referrer(self, f):
                self.pv.required.remove(f.name)

            def rewire_on(self, var):
                parent = var.parent
                if self.parent is not None:
                    self.parent.disable()
                self.parent = parent
                var.parent = None
                self.variable = var.variable

                # Replace var with self for var.parent.outputs
                if parent is None:
                    return
                new_outputs = []
                for o in parent.outputs:
                    new = o
                    if o is var:
                        new = self
                    new_outputs.append(new)
                # using up stream as current
                self.parent.outputs = new_outputs

            @property
            def stop(self):
                return self.pv.stop

            @stop.setter
            def stop(self, s):
                self.pv.stop = s

            @property
            def need_grad(self):
                return self.pv.need_grad

            @need_grad.setter
            def need_grad(self, need_grad):
                self.pv.need_grad = need_grad

            @property
            def variable(self):
                return self.pv.variable_instance

            @variable.setter
            def variable(self, v):
                self.pv.variable_instance = v

        class FunctionDelegate:
            def __init__(self, pf, pn):
                self.pf = pf
                self.pn = pn
                pf.delegate = weakref.ref(self)

            @property
            def proto(self):
                return self.pf.proto

            @proto.setter
            def proto(self, proto):
                self.pf.proto = proto

            @property
            def name(self):
                return self.pf.name

            @property
            def inputs(self):
                return [self.pn.variables[v_name].delegate() if v_name in self.pn.variables
                        else self.pn.parameters[v_name].delegate() for v_name in self.pf.proto.input]

            @inputs.setter
            def inputs(self, inputs):
                self.pf.proto.ClearField('input')
                self.pf.proto.input.extend([f.name for f in inputs])

            @property
            def outputs(self):
                return [self.pn.variables[v_name].delegate() if v_name in self.pn.variables
                        else self.pn.parameters[v_name].delegate() for v_name in self.pf.proto.output]

            @outputs.setter
            def outputs(self, outputs):
                self.pf.proto.ClearField('output')
                self.pf.proto.output.extend([f.name for f in outputs])

            def disable(self):
                self.pf.disabled = True
                for i in self.inputs:
                    i.delete_referrer(self)  # Forget me.
                for o in self.outputs:
                    o.parent = None  # Forget me.
                self.inputs = []
                self.outputs = []

            @property
            def disabled(self):
                return self.pf.disabled

        def filter_variable_by_callback(variables, callback):
            var_list = []
            for ppv in variables:
                var_list.append(callback._apply_generate_variable(ppv))
            return var_list

        def filter_function_by_callback(functions, callback):
            func_list = []
            for pf in functions:
                pf = callback._apply_generate_function_by_type(pf)
                pf = callback._apply_generate_function_by_name(pf)
                func_list.append(pf)
            return func_list

        scope = self.owner().parameter_scope
        n = self.clone()

        variables = [VariableDelegate(pv, n) for pv in n.variables.values()]
        parameters = [VariableDelegate(pv, n) for pv in n.parameters.values()]
        functions = [FunctionDelegate(pf, n) for pf in n.functions.values()]

        variables = filter_variable_by_callback(variables, callback)
        parameters = filter_variable_by_callback(parameters, callback)
        functions = filter_function_by_callback(functions, callback)

        n.variables = OrderedDict([(v.name, v.pv) for v in variables])
        n.parameters = OrderedDict([(v.name, v.pv) for v in parameters])
        n.functions = OrderedDict([(v.name, v.pf) for v in functions])

        variables += parameters

        n._update_network_ports()
        for pf in n.forward_sequence():
            if pf.delegate().disabled:
                continue
            callback._apply_function_pass_by_type(
                pf.delegate(), variables, scope)
            callback._apply_function_pass_by_name(
                pf.delegate(), variables, scope)

        renames = {}
        for pv in chain(n.variables.values(), n.parameters.values()):
            pv.prepare_renaming(renames)

        def shorten_route(naming):
            ret = {}
            flags = {k: True for k, _ in naming.items()}
            for k, v in naming.items():
                d = v
                while v in naming:
                    d = naming[v]
                    flags[v] = False
                    v = d
                if flags[k]:
                    ret[k] = d
            return ret

        renames = shorten_route(renames)

        n._update_network_ports()
        for pf in n.functions.values():
            callback._apply_use_up_to(pf.delegate().inputs)

        rng = np.random.RandomState(1223)
        for pv in chain(n.variables.values(), n.parameters.values()):
            pv.update_from_proto(rng, renames)

        n.variables = OrderedDict([(renames.get(k, k), v)
                                  for k, v in n.variables.items()])
        n.parameters = OrderedDict([(renames.get(k, k), v)
                                   for k, v in n.parameters.items()])

        for pf in n.functions.values():
            pf.update_from_proto(renames)
        n.inputs = [renames.get(name, name) for name in n.inputs]
        n.outputs = [renames.get(name, name) for name in n.outputs]

        # apply disable()
        for _ in n.forward_sequence():
            pass

        n._update_network_ports()
        return n

    def promote(self, callback):
        '''User may manipulate a proto network by a callback, like NnpNetworkPass.

        Args:
            callback (NnpNetworkPass, ):
                Currently, only NnpNetworkPass object is supported as a network promotion
                callback.

        Developers may manipulate a proto network by a modifier, acts as a callback.
        nnabla.utils.nnp_graph.NnpNetworkPass is a kind of modifier. The following gives
        a simple example to illustrate this usage:

        Example:

            .. code-block:: python

                from nnabla as nn
                from nnabla.utils import nnp_graph

                verbose = 1
                callback = nnp_graph.NnpNetworkPass(verbose)

                @callback.on_generate_function_by_name('Convolution')
                def change_convolution_param(f):
                    print('{}'.format(f.proto.convolution_param.pad.dim[:]))
                    f.proto.convolution_param.pad.dim[:] = [1, 1]
                    return f

                g = nn.graph_def.load("my_model.nnp")
                n = g.default_graph().promote(callback)
                x = nn.Variable(input_shape)
                y = n(x)
                y.forward()

            In this example, a callback is defined to change pad of a Convolution function,
            locating this target function by the name of function, here, only the function with the name `'Convolution'`
            is located and operated.
        '''
        from nnabla.utils.nnp_graph import NnpNetworkPass
        if callback is None:
            return self
        if isinstance(callback, NnpNetworkPass):
            return self._patch_by_network_pass(callback)
        # TODO: handle network by other modifier
        # Not implemented yet
        return None


class ProtoGraph:
    """This class represents a group of proto networks. It normally corresponds to a `.nnp` file.
    In a `.nnp` file, there might be one or multiple networks, for example, there might be a network for
    doing directly inferring, another network with similar network structure, sharing same parameters, using
    for training. This class works as a container of proto networks, providing a group of functions
    for accessing proto network by its name. Especially, when there is only one network in it, also some
    short-cut functions are provided for directly operating with this network. For example,

    .. code-block:: python

        import nnabla as nn

        g = nn.graph_def.load("my_model.nnp") # Suppose there is only one network in this file.
        x1 = nn.Variable(input_shape)
        x1.d = ... # load data here.
        y1 = g.networks['executor_net'](x1)  #<== (1)
        y1.forward()
        print(y1.d)

        x2 = nn.Variable(input_shape)
        x2.d = ... # load data here.
        y2 = g(x2) #<== (2)
        # y2 = g.default_graph()(x2) #<== (3)
        y2.forward()
        print(y2.d)

    The computation graph `y1` and `y2` are exactly same. (2) and (3) are equal.
    If there are multiple networks in a graph, the first network being loaded acted as its `default` network.
    Please not use `default_graph()` when there are multiple networks in graph, since the `default` heavily depends on
    concrete implementation.

    If you know the name of each network, you may access proto network in this graph by its member name. For example,

    .. code-block:: python

        g = nn.graph_def.load("my_model.nnp") # Suppose there is only one network in this file.
        x = nn.Variable(input_shape)
        x.d = ... # load data here.
        y = g.executor_net(x) # here, we knew there is a network named "executor_net" existed.
        y.forward()
        print(y.d)

    """

    def __contains__(self, item):
        return item in self.__dict__

    def __getattr__(self, item):
        if item in __graph_members__:
            return self.default_graph()[item]
        if item in self.__dict__.get('networks', {}):
            return self.__dict__['networks'][item]
        return self.__dict__[item]

    def __getitem__(self, item):
        return self.__getattr__(item)

    def __setitem__(self, key, value):
        if isinstance(value, ProtoNetwork):
            self.networks[key] = value
        else:
            self.__dict__[key] = value

    def get(self, key, value=None):
        return self.__dict__.get(key, value)

    def __init__(self, networks=None, parameter_scope=None):
        self.networks = networks if networks is not None else OrderedDict()
        self.executors = OrderedDict()
        self.parameter_scope = parameter_scope if parameter_scope else OrderedDict()
        self.global_variables = {}

    def get_parameters(self, grad_only=False):
        """Get parameters in current module name scope.
        """
        with nn.parameter_scope('', self.parameter_scope):
            params = nn.get_parameters(grad_only=grad_only)
        return params

    def set_parameter(self, key, param):
        with nn.parameter_scope('', self.parameter_scope):
            nn.parameter.set_parameter(key, param)

    def set_parameters(self, params):
        for key, param in params.items():
            self.set_parameter(key, param)

    @property
    def current_context(self):
        """ Current backend context of this proto network.
        """
        if '_context' not in self.__dict__:
            return nn.get_current_context()
        return self.__dict__['_context']

    @current_context.setter
    def current_context(self, ctx):
        current = self.current_context
        if ctx != current:
            # If context is changed, we clear all
            # function instances
            for n in self.networks.values():
                for f in n.functions.values():
                    f.function_instance = None
        self.__dict__['_context'] = ctx

    @staticmethod
    def from_proto(proto, batch_size=None, param_scope=None, rng=None):
        """This function create a proto graph object from a protobuf data structure.

        Args:
            proto (protobuf object):
                A protobuf data structure.
            batch_size (int, optional, default=None):
                The batch size will be applied to this graph. If it is None, it is pending
                to apply a the batch size value.
            param_scope (OrderedDict, optional, default=None):
                User may provide an owned parameter scope.
            rng (np.random.RandomState, optional, default=None):
                    A random seed, which is used in parameter initialization.

        """
        if not rng:
            rng = np.random.RandomState(0)

        g = ProtoGraph()
        if param_scope is not None:
            g.parameter_scope = param_scope

        for p in proto.network:
            g.networks[p.name] = ProtoNetwork.from_proto(g, p, rng, batch_size)

        for e in proto.executor:
            g.executors[e.name] = ProtoExecutor.from_proto(e)

        return g

    def as_proto(self, include_parameter=False, only_parameter=False, networks=None, variable_batch_size=True):
        """This function exports a protobuf data structure, which can be manipulated by google protobuf APIs.

        Args:
            include_parameter (bool, optional, default=False):
                Whether exports the parameters to protobuf data structure.
            only_parameter (bool, optional, default=False):
                Whether only exports the parameters to protobuf data structure.
            networks (array of proto networks, optional, default=None):
                User may provides their networks to export a protobuf data structure.
            variable_batch_size (bool, optional, default=True):
                Replace batch size of current network with an abstract placeholder, so that
                batch size can be replaced with other value in use time.

        """
        from nnabla.utils import nnabla_pb2
        proto = nnabla_pb2.NNablaProtoBuf()
        if not only_parameter:
            if networks is None:
                networks = [m.as_proto(variable_batch_size=variable_batch_size)
                            for m in self.networks.values()]
            else:
                networks = [m.as_proto(
                    variable_batch_size=variable_batch_size) for m in networks]
            proto.network.extend(networks)
            if self.executors:
                executors = [e.as_proto() for e in self.executors.values()]
                proto.executor.extend(executors)
        if include_parameter:
            for k, v in self.get_parameters().items():
                parameter = proto.parameter.add()
                parameter.variable_name = k
                parameter.shape.dim.extend(v.shape)
                parameter.data.extend(np.array(v.d).flatten().tolist())
                parameter.need_grad = v.need_grad
        return proto

    def default_graph(self):
        """This function returns default proto network in this graph.
        Which network is `default` graph depends on its loading sequence. Hence,
        it is safe to be used when there is only one network.
        """
        if self.networks:
            return list(self.networks.values())[0]
        raise ValueError("No network is found in current graph.")

    def __call__(self, *args, **kwargs):
        return self.default_graph()(*args, **kwargs)

    def save(self, filename, include_parameter=False, variable_batch_size=True):
        # TODO: Add commit_network() here.
        save(filename, [self], include_parameters=include_parameter,
             variable_batch_size=variable_batch_size)

    def expand_loop_control(self):
        """ This function expands loop control statements for all networks in
        this graph.
        """
        expanded = []
        for name, network in self.networks.items():
            n = network.expand_loop_control()
            n.owner_ = None  # remove hard-reference to avoid cycle reference.
            expanded.append((name, n))
        self.networks = OrderedDict(expanded)


class FlatModule:
    """FlatModule is a module-like placeholder for generating
    graph_def from a flat-style network definition. -- without module hierarchy
    """

    @staticmethod
    def get_parameters(recursive=True, grad_only=False, memo=None):
        return nn.get_parameters()

    def get_path_name(self):
        return ('@' + self.name) if self.name != 'model' else ''

    def __init__(self, name):
        self.name = name if name else "model"


class ProtoGraphBuilder:
    """ProtoGraphBuilder holds the context during the construction of ProtoGraph.
    """

    def __init__(self, **kwargs):
        self.networks = OrderedDict()
        self.stack = []
        self.current = None
        self.module = None
        self.names = {}
        self.dirty_flag = False
        self.graph_name = kwargs.get('name', None)
        self.proto_graph = ProtoGraph(
            self.networks, kwargs.get('parameter_scope', None))

    def get_current(self):
        if self.current is not None:
            return self.current
        self.current = ProtoNetwork(self.proto_graph)
        return self.current

    def get_module(self):
        if self.module is not None:
            return self.module
        self.module = FlatModule(self.graph_name)
        return self.module

    def dirty(self):
        self.dirty_flag = True

    def is_dirty(self):
        return self.dirty_flag

    def get_graph(self):
        return self.proto_graph

    def begin_module(self, module):
        self.stack.append(module)
        self.module = module
        return self.get_current()

    def end_module(self):
        self.stack.pop()
        if self.stack:
            # non-top module
            self.module = self.stack[-1]
        else:
            # Top module
            self.commit_network()
            # self.module = None

    def commit_network(self):
        if self.graph_name:
            g_name = self.graph_name
            self.graph_name = None
        else:
            g_name = self.module.name
        g_name = _get_unique_name(self.names, g_name)
        params = {v.data: k for k, v in self.module.get_parameters().items()}
        self.current.commit(g_name, self.networks, params)
        self.dirty_flag = False
        self.current = None
        self.module = None


@contextmanager
def graph_name(name):
    g_name = proto_graph_builder.graph_name
    proto_graph_builder.graph_name = name
    yield proto_graph_builder.current
    if proto_graph_builder.is_dirty():
        proto_graph_builder.commit_network()
    proto_graph_builder.graph_name = g_name


previous_graph_builder = None
proto_graph_builder = ProtoGraphBuilder()


@contextmanager
def graph(**kwargs):
    """This function is only used in with statement.

    Args:
        name (str, optional, default=None):
            User may specify a name for the generated proto network.
            This name is useful for saving to .nnp.

        parameter_scope (OrderedDict, optional, default=None):
            User may specify a parameter scope, thus, the parameters
            are created during creating model will be placed into this
            parameter scope.

    For example,

        .. code-block:: python

            import nnabla as nn

            proto_variable_inputs = [nn.ProtoVariable(v.d.shape) for v in inputs]
            with nn.graph_def.graph() as g:
                outputs = module(*proto_variable_inputs)

            g.save("my_model.nnp")

        Here, `inputs` is an array of input nn.Variables. `module` is a module object
        instantiated from a Module definition.

    """
    global previous_graph_builder, proto_graph_builder
    previous_graph_builder = proto_graph_builder
    proto_graph_builder = ProtoGraphBuilder(**kwargs)
    yield proto_graph_builder.get_graph()
    if proto_graph_builder.is_dirty():
        proto_graph_builder.commit_network()
    proto_graph_builder = previous_graph_builder


@contextmanager
def module_scope(pgb, module):
    gd = pgb.begin_module(module)
    yield gd
    pgb.end_module()


def current_graph_builder():
    return proto_graph_builder


def current_network():
    return proto_graph_builder.get_current()


def current_module():
    return proto_graph_builder.get_module()


def reset_default_graph():
    """This function clear all information in global graph scope.
    """
    global proto_graph_builder
    proto_graph_builder = ProtoGraphBuilder()


def get_default_graph(*args, **kwargs):
    """This function obtain current default graph_def.

    If user does not create their proto network in a with statement
    scope, proto network will default be created in a global scope.
    User may retrieve this proto graph by this function.

    Example:

        .. code-block:: python

            import nnabla as nn
            from nnabla.core.modules import ResUnit

            resunit = ResUnit(16)
            input = nn.ProtoVariable((64, 3, 32, 32))
            y = resunit(input)
            graph_def = nn.graph_def.get_graph_graph()

    Note:
        If user does not ensure whether there is any previous existing
        proto graph remained in global graph scope. It is better to call
        reset_default_graph(). If user uses with statement like `with nn.graph_def.graph() as g`,
        this point is no need to care about.

    Returns:
        ProtoGraph:
            A proto graph is returned

    """
    network_name = None
    if args:
        if isinstance(args[0], str):
            network_name = args[0]
    else:
        network_name = kwargs.get('name', None)

    if proto_graph_builder.is_dirty():
        proto_graph_builder.commit_network()

    if network_name is None:
        return proto_graph_builder.proto_graph

    return proto_graph_builder.proto_graph.networks.get(network_name, None)


def get_default_graph_by_variable(proto_variable):
    """This function obtain a specify network by its outputs.

    User may retrieve one of the networks in default proto graph scope, if this network has the specified
    outputs. Let us image that there is a global proto graph, when user passed a ProtoVariable to
    model, during the procedure that create output variables, proto network is generated in this global
    proto graph. By this function, user may retrieve this generated proto network, saving it or do
    any other operations.

    Note:
        This proto network will become invalid after reset_default_graph().
        For example,

            .. code-block:: python

                proto_variable_inputs = [nn.ProtoVariable(v.d.shape) for v in inputs]
                outputs = module(*proto_variable_inputs)
                net = nn.graph_def.get_default_graph_by_variable(outputs[0])
                ...
                nn.graph_def.reset_default_graph()
                y = net(x) # Cannot access net anymore, become invalid at this point

    """

    if proto_graph_builder.is_dirty():
        proto_graph_builder.commit_network()

    for network in proto_graph_builder.proto_graph.networks.values():
        if proto_variable.name in network.outputs:
            return network
    raise ValueError(
        "{} is not the output variable of any network".format(proto_variable.name))


class ProtoGraphGenerator:
    '''Only used for create graph from variable
    '''

    def __init__(self, names, params):
        if names is not None:
            self.names = {v.data: k for k, v in names.items()}
        else:
            self.names = {}
        self.names.update(params)
        self.params = params
        self.variables = {}

    def __enter__(self):
        proto_network = current_network()
        proto_network.renaming = self.renaming
        return self

    def __exit__(self, type, value, traceback):
        proto_network = current_network()
        variables = OrderedDict()
        parameters = OrderedDict()
        for pv_name, pv in proto_network.variables.items():
            if pv.type == 'Buffer':
                variables[pv.name] = pv
                pv.variable_instance = None
            else:
                parameters[pv.name] = pv
        proto_network.parameters = parameters
        proto_network.variables = variables

    def renaming(self, i, v_name):
        return self.names.get(self.outputs[i].data, v_name)

    def __call__(self, func):
        if str(func) == "Sink":
            return
        inputs = []
        for v in func.inputs:
            if v.data in self.variables:
                inputs.append(self.variables[v.data])
            else:
                if v.data in self.params:
                    pv = ProtoVariable(v.d.shape, var_type='Parameter')
                else:
                    pv = ProtoVariable(v.d.shape, var_type='Buffer')
                pv.variable_instance = v
                pv.name = self.names.get(v.data, None)
                inputs.append(pv)
                self.variables[v.data] = pv

        self.outputs = func.outputs
        if len(inputs) == 0:
            outputs = ProtoFunction(func, func.name, func.arguments)(
                inputs=[], n_outputs=len(func.outputs))
        else:
            outputs = func(*inputs, n_outputs=len(func.outputs),
                           auto_forward=False)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        for pv, v in zip(outputs, func.outputs):
            self.variables[v.data] = pv


def create_graph_from_variable(name, variables, names=None, parameter_scope=None):
    """Create a Proto Graph from one or multiple outputs.

    If developers have a computation graph, it means that they have a nn.Variable() object,
    it might be loss of a network or an output variable of an executor network, this variable
    inherently corresponds to a computation network. From these variables, we can create
    corresponding proto network by this function.

    Args:
        name (str):
            The name of generated proto_network.
        variables (nn.Variables):
            One or multiple variables, if multiple variables, this function adds a sink function
            to reduce these multiple outputs to one.
        names (dict, optional, default=None) :
            A name to nn.Variable mapping table. This function default names all activation variables
            and parameters with internal naming rule. But sometimes, developers expects specially name some of
            variables so that these variable can be accessed conveniently. In generating proto network,
            when the variable occurs in this mapping table, the corresponding name of that variable will be used to name
            the variable in proto network.
        parameter_scope (OrderedDict, optional, default=None):
            Developers may provide a parameter scope, thus, when create proto networks, the name will be replaced
            if corresponding variable is found in specified parameter_scope, which make the name of weights or some
            parameters meaningful.


    Example:

        .. code-block:: python

            import nnabla as nn

            x = nn.Variable((1, 3, 32, 32))
            y = my_model(x)
            g = nn.graph_def.create_graph_from_variable("proto_network_name", y)
            g.save("my_model.nnp")

    """
    import nnabla.functions as F
    if isinstance(variables, (list, tuple)):
        output = F.sink(*variables)
    else:
        output = variables

    if parameter_scope:
        with nn.parameter_scope('', parameter_scope):
            params = {v.data: k for k, v in nn.get_parameters(
                grad_only=False).items()}
    else:
        params = {v.data: k for k, v in nn.get_parameters(
            grad_only=False).items()}

    with graph(parameter_scope=parameter_scope, name=name) as g:
        with ProtoGraphGenerator(names, params) as pgg:
            output.visit(pgg)
    return g


class ProtoVariable:
    """This class represents a variable, so-called proto variable.
    Passing this variable to network definition, a proto network will
    be generated in a proto graph scope. If this procedure is done
    under a with statement as `g`, a proto network will be generated in `g`.
    Otherwise, a global graph scope is used, a proto network will be
    generated in global graph scope.

    """

    def __init__(self, shape, name=None, need_grad=False, var_type='Buffer'):
        self.name = name
        self.shape = tuple(shape)
        self.need_grad = need_grad
        self.parent = None
        self.required = []
        self.type = var_type
        self.info = None
        self.initializer = None
        self.variable_instance = None
        self.repeat_id = []
        self.stop = False
        self.delegate = None
        self._proto = None

    def clone(self):
        v = ProtoVariable(self.shape[:], self.name,
                          self.need_grad, self.type)
        v.parent = self.parent
        v.required = self.required.copy()
        v.initializer = self.initializer
        v.info = self.info.copy() if self.info is not None else None
        if v.info:
            v.initializer = v.info.initialze
        v.variable_instance = self.variable_instance
        v.repeat_id = self.repeat_id.copy()
        return v

    @property
    def proto(self):
        if self._proto is not None:
            return self._proto
        from nnabla.utils import nnabla_pb2
        self._proto = nnabla_pb2.Variable()
        self._proto.name = self.name
        self._proto.type = self.type
        self._proto.repeat_id.extend(self.repeat_id)
        self._proto.shape.dim.extend(list(self.shape))

        if self.initializer:
            initializer_mapper = {
                'Constant': lambda v: v.value,
                'Uniform': lambda v: -v.lim[0],
                'Normal': lambda v: v.sigma
                # Only support above initializer to proto
            }
            i = self._proto.initializer
            i.type = self.initializer.__class__.__name__.replace(
                'Initializer', '')
            i.multiplier = initializer_mapper.get(i.type,
                                                  lambda v: 0.0)(self.initializer)
        return self._proto

    def prepare_renaming(self, renames):
        if self._proto is None:
            return
        if self.name != self._proto.name:
            renames[self.name] = self._proto.name

    def update_from_proto(self, rng, renames):
        if self._proto is None:
            return
        self.name = renames.get(self.name, self.name)
        assert self.type == self._proto.type, "Change type of variable is not allowed"
        self.shape = self._proto.shape.dim[:]
        self.initializer = _create_initializer(self._proto, rng)
        self._proto = None

    def __pos__(self):
        """
        This function simply returns itself.
        Implements the unary plus operator, ``+A``.

        Returns: :class:`nnabla.ProtoVariable`

        """
        return self

    def __neg__(self):
        """
        Element-wise negation.
        Implements the unary negation expression ``-A`` .

        Returns: :class:`nnabla.ProtoVariable`

        """
        import nnabla.functions as F
        return F.mul_scalar(self, -1)

    def __add__(x, y):
        """
        Element-wise addition.

        Implements the addition operator expression ``x + y``.
        When both of ``x`` and ``y`` are either :obj:`~nnabla.ProtoVariable`, :func:`~nnabla.functions.add2`` is
        internally called.
        When one of ``x`` and ``y`` is a scalar,
        :func:`~nnabla.functions.add_scalar` is called.

        Args:
            x (float or ~nnabla.ProtoVariable): Left operand.
            y (float or ~nnabla.ProtoVariable): Right operand.

        Returns: :class:`~nnabla.ProtoVariable`

        """
        import nnabla.functions as F

        if isinstance(x, (nn.NdArray, nn.Variable)):
            x = ProtoVariable(x.shape)

        if isinstance(y, (nn.NdArray, nn.Variable)):
            y = ProtoVariable(y.shape)

        if isinstance(x, ProtoVariable):
            if isinstance(y, ProtoVariable):
                return F.add2(x, y)
            else:
                return F.add_scalar(x, y)
        else:
            if isinstance(y, ProtoVariable):
                return F.add_scalar(y, x)
            else:
                return x + y

    def __sub__(x, y):
        """
        Element-wise subtraction.

        Implements the subtraction operator expression ``x - y``.
        When both of ``x`` and ``y`` are either :obj:`~nnabla.ProtoVariable`, :func:`~nnabla.functions.sub2`` is
        internally called.
        When one of ``x`` and ``y`` is a scalar,
        :func:`~nnabla.functions.sub_scalar` is called.

        Args:
            x (float or ~nnabla.ProtoVariable): Left operand.
            y (float or ~nnabla.ProtoVariable): Right operand.

        Returns: :class:`~nnabla.ProtoVariable`

        """

        import nnabla.functions as F

        if isinstance(x, (nn.NdArray, nn.Variable)):
            x = ProtoVariable(x.shape)

        if isinstance(y, (nn.NdArray, nn.Variable)):
            y = ProtoVariable(y.shape)

        if isinstance(x, ProtoVariable):
            if isinstance(y, ProtoVariable):
                return F.sub2(x, y)
            else:
                return F.add_scalar(x, -y)
        else:
            if isinstance(y, ProtoVariable):
                return F.r_sub_scalar(y, x)
            else:
                return x - y

    def __mul__(x, y):
        """
        Element-wise multiplication.

        Implements the subtraction operator expression ``x - y``.
        When both of ``x`` and ``y`` are either :obj:`~nnabla.ProtoVariable`, :func:`~nnabla.functions.mul2`` is
        internally called.
        When one of ``x`` and ``y`` is a scalar,
        :func:`~nnabla.functions.mul_scalar` is called.

        Args:
            x (float or ~nnabla.ProtoVariable): Left operand.
            y (float or ~nnabla.ProtoVariable): Right operand.

        Returns: :class:`~nnabla.ProtoVariable`

        """

        import nnabla.functions as F

        if isinstance(x, (nn.NdArray, nn.Variable)):
            x = ProtoVariable(x.shape)

        if isinstance(y, (nn.NdArray, nn.Variable)):
            y = ProtoVariable(y.shape)

        if isinstance(x, ProtoVariable):
            if isinstance(y, ProtoVariable):
                return F.mul2(x, y)
            else:
                return F.mul_scalar(x, y)
        else:
            if isinstance(y, ProtoVariable):
                return F.mul_scalar(y, x)
            else:
                return x * y

    def __truediv__(x, y):
        """
        Element-wise division.

        Implements the division operator expression ``x / y``.
        When both of ``x`` and ``y`` are either :obj:`~nnabla.ProtoVariable`, :func:`~nnabla.functions.div2`` is
        internally called.
        When ``y`` is a scalar, :func:`~nnabla.functions.div_scalar`(x, y) is
        called. When ``x`` is a scalar,
        :func:`~nnabla.functions.r_div_scalar`(y, x) is called.

        Args:
            x (float or ~nnabla.ProtoVariable): Left operand.
            y (float or ~nnabla.ProtoVariable): Right operand.

        Returns: :class:`~nnabla.ProtoVariable`

        """
        import nnabla.functions as F
        if isinstance(x, (nn.NdArray, nn.Variable)):
            x = ProtoVariable(x.shape)

        if isinstance(y, (nn.NdArray, nn.Variable)):
            y = ProtoVariable(y.shape)

        if isinstance(x, ProtoVariable):
            if isinstance(y, ProtoVariable):
                return F.div2(x, y)
            else:
                return F.mul_scalar(x, 1.0 / y)
        else:
            if isinstance(y, ProtoVariable):
                return F.r_div_scalar(y, x)
            else:
                return x / y

    def __div__(x, y):
        """
        Element-wise division.

        Implements the division operator expression ``x / y``.
        When both of ``x`` and ``y`` are either :obj:`~nnabla.ProtoVariable`, :func:`~nnabla.functions.div2`` is
        internally called.
        When ``y`` is a scalar, :func:`~nnabla.functions.div_scalar`(x, y) is
        called. When ``x`` is a scalar,
        :func:`~nnabla.functions.r_div_scalar`(y, x) is called.

        Args:
            x (float or ~nnabla.ProtoVariable): Left operand.
            y (float or ~nnabla.ProtoVariable): Right operand.

        Returns: :class:`~nnabla.ProtoVariable`

        """
        import nnabla.functions as F
        if isinstance(x, (nn.NdArray, nn.Variable)):
            x = ProtoVariable(x.shape)

        if isinstance(y, (nn.NdArray, nn.Variable)):
            y = ProtoVariable(y.shape)

        if isinstance(x, ProtoVariable):
            if isinstance(y, ProtoVariable):
                return F.div2(x, y)
            else:
                return F.mul_scalar(x, 1.0 / y)
        else:
            if isinstance(y, ProtoVariable):
                return F.r_div_scalar(y, x)
            else:
                return x / y

    def __pow__(x, y, z):
        """
        Element-wise power function.

        Implements the power operator expression ``x ** y``,
        optionally ``x ** y % z`` (but not implemented).
        When both of ``x`` and ``y`` are either :obj:`~nnabla.ProtoVariable`, :func:`~nnabla.functions.pow2`` is
        internally called.
        When ``y`` is a scalar, :func:`~nnabla.functions.pow_scalar`(x, y) is
        called. When ``x`` is a scalar,
        :func:`~nnabla.functions.r_pow_scalar`(y, x) is called.

        Args:
            x (float or ~nnabla.ProtoVariable): Left operand.
            y (float or ~nnabla.ProtoVariable): Right operand.
            z (float or ~nnabla.ProtoVariable): Modulo (optional).

        Returns: :class:`~nnabla.ProtoVariable`.

        """
        import nnabla.functions as F
        if isinstance(x, (nn.NdArray, nn.Variable)):
            x = ProtoVariable(x.shape)

        if isinstance(y, (nn.NdArray, nn.Variable)):
            y = ProtoVariable(y.shape)

        if z is not None:
            return NotImplemented
        if isinstance(x, ProtoVariable):
            if isinstance(y, ProtoVariable):
                return F.pow2(x, y)
            else:
                return F.pow_scalar(x, y)
        else:
            if isinstance(y, ProtoVariable):
                return F.r_pow_scalar(y, x)
            else:
                return x ** y


class ProtoFunction:
    """This class represent a function that is used to define a proto network.

    There are the following properties to describe a proto function:
       * name: The name of this function.
       * type: The type of this function, e.g. ReLU.
       * inputs: An array of string name, which represents the proto variables of inputs.
       * outputs: An array of string name, which represents the proto variables of outputs.
    """

    def __init__(self, func, f_type, args, name=None, owner=None):
        self.type = f_type
        if name is None:
            self.path_name = current_module().get_path_name()
            func_name = '/'.join([self.path_name, self.type]
                                 ) if self.path_name else self.type
            self.name = _get_unique_name(
                current_network().func_names, func_name)
            self.owner = weakref.ref(current_network())
        else:
            self.name = name
            self.owner = weakref.ref(owner)
        self.args = args
        self.inputs = []
        self.outputs = []

        # Keep a function_instance to obtain output shape later.
        self.function_instance = func

        # The members for loop controls
        self.repeat_id = []
        self.repeat_param = None
        self.recurrent_param = None
        self.disabled = False
        self.delegate = None
        self._proto = None

    def clone(self, owner):
        proto_function = ProtoFunction(
            None, self.type, self.args, self.name, owner)
        proto_function.inputs = self.inputs.copy()
        proto_function.outputs = self.outputs.copy()
        proto_function.function_instance = None
        proto_function.repeat_id = self.repeat_id
        proto_function.repeat_param = self.repeat_param
        proto_function.recurrent_param = self.recurrent_param
        proto_function.disabled = self.disabled
        return proto_function

    @property
    def proto(self):
        if self._proto:
            return self._proto
        from nnabla.utils import nnabla_pb2
        from nnabla.utils.save_function import _create_function_nntxt
        self._proto = nnabla_pb2.Function()
        _create_function_nntxt(self._proto, self.name, self)
        return self._proto

    def update_from_proto(self, renames):
        from nnabla.utils.load_function import _create_function_instance

        # temporarily create a function instance for obtaining
        # arguments
        function_instance = _create_function_instance(
            self.owner().current_context, self._proto)
        self.args = function_instance.arguments
        self.type = self._proto.type
        self.name = self._proto.name
        self.inputs = [renames.get(n, n) for n in self._proto.input]
        self.outputs = [renames.get(n, n) for n in self._proto.output]
        self._proto.ClearField('input')
        self._proto.input.extend(self.inputs)
        self._proto.ClearField('output')
        self._proto.output.extend(self.outputs)

        self.repeat_id = self._proto.repeat_id[:]
        self.repeat_param = self._proto.repeat_param
        self.recurrent_param = self._proto.recurrent_param

        # function_instance is invalid
        self.function_instance = None
        return self

    def __getitem__(self, item):
        return self.__dict__[item]

    def _proto_call(self, inputs, n_outputs):
        def join_name(name):
            if self.path_name:
                return '/'.join([self.path_name, name])
            return name
        for v in inputs:
            if isinstance(v, nn.Variable):
                self.owner().parameters[str(v.data.__hash__())] = v
                self.inputs.append(str(v.data.__hash__()))
            else:
                if v in self.owner().variables.values():
                    v_name = list(self.owner().variables.keys())[
                                  list(self.owner().variables.values()).index(v)]
                else:
                    if v.name is not None:
                        self.owner().variables[v.name] = v
                        v_name = v.name
                    else:
                        v_name = join_name(
                            self.type + '_in') if v.name is None else v.name
                        v_name = _get_unique_name(
                            self.owner().var_names, v_name)
                        self.owner().variables[v_name] = v
                        v.name = v_name
                v.required.append(self.name)
                self.inputs.append(v_name)

        n_outputs = self.function_instance.min_outputs() if n_outputs < 0 else n_outputs
        input_vars = [nn.Variable(v.shape) for v in inputs]
        output_vars = [nn.Variable() for _ in range(n_outputs)]

        # Obtain output shape from a function_instance
        self.function_instance.setup(input_vars, output_vars)

        # Release this function instance immediately.
        # function_instance cannot call setup() multiple times for current situation.
        self.function_instance = None
        for i, v in enumerate(output_vars):
            v_name = join_name(self.type + '_out')
            renamed = False
            if hasattr(self.owner(), 'renaming'):
                n_v_name = self.owner().renaming(i, v_name)
                if n_v_name != v_name:
                    renamed = True
                v_name = n_v_name
            n_v_name = _get_unique_name(self.owner().var_names, v_name)
            if renamed and n_v_name != v_name:
                raise ValueError(
                    "User specified name is not unique in name space!")
            v_name = n_v_name
            pv = self.owner().variables[v_name] = ProtoVariable(
                v.shape, v_name, v.need_grad, 'Buffer')
            pv.parent = self.name
            self.outputs.append(v_name)
        self.owner().functions[self.name] = self

        # Since a new function is added to graph, graph should be
        # set to dirty
        current_graph_builder().dirty()
        if len(self.outputs) == 1:
            return self.owner().variables[self.outputs[0]]
        return tuple([self.owner().variables[k] for k in self.outputs])

    def graph_call(self, **kwargs):
        """This function create function instance for generating
        computation graph.
        """
        from nnabla.parameter import get_parameter_or_create
        from nnabla.utils.load_function import _create_function_instance
        inputs = []
        batch_size = kwargs.get('batch_size', 1)
        for k in self.inputs:
            pv = self.owner().variables[k] if k in self.owner().variables \
                else self.owner().parameters[k]
            if pv.type == "Parameter":
                pv.variable_instance = get_parameter_or_create(
                    pv.name, pv.shape, pv.initializer)
                pv.variable_instance.need_grad = pv.need_grad
            elif pv.variable_instance is None:
                raise ValueError(
                    "Input variable:{} should not be None.".format(pv.name))
            inputs.append(pv.variable_instance)
        if self.function_instance is None:
            # Resolve function params, Such as Reshape, Broadcast...
            # Replace -1 with batch_size
            pf = self.clone(self.owner())
            if pf.args is not None and 'shape' in pf.args and pf.args['shape'] and pf.args['shape'][0] == -1:
                pf.args['shape'] = [batch_size] + pf.args['shape'][1:]
            self.function_instance = _create_function_instance(
                self.owner().current_context, pf.proto)
        outputs = self.function_instance(*inputs, n_outputs=len(self.outputs),
                                         auto_forward=nn.get_auto_forward())
        if not isinstance(outputs, tuple):
            outputs = (outputs,)

        func_outputs = [self.owner().variables[k] for k in self.outputs]
        for p, o in zip(func_outputs, outputs):
            p.variable_instance = o

    def __call__(self, *args, **kwargs):
        return self._proto_call(*args, **kwargs)
