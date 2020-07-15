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

"""
This module implements a concept `graph_def`, which is like a protocol buffer representation,
which is a kind of storage-oriented format that might be used as network manipulation, format
converting, and so on.
User may create this protocol buffer representation by passing into a nn.ProtoVariable() to the
computation graph. `graph_def` will be recorded and generated automatically.

# TODO:
#   - context priority hierarchy testing
#   - default context setting change
#   - batch size change
#   - operators of nn.ProtoVariable()

"""

import nnabla as nn
import numpy as np
import weakref
from collections import OrderedDict
from contextlib import contextmanager

__graph_members__ = ['inputs', 'outputs',
                     'parameters', 'functions', 'variables']


def load(filename, batch_size=None, exclude_parameter=False, parameter_only=False):
    """load
    Load network from files

    Args:
        filenames (list): List of filenames.
        batch_size (int): The batch size expected to be set
        exclude_parameter (bool): If True, only load model, not load parameters of this model.
        Default is False.
        parameter_only (bool): If True, only load model parameters. Default is False.
    Returns:
        ProtoGraph object which might contain one or multiple ProtoNetwork objects.

    Example:
        The following example loads a model and generate the output variable
        through this model:

        .. code-block:: python
        import nnabla as nn

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
        g = nn.graph_def.load("fusion_net.nnp") # with parameters
        x = nn.Variable((64, 3, 32, 32))
        y = g(x) # create computation graph by passing in nn.Variable()
        y.forward()
        ...
        # You may use your special context:
        with context_scope(ctx):
           y = g(x)
           y.forward()
    """
    import google.protobuf.text_format as text_format
    import os
    import shutil
    import tempfile
    import zipfile

    from nnabla.logger import logger
    from nnabla.utils import nnabla_pb2

    def _load_files():
        proto = nnabla_pb2.NNablaProtoBuf()
        for filename in filenames:
            _, ext = os.path.splitext(filename)

            # TODO: Here is some known problems.
            #   - Even when protobuf file includes network structure,
            #     it will not loaded.
            #   - Even when prototxt file includes parameter,
            #     it will not loaded.

            if ext in ['.nntxt', '.prototxt']:
                if not parameter_only:
                    with open(filename, 'rt') as f:
                        try:
                            text_format.Merge(f.read(), proto)
                        except:
                            logger.critical(
                                'Failed to read {}.'.format(filename))
                            logger.critical(
                                '2 byte characters may be used for file name or folder name.')
                            raise
                if len(proto.parameter) > 0:
                    if not exclude_parameter:
                        nn.load_parameters(filename)
            elif ext in ['.protobuf', '.h5']:
                if not exclude_parameter:
                    nn.load_parameters(filename)
                else:
                    logger.info('Skip loading parameter.')

            elif ext == '.nnp':
                try:
                    tmpdir = tempfile.mkdtemp()
                    with zipfile.ZipFile(filename, 'r') as nnp:
                        for name in nnp.namelist():
                            _, ext = os.path.splitext(name)
                            if name == 'nnp_version.txt':
                                nnp.extract(name, tmpdir)
                                with open(os.path.join(tmpdir, name), 'rt') as f:
                                    # TODO currently do nothing with version.
                                    pass
                            elif ext in ['.nntxt', '.prototxt']:
                                nnp.extract(name, tmpdir)
                                if not parameter_only:
                                    with open(os.path.join(tmpdir, name), 'rt') as f:
                                        text_format.Merge(f.read(), proto)
                                if len(proto.parameter) > 0:
                                    if not exclude_parameter:
                                        nn.load_parameters(
                                            os.path.join(tmpdir, name))
                            elif ext in ['.protobuf', '.h5']:
                                nnp.extract(name, tmpdir)
                                if not exclude_parameter:
                                    nn.load_parameters(
                                        os.path.join(tmpdir, name))
                                else:
                                    logger.info('Skip loading parameter.')
                finally:
                    shutil.rmtree(tmpdir)
        return proto

    if isinstance(filename, str):
        filenames = [filename]
    rng = np.random.RandomState(0)
    parameter_scope = OrderedDict()
    with nn.parameter_scope('', parameter_scope):
        proto = _load_files()
    g = ProtoGraph.from_proto(proto,
                              param_scope=parameter_scope,
                              rng=rng)
    return g


def save(filename, content, include_parameters=False, variable_batch_size=True):
    """Save network definition, inference/training execution
    configurations etc.

    Args:
        filename (str or file object): Filename to store information. The file
            extension is used to determine the saving file format.
            ``.nnp``: (Recommended) Creating a zip archive with nntxt (network
            definition etc.) and h5 (parameters).
            ``.nntxt``: Protobuf in text format.
            ``.protobuf``: Protobuf in binary format (unsafe in terms of
             backward compatibility).
        content (list): Currently only ProtoGraph or PhotoNetwork objects are
                       supported
        include_params (bool): Includes parameter into single file. This is
            ignored when the extension of filename is nnp.
        variable_batch_size (bool): Not used yet

    Example:
        The following example creates a two inputs and two
        outputs MLP, and save the network structure and the initialized
        parameters.

        .. code-block:: python

            import nnabla as nn
            import nnabla.functions as F
            import nnabla.parametric_functions as PF
            from nnabla.utils.save import save

            def mlp_module(x1, x2):
                h1_0 = PF.affine(x0, 100, name='affine1_0')
                h1_1 = PF.affine(x1, 100, name='affine1_0')
                h1 = F.tanh(h1_0 + h1_1)
                h2 = F.tanh(PF.affine(h1, 50, name='affine2'))
                y0 = PF.affine(h2, 10, name='affiney_0')
                y1 = PF.affine(h2, 10, name='affiney_1')
                return y0, y1

            with nn.graph_def().graph() as g:
                x1 = nn.ProtoVariable((64, 100))
                x2 = nn.ProtoVariable((64, 100))
                y1, y2 = mlp_module(x1, x2)

            g.save("mlp_net.nnp")

    """
    import os
    from six import iteritems
    from nnabla.utils import nnabla_pb2
    from nnabla.logger import logger
    import shutil
    import tempfile
    import zipfile
    import google.protobuf.text_format as text_format
    from nnabla.utils.nnp_format import nnp_version

    def _save_parameters(path, params):
        _, ext = os.path.splitext(path)
        if ext == '.h5':
            # TODO temporary work around to suppress FutureWarning message.
            import warnings
            warnings.simplefilter('ignore', category=FutureWarning)
            import h5py
            with h5py.File(path, 'w') as hd:
                for i, (k, v) in enumerate(iteritems(params)):
                    hd[k] = v.d
                    hd[k].attrs['need_grad'] = v.need_grad
                    # To preserve order of parameters
                    hd[k].attrs['index'] = i
        elif ext == '.protobuf':
            proto = nnabla_pb2.NNablaProtoBuf()
            for variable_name, variable in params.items():
                parameter = proto.parameter.add()
                parameter.variable_name = variable_name
                parameter.shape.dim.extend(variable.shape)
                parameter.data.extend(np.array(variable.d).flatten().tolist())
                parameter.need_grad = variable.need_grad

            with open(path, "wb") as f:
                f.write(proto.SerializeToString())
        else:
            logger.critical('Only supported hdf5 or protobuf.')
            assert False
        logger.info("Parameter save ({}): {}".format(ext, path))

    def _save_nntxt(save_file, proto):
        logger.info("Saving {} as prototxt".format(save_file))
        with open(save_file, 'w') as file:
            text_format.PrintMessage(proto, file)

    def _save_pb(save_file, proto):
        logger.info("Saving {} as protobuf".format(save_file))
        with open(save_file, 'wb') as file:
            file.write(proto.SerializeToString())

    def _save_nnp(save_file, proto, params):
        logger.info("Saving {}".format(save_file))
        try:
            tmpdir = tempfile.mkdtemp()
            _save_nntxt('{}/network.nntxt'.format(tmpdir), proto)
            with open('{}/nnp_version.txt'.format(tmpdir), 'w') as file:
                file.write('{}\n'.format(nnp_version()))

            if params:
                _save_parameters(
                    '{}/parameter.protobuf'.format(tmpdir), params)

            with zipfile.ZipFile(save_file, 'w') as nnp:
                nnp.write('{}/nnp_version.txt'.format(tmpdir),
                          'nnp_version.txt')
                nnp.write('{}/network.nntxt'.format(tmpdir), 'network.nntxt')
                nnp.write('{}/parameter.protobuf'.format(tmpdir),
                          'parameter.protobuf')
        finally:
            shutil.rmtree(tmpdir)

    def _create_proto(contents, include_params, variable_batch_size):
        params = None
        for g in contents:
            if isinstance(g, ProtoGraph):
                proto = g.as_proto(include_params)
                params = g.get_parameters()
                break
            if isinstance(g, ProtoNetwork):
                proto = g.owner().as_proto(
                    include_parameter=include_params, networks=[g])
                params = g.owner().get_parameters()
                break
        return proto, params

    _, ext = os.path.splitext(os.path.basename(filename))
    p, parameters = _create_proto(
        content, include_parameters, variable_batch_size)
    if ext == '.nntxt' or ext == '.prototxt':
        _save_nntxt(filename, p)
    elif ext == '.protobuf':
        _save_pb(filename, p)
    elif ext == '.nnp':
        _save_nnp(filename, p, parameters)


def _get_unique_name(names, prefix):
    if prefix in names:
        name = "{}_{}".format(prefix, names[prefix])
        names[prefix] += 1
    else:
        name = prefix
        names[prefix] = 1
    return name


def _create_initializer(v, rng):
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
    elif v.initializer.type == 'Uniform':
        initializer = UniformInitializer(
            lim=[-v.initializer.multiplier, v.initializer.multiplier], rng=rng)
    elif v.initializer.type == 'UniformAffineGlorot':
        initializer = (lambda shape: UniformInitializer(calc_uniform_lim_glorot(
            shape[0], np.prod(shape[1:])), rng=rng)(shape) * v.initializer.multiplier)
    elif v.initializer.type == 'UniformConvolutionGlorot':
        initializer = (lambda shape: UniformInitializer(calc_uniform_lim_glorot(
            shape[-3], shape[0], kernel=shape[-2:]), rng=rng)(shape) * v.initializer.multiplier)
    elif v.initializer.type == 'Range':
        initializer = (lambda shape: RangeInitializer(0, 1)
                       (shape) * v.initializer.multiplier)
    elif v.initializer.type == 'Constant':
        initializer = ConstantInitializer(value=v.initializer.multiplier)
    else:
        initializer = None

    return initializer


class ProtoNetwork:
    def __init__(self, owner):
        members = ['arguments', 'names'] + __graph_members__
        for key in members:
            self.__dict__[key] = OrderedDict()

        # set built-in super arguments to default value
        self.arguments['batch_size'] = 1
        self.name = None
        self.owner = weakref.ref(owner)

    @property
    def current_context(self):
        return self.owner().current_context

    def __getitem__(self, item):
        return self.__dict__[item]

    def resolve_function_parameters(self, function, params, names):
        inputs = OrderedDict()
        input_index = 0
        for k, v in function.inputs.items():
            if isinstance(v, nn.Variable):
                if v.data in params:
                    v_name = params[v.data]
                else:
                    # some module(parameters) are created as local variable
                    v_name = _get_unique_name(
                        names, "{}_{}".format(function.name, input_index))
                p = self.parameters[v_name] = ProtoVariable(
                    v.shape, v_name, v.need_grad, 'Parameter')
                if v.info:
                    p.info = v.info
                p.required.append(weakref.ref(function))
                p.variable_instance = v
                inputs[v_name] = p
            else:
                inputs[k] = v
            input_index += 1
        function.inputs = inputs

    def commit(self, name, networks, params, names):
        networks[name] = self
        for f in self.functions.values():
            self.resolve_function_parameters(f, params, names)
        with nn.parameter_scope('', self.owner().parameter_scope):
            for k, v in self.parameters.items():
                nn.parameter.set_parameter(k, v.variable_instance)
        self.inputs = OrderedDict([(k, v) for k, v in self.variables.items()
                                   if not v.parent and v.required])
        self.outputs = OrderedDict([(k, v) for k, v in self.variables.items()
                                    if not v.required and (v.parent() if v.parent else None)])
        self.name = name

    def as_proto(self, **kwargs):
        from nnabla.utils import nnabla_pb2
        from nnabla.utils.save_function import _create_function_nntxt
        if kwargs:
            self.arguments.update(kwargs)
        n = nnabla_pb2.Network()
        n.name = self.name
        n.batch_size = self.arguments.get('batch_size', 1)
        variables = OrderedDict(self.variables)
        variables.update(self.parameters)
        functions = self.functions
        for name, variable in variables.items():
            v = n.variable.add()
            v.name = name
            v.type = variable.type
            shape = list(variable.shape)
            v.shape.dim.extend(shape)
            if variable.info:
                i = v.initializer
                i.type = variable.info.initializer.__class__.__name__.replace(
                    'Initializer', '')
                i.multiplier = 0.0
                if i.type == 'Constant':
                    i.multiplier = variable.info.initializer.value
                elif i.type == 'Uniform':
                    i.multiplier = -variable.info.initializer.lim[0]
                elif i.type == 'Normal':
                    i.multiplier = variable.info.initializer.sigma
                else:
                    pass  # TODO Error

        for name, function in functions.items():
            f = n.function.add()
            _create_function_nntxt(f, name, function)

        return n

    def __call__(self, *args, **kwargs):
        with nn.parameter_scope('', self.owner().parameter_scope):
            for pv, v in zip(self.inputs.values(), args):
                if tuple(pv.shape) != tuple(v.shape):
                    raise ValueError("variable {} {} != {}.".format(
                        pv.name, pv.shape, v.shape))
                pv.variable_instance = v
            # TODO: The function sequence cannot be assumed as forward sequence.
            for func in self.functions.values():
                func.graph_call()
            outputs = [v.variable_instance for v in self.outputs.values()]
            if len(outputs) == 1:
                return outputs[0]
            return tuple(outputs)

    @staticmethod
    def from_proto(owner, proto, rng):
        from nnabla.utils.load_function import _create_function_instance
        g = ProtoNetwork(owner)
        g.name = proto.name
        for v in proto.variable:
            if v.type == 'Buffer':
                g.variables[v.name] = ProtoVariable(
                    list(v.shape.dim), v.name, False, v.type)
            elif v.type == 'Parameter':
                g.parameters[v.name] = ProtoVariable(
                    list(v.shape.dim), v.name, True, v.type)
                g.parameters[v.name].initializer = _create_initializer(v, rng)
        for f in proto.function:
            # Here, we temporarily created a function instance for converting
            # arguments from proto-oriented representation to general-purpose representation
            func = _create_function_instance(nn.get_current_context(), f)
            arguments = func.arguments
            pf = g.functions[f.name] = ProtoFunction(None,
                                                     f.type,
                                                     arguments,
                                                     f.name,
                                                     g)
            for n in f.input:
                pv = g.variables[n] if n in g.variables else g.parameters[n]
                pf.inputs[pv.name] = pv
                pv.required.append(weakref.ref(pf))
            for n in f.output:
                pv = g.variables[n] if n in g.variables else g.parameters[n]
                pf.outputs[pv.name] = pv
                pv.parent = weakref.ref(pf)
        g.inputs = OrderedDict([(k, v) for k, v in g.variables.items()
                                if not v.parent and any([r() for r in v.required])])
        g.outputs = OrderedDict([(k, v) for k, v in g.variables.items()
                                 if not (any([r() for r in v.required])) and (v.parent() if v.parent else None)])
        return g

    def save(self, filename, include_parameter=False):
        save(filename, [self], include_parameters=include_parameter)


class ProtoGraph:
    def __contains__(self, item):
        return item in self.__dict__

    def __getattr__(self, item):
        if item in __graph_members__:
            if self.default_graph:
                return self.default_graph[item]
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

    def __init__(self, networks=None):
        self.networks = networks if networks is not None else OrderedDict()
        self.default_graph = None
        self.parameter_scope = OrderedDict()

    def get_parameters(self, grad_only=False):
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
    def from_proto(proto, param_scope=None, rng=None):
        if not rng:
            rng = np.random.RandomState(0)

        g = ProtoGraph()
        if param_scope:
            g.parameter_scope = param_scope

        for p in proto.network:
            g.networks[p.name] = ProtoNetwork.from_proto(g, p, rng)

        g.commit_all()
        return g

    def commit_all(self):
        if self.networks:
            self.default_graph = list(self.networks.values())[0]

    def as_proto(self, include_parameter=False, only_parameter=False, networks=None):
        from nnabla.utils import nnabla_pb2
        proto = nnabla_pb2.NNablaProtoBuf()
        if not only_parameter:
            if networks is None:
                networks = [m.as_proto() for m in self.networks.values()]
            else:
                networks = [m.as_proto() for m in networks]
            proto.network.extend(networks)
        if include_parameter:
            for k, v in self.get_parameters().items():
                parameter = proto.parameter.add()
                parameter.variable_name = k
                parameter.shape.dim.extend(v.shape)
                parameter.data.extend(np.array(v.d).flatten().tolist())
                parameter.need_grad = v.need_grad
        return proto

    def __call__(self, *args, **kwargs):
        if self.networks:
            if len(self.networks) > 1:
                print('Multiple networks are found in current graph,'
                      'the first one: {} is chosen.'.format(
                          list(self.networks.keys())[0]))
            g = list(self.networks.values())[0]
            return g(*args, **kwargs)
        raise ValueError("No network is found in current graph.")

    def save(self, filename, include_parameter=False):
        # TODO: Add commit_network() here.
        save(filename, [self], include_parameters=include_parameter)


class FlatModule:
    """FlatModule is a module-like placeholder for generating
    graph_def from a flat-style network definition. -- without module hierarchy
    """

    @staticmethod
    def get_parameters(recursive=True, grad_only=False, memo=None):
        return nn.get_parameters()

    @staticmethod
    def get_path_name():
        return ''

    def __init__(self):
        self.name = "model"


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
        self.proto_graph = ProtoGraph(self.networks)

    def get_current(self):
        if self.current is not None:
            return self.current
        self.current = ProtoNetwork(self.proto_graph)
        return self.current

    def get_module(self):
        if self.module is not None:
            return self.module
        self.module = FlatModule()
        return self.module

    def dirty(self):
        self.dirty_flag = True

    def is_dirty(self):
        return self.dirty_flag

    def get_graph(self):
        return self.proto_graph

    def begin_module(self, module):
        if not self.stack:
            current = ProtoNetwork(self.proto_graph)
            self.stack.append((current, module))
        else:
            current, _ = self.stack[-1]
            self.stack.append((current, module))
        self.current = current
        self.module = module
        return current

    def end_module(self):
        current, module = self.stack[-1]
        self.stack.pop()
        if self.stack:
            # non-top module
            self.current, self.module = self.stack[-1]
        else:
            # Top module
            self.commit_network(current, module)
            self.current = None
            self.module = None

    def commit_network(self, current, module):
        if self.graph_name:
            g_name = self.graph_name
            self.graph_name = None
        else:
            g_name = module.name
        g_name = _get_unique_name(self.names, g_name)
        params = {v.data: k for k, v in module.get_parameters().items()}
        current.commit(g_name, self.networks, params, self.names)
        self.proto_graph.commit_all()
        self.dirty_flag = False


@contextmanager
def graph_name(name):
    g_name = proto_graph_builder.graph_name
    proto_graph_builder.graph_name = name
    yield proto_graph_builder.current
    proto_graph_builder.graph_name = g_name


previous_graph_builder = None
proto_graph_builder = ProtoGraphBuilder()


@contextmanager
def graph(**kwargs):
    global previous_graph_builder, proto_graph_builder
    previous_graph_builder = proto_graph_builder
    proto_graph_builder = ProtoGraphBuilder(**kwargs)
    yield proto_graph_builder.get_graph()
    if proto_graph_builder.is_dirty():
        proto_graph_builder.commit_network(current_network(), current_module())
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
    global proto_graph_builder
    proto_graph_builder = ProtoGraphBuilder()


def get_default_graph(*args, **kwargs):
    '''This function obtain current default graph_def.

    Example:

        .. code-block:: python

            resunit = ResUnit(16)
            input = nn.ProtoVariable((64, 3, 32, 32))
            y = resunit(input)
            graph_def = nn.graph_def.get_graph_graph()
    '''
    network_name = None
    if args:
        if isinstance(args[0], str):
            network_name = args[0]
    else:
        network_name = kwargs.get('name', None)

    if proto_graph_builder.is_dirty():
        proto_graph_builder.commit_network(current_network(), current_module())

    if network_name is None:
        return proto_graph_builder.proto_graph

    return proto_graph_builder.proto_graph.networks.get(network_name, None)


def get_default_graph_by_variable(proto_variable):
    if proto_graph_builder.is_dirty():
        proto_graph_builder.commit_network(current_network(), current_module())

    for network in proto_graph_builder.proto_graph.networks.values():
        if proto_variable in network.outputs.values():
            return network
    raise ValueError(
        "{} is not the output variable of any network".format(proto_variable.name))


class ProtoSize:
    def __init__(self, name, default=1):
        self.name = name
        self.default = default


class ProtoVariable:
    def __init__(self, shape, name=None, need_grad=False, var_type='Buffer'):
        self.name = name
        self.shape = shape
        self.need_grad = need_grad
        self.parent = None
        self.required = []
        self.type = var_type
        self.info = None
        self.initializer = None
        self.variable_instance = None

    def shape(self):
        arguments = current_network().augments
        shape = []
        for v in self.shape:
            if isinstance(v, ProtoSize):
                if v.name in arguments:
                    shape.append(arguments[v.name])
                else:
                    raise ValueError(
                        'super argument {} is not declared.'.format(v.name))
            else:
                shape.append(v)
        return tuple(shape)

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
    def __init__(self, func, f_type, args, name=None, owner=None):
        self.type = f_type
        if name is None:
            self.path_name = current_module().get_path_name()
            func_name = '/'.join([self.path_name, self.type]
                                 ) if self.path_name else self.type
            self.name = _get_unique_name(current_network().names, func_name)
            self.owner = weakref.ref(current_network())
        else:
            self.name = name
            self.owner = weakref.ref(owner)
        self.args = args
        self.inputs = OrderedDict()
        self.outputs = OrderedDict()

        # Keep a function_instance to obtain output shape later.
        self.function_instance = func

    def __getitem__(self, item):
        if item in ['inputs', 'outputs']:
            d = self.__dict__[item]
            return list(d.keys())
        return self.__dict__[item]

    def _proto_call(self, inputs, n_outputs):
        def join_name(name):
            if self.path_name:
                return '/'.join([self.path_name, name])
            return name
        for v in inputs:
            if isinstance(v, nn.Variable):
                self.inputs[str(v.data.__hash__())] = v
            else:
                if v in self.owner().variables.values():
                    v_name = list(self.owner().variables.keys())[
                                  list(self.owner().variables.values()).index(v)]
                else:
                    v_name = join_name(
                        self.type + '_in') if v.name is None else v.name
                    v_name = _get_unique_name(self.owner().names, v_name)
                    self.owner().variables[v_name] = v
                    v.name = v_name
                v.required.append(weakref.ref(self))
                self.inputs[v_name] = v

        n_outputs = self.function_instance.min_outputs() if n_outputs < 0 else n_outputs
        input_vars = [nn.Variable(v.shape) for v in inputs]
        output_vars = [nn.Variable() for _ in range(n_outputs)]

        # Obtain output shape from a function_instance
        self.function_instance.setup(input_vars, output_vars)

        # Release this function instance immediately.
        # function_instance cannot call setup() multiple times for current situation.
        self.function_instance = None
        for v in output_vars:
            v_name = join_name(self.type + '_out')
            v_name = _get_unique_name(self.owner().names, v_name)
            p = self.owner().variables[v_name] = ProtoVariable(
                v.shape, v_name, v.need_grad, 'Buffer')
            self.owner().variables[v_name].parent = weakref.ref(self)
            self.outputs[v_name] = p
        self.owner().functions[self.name] = self

        # Since a new function is added to graph, graph should be
        # set to dirty
        current_graph_builder().dirty()
        if len(self.outputs.values()) == 1:
            return list(self.outputs.values())[0]
        return tuple(self.outputs.values())

    def graph_call(self):
        """This function create function instance for generating
        computation graph.
        """
        from nnabla.parameter import get_parameter_or_create
        from nnabla.utils import nnabla_pb2
        from nnabla.utils.save_function import _create_function_nntxt
        from nnabla.utils.load_function import _create_function_instance
        for pv in self.inputs.values():
            if pv.type == 'Parameter':
                pv.variable_instance = get_parameter_or_create(
                    pv.name, pv.shape, pv.initializer)
            elif pv.variable_instance is None:
                raise ValueError(
                    "Input variable:{} should not be None.".format(pv.name))
        inputs = [v.variable_instance for v in self.inputs.values()]
        if self.function_instance is None:
            function_proto = nnabla_pb2.Function()
            _create_function_nntxt(function_proto, self.name, self)
            self.function_instance = _create_function_instance(
                self.owner().current_context, function_proto)
        outputs = self.function_instance(*inputs, n_outputs=len(self.outputs),
                                         auto_forward=nn.get_auto_forward())
        if not isinstance(outputs, tuple):
            outputs = (outputs,)

        for p, o in zip(self.outputs.values(), outputs):
            p.variable_instance = o

    def __call__(self, *args, **kwargs):
        return self._proto_call(*args, **kwargs)
