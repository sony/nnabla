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

'''
Save network structure into file.

'''

from collections import OrderedDict
import google.protobuf.text_format as text_format
import numpy
import os
import re
import shutil
import tempfile
import zipfile

from nnabla import save_parameters
from nnabla.logger import logger
from nnabla.parameter import get_parameters
from nnabla.utils import nnabla_pb2
from nnabla.utils.save_function import _create_function_nntxt
from nnabla.utils.nnp_format import nnp_version

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------


def _get_unique_function_name(function_type, functions):
    '''Get a unique function name.

    Args:
        function_type(str): Name of Function. Ex) Convolution, Affine
        functions(OrderedDict of (str, Function)

    Returns: str
        A unique function name
    '''
    function_name = function_name_base = function_type
    count = 2
    while function_name in functions:
        function_name = '{}_{}'.format(function_name_base, count)
        count += 1
    return function_name


def _get_unique_variable_name(vname, variables):
    '''Get a unique variable name.

    Args:
        vname(str): A candidate name.
        variable(OrderedDict of str and Variable)

    Returns: str
        A unique variable name
    '''
    count = 2
    vname_base = vname
    while vname in variables:
        vname = '{}_{}'.format(vname_base, count)
        count += 1
    return vname


def _get_variable_name_or_register(var, variables, names, params, prefix):
    '''
    Args:
        var (~nnabla.Variable)
        variables (OrderedDict)
        names (dict): Force name table, Variable -> str
        params (dict): NdArray -> str
        prefix(str)
    '''
    if var not in variables.values():
        vname = prefix
        if var.data in params:
            vname = params[var.data]
        elif var in names:
            vname = names[var]
        vname = _get_unique_variable_name(vname, variables)
        variables[vname] = var
    else:
        vname = list(variables.keys())[list(variables.values()).index(var)]

    return vname


def _get_net_variables(net):
    inputs = []
    outputs = []
    variables = {}
    net_parameters = []
    for v in net.variable:
        variables[v.name] = v
        if v.type == 'Parameter':
            net_parameters.append(v.name)
    for f in net.function:
        for v in f.input:
            if variables[v].type == 'Buffer':
                inputs.append(v)
        for v in f.output:
            if variables[v].type == 'Buffer':
                outputs.append(v)

    net_inputs = list(set(inputs) - set(outputs))
    net_outputs = list(set(outputs) - set(inputs))
    return net_inputs, net_outputs, net_parameters


def _create_global_config(ctx):
    g = nnabla_pb2.GlobalConfig()
    g.default_context.backends[:] = ctx.backend
    g.default_context.array_class = ctx.array_class
    g.default_context.device_id = ctx.device_id
    return g


def _create_training_config(max_epoch, iter_per_epoch, save_best):
    t = nnabla_pb2.TrainingConfig()
    t.max_epoch = max_epoch
    t.iter_per_epoch = iter_per_epoch
    t.save_best = save_best
    return t


def _create_dataset(name, uri, cache_dir, variables, shuffle, batch_size, no_image_normalization):
    d = nnabla_pb2.Dataset()
    d.name = name
    d.uri = uri
    if cache_dir is not None:
        d.cache_dir = cache_dir
    d.shuffle = shuffle
    d.batch_size = batch_size
    d.variable.extend(variables)
    d.no_image_normalization = no_image_normalization
    return d


def _get_network_sink(outputs):
    import nnabla.functions as F
    outputs = [o for o in outputs.values()]
    return F.sink(*outputs)


def _create_network(net):
    n = nnabla_pb2.Network()
    n.name = net['name']
    n.batch_size = net['batch_size']

    # List (dict: name -> Variable) of outputs.
    outputs = net['outputs']
    sink = _get_network_sink(outputs)

    # Create force name table: Variable -> name.
    names = {}
    names.update(net['names'])
    names.update(outputs)
    # Reverse dict: Variable --> Name
    names = {v: k for k, v in names.items()}

    # Create table: NdArray -> str
    # (Use Ndarray instead of Variable because parameter variable might be
    # unlinked)
    params = {v.data: k for k, v in get_parameters(grad_only=False).items()}

    # ----------------------------------------------------------------------
    # Parse graph to get variables and functions
    # ----------------------------------------------------------------------
    variables = OrderedDict()
    functions = OrderedDict()

    def collect_info(func):
        # Collect information.
        function_type = func.info.type_name
        if function_type == 'Sink':
            return
        function_name = _get_unique_function_name(function_type, functions)
        functions[function_name] = {
            'type': function_type,
            'args': func.info.args,
            'inputs': [],
            'outputs': []
        }
        for i in func.inputs:
            base_name = '{}_Input'.format(function_name)
            vname = _get_variable_name_or_register(
                i, variables, names, params, base_name)
            functions[function_name]['inputs'].append(vname)
        for o in func.outputs:
            base_name = '{}_Output'.format(function_name)
            vname = _get_variable_name_or_register(
                o, variables, names, params, base_name)
            functions[function_name]['outputs'].append(vname)

    sink.visit(collect_info)

    # ----------------------------------------------------------------------
    # Convert variables and functions into proto
    # ----------------------------------------------------------------------
    for name, variable in variables.items():
        v = n.variable.add()
        v.name = name
        shape = list(numpy.array(variable.d).shape)
        if variable.data in params:
            v.type = 'Parameter'
        else:
            v.type = 'Buffer'
            # TODO: The first dimension is always considered as batch size.
            if len(shape) > 0:
                shape[0] = -1
        v.shape.dim.extend(shape)
        # ----------------------------------------------------------------------
        # Add info to variable
        # ----------------------------------------------------------------------
        # TODO: Only required for Parameter variables?
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
        if function['type'] == 'Reshape':
            # TODO: The first dimension is always considered as batch size.
            function['args']['shape'][0] = -1
        _create_function_nntxt(f, name, function)

    return n


def _create_optimizer(name, solver, network, dataset, weight_decay, lr_decay, lr_decay_interval, update_interval):
    o = nnabla_pb2.Optimizer()
    o.name = name
    o.network_name = network.name
    o.dataset_name = dataset.name
    o.solver.type = re.sub(r'(|Cuda)$', '', str(solver.name))
    if o.solver.type == 'Adadelta':
        o.solver.adadelta_param.lr = solver.info['lr']
        o.solver.adadelta_param.decay = solver.info['decay']
        o.solver.adadelta_param.eps = solver.info['eps']
    elif o.solver.type == 'Adagrad':
        o.solver.adagrad_param.lr = solver.info['lr']
        o.solver.adagrad_param.eps = solver.info['eps']
    elif o.solver.type == 'Adam':
        o.solver.adam_param.alpha = solver.info['alpha']
        o.solver.adam_param.beta1 = solver.info['beta1']
        o.solver.adam_param.beta2 = solver.info['beta2']
        o.solver.adam_param.eps = solver.info['eps']
    elif o.solver.type == 'Adamax':
        o.solver.adamax_param.alpha = solver.info['alpha']
        o.solver.adamax_param.beta1 = solver.info['beta1']
        o.solver.adamax_param.beta2 = solver.info['beta2']
        o.solver.adamax_param.eps = solver.info['eps']
    elif o.solver.type == 'Momentum':
        o.solver.momentum_param.lr = solver.info['lr']
        o.solver.momentum_param.momentum = solver.info['momentum']
    elif o.solver.type == 'Nesterov':
        o.solver.nesterov_param.lr = solver.info['lr']
        o.solver.nesterov_param.momentum = solver.info['momentum']
    elif o.solver.type == 'RMSprop':
        o.solver.rmsprop_param.lr = solver.info['lr']
        o.solver.rmsprop_param.decay = solver.info['decay']
        o.solver.rmsprop_param.eps = solver.info['eps']
    elif o.solver.type == 'Sgd':
        o.solver.sgd_param.lr = solver.info['lr']
    inputs, outputs, params = _get_net_variables(network)
    o.solver.weight_decay = weight_decay
    o.solver.lr_decay = lr_decay
    o.solver.lr_decay_interval = lr_decay_interval
    o.update_interval = update_interval
    for n, inp in enumerate(inputs):
        d = o.data_variable.add()
        d.variable_name = inp
        d.data_name = dataset.variable[n]
    for out in outputs:
        d = o.loss_variable.add()
        d.variable_name = out
    for param in params:
        d = o.parameter_variable.add()
        d.variable_name = param
        d.learning_rate_multiplier = 1.0
    return o


def _create_monitor(name, network, dataset):
    m = nnabla_pb2.Monitor()
    m.name = name
    m.network_name = network.name
    m.dataset_name = dataset.name
    inputs, outputs, params = _get_net_variables(network)
    for n, inp in enumerate(inputs):
        d = m.data_variable.add()
        d.variable_name = inp
        d.data_name = dataset.variable[n]
    for out in outputs:
        d = m.monitor_variable.add()
        d.type = 'Error'
        d.variable_name = out
    return m


def _create_executor(name, network, data, output, remap=None):
    '''
    '''
    if remap is None:
        remap = {}
    e = nnabla_pb2.Executor()
    e.name = name
    e.network_name = network.name
    _, _, params = _get_net_variables(network)
    var_dict = {v.name: v for v in network.variable}
    for vname in data:
        try:
            _ = var_dict[vname]
        except KeyError:
            raise KeyError("{} not found in {}".format(vname, network.name))
        dv = e.data_variable.add()
        dv.variable_name = vname
        dv.data_name = remap.get(vname, vname)
    for vname in output:
        try:
            _ = var_dict[vname]
        except KeyError:
            raise KeyError("{} not found in {}".format(vname, network.name))
        ov = e.output_variable.add()
        ov.variable_name = vname
        ov.data_name = remap.get(vname, vname)
    for param in params:
        d = e.parameter_variable.add()
        d.variable_name = param
    return e
# ----------------------------------------------------------------------
# Helper functions (END)
# ----------------------------------------------------------------------


def create_proto(contents, include_params=False):
    proto = nnabla_pb2.NNablaProtoBuf()
    if 'global_config' in contents:
        proto.global_config.MergeFrom(
            _create_global_config(contents['global_config']['default_context'])
        )
    if 'training_config' in contents:
        proto.training_config.MergeFrom(
            _create_training_config(contents['training_config']['max_epoch'],
                                    contents['training_config'][
                                        'iter_per_epoch'],
                                    contents['training_config']['save_best']))
    networks = {}
    if 'networks' in contents:
        proto_nets = []
        for net in contents['networks']:
            networks[net['name']] = _create_network(net)
            proto_nets.append(networks[net['name']])
        proto.network.extend(proto_nets)
    datasets = {}
    if 'datasets' in contents:
        proto_datasets = []
        for d in contents['datasets']:
            if 'cache_dir' in d:
                cache_dir = d['cache_dir']
            else:
                cache_dir = None
            datasets[d['name']] = _create_dataset(d['name'],
                                                  d['uri'],
                                                  cache_dir,
                                                  d['variables'],
                                                  d['shuffle'],
                                                  d['batch_size'],
                                                  d['no_image_normalization'])
            proto_datasets.append(datasets[d['name']])
        proto.dataset.extend(proto_datasets)
    if 'optimizers' in contents:
        proto_optimizers = []
        for o in contents['optimizers']:
            proto_optimizers.append(_create_optimizer(o['name'], o['solver'],
                                                      networks[o['network']],
                                                      datasets[o['dataset']],
                                                      o['weight_decay'],
                                                      o['lr_decay'],
                                                      o['lr_decay_interval'],
                                                      o['update_interval']))
        proto.optimizer.extend(proto_optimizers)
    if 'monitors' in contents:
        proto_monitors = []
        for m in contents['monitors']:
            proto_monitors.append(_create_monitor(m['name'],
                                                  networks[m['network']],
                                                  datasets[m['dataset']]))
        proto.monitor.extend(proto_monitors)
    if 'executors' in contents:
        proto_executors = []
        for e in contents['executors']:
            proto_executors.append(
                _create_executor(e['name'], networks[e['network']],
                                 e['data'], e['output'], e.get('remp', {})))
        proto.executor.extend(proto_executors)

    if include_params is True:
        params = get_parameters(grad_only=False)
        for variable_name, variable in params.items():
            parameter = proto.parameter.add()
            parameter.variable_name = variable_name
            parameter.shape.dim.extend(variable.shape)
            parameter.data.extend(numpy.array(variable.d).flatten().tolist())
            parameter.need_grad = variable.need_grad

    return proto


def save(filename, contents, include_params=False):
    '''Save network definition, inference/training execution
    configurations etc.

    Args:
        filename (str): Filename to store information. The file
            extension is used to determine the saving file format.
            ``.nnp``: (Recommended) Creating a zip archive with nntxt (network
            definition etc.) and h5 (parameters).
            ``.nntxt``: Protobuf in text format.
            ``.protobuf``: Protobuf in binary format (unsafe in terms of
             backward compatibility).
        contents (dict): Information to store.
        include_params (bool): Includes parameter into single file. This is
            ignored when the extension of filename is nnp.

    Example:
        The following example creates a two inputs and two
        outputs MLP, and save the network structure and the initialized
        parameters.

        .. code-block:: python

            import nnabla as nn
            import nnabla.functions as F
            import nnabla.parametric_functions as PF

            x0 = nn.Variable([batch_size, 100])
            x1 = nn.Variable([batch_size, 100])
            h1_0 = PF.affine(x0, 100, name='affine1_0')
            h1_1 = PF.affine(x1, 100, name='affine1_0')
            h1 = F.tanh(h1_0 + h1_1)
            h2 = F.tanh(PF.affine(h1, 50, name='affine2'))
            y0 = PF.affine(h2, 10, name='affiney_0')
            y1 = PF.affine(h2, 10, name='affiney_1')

            contents = {
                'networks': [
                    {'name': 'net1',
                     'batch_size': batch_size,
                     'outputs': {'y0': y0, 'y1': y1},
                     'names': {'x0': x0, 'x1': x1}}],
                'executors': [
                    {'name': 'runtime',
                     'network': 'net1',
                     'data': ['x0', 'x1'],
                     'output': ['y0', 'y1']}]}
            save('net.nnp', contents)


        To get a trainable model, use following code instead.

        .. code-block:: python

            contents = {
            'global_config': {'default_context': ctx},
            'training_config':
                {'max_epoch': args.max_epoch,
                 'iter_per_epoch': args_added.iter_per_epoch,
                 'save_best': True},
            'networks': [
                {'name': 'training',
                 'batch_size': args.batch_size,
                 'outputs': {'loss': loss_t},
                 'names': {'x': x, 'y': t, 'loss': loss_t}},
                {'name': 'validation',
                 'batch_size': args.batch_size,
                 'outputs': {'loss': loss_v},
                 'names': {'x': x, 'y': t, 'loss': loss_v}}],
            'optimizers': [
                {'name': 'optimizer',
                 'solver': solver,
                 'network': 'training',
                 'dataset': 'mnist_training',
                 'weight_decay': 0,
                 'lr_decay': 1,
                 'lr_decay_interval': 1,
                 'update_interval': 1}],
            'datasets': [
                {'name': 'mnist_training',
                 'uri': 'MNIST_TRAINING',
                 'cache_dir': args.cache_dir + '/mnist_training.cache/',
                 'variables': {'x': x, 'y': t},
                 'shuffle': True,
                 'batch_size': args.batch_size,
                 'no_image_normalization': True},
                {'name': 'mnist_validation',
                 'uri': 'MNIST_VALIDATION',
                 'cache_dir': args.cache_dir + '/mnist_test.cache/',
                 'variables': {'x': x, 'y': t},
                 'shuffle': False,
                 'batch_size': args.batch_size,
                 'no_image_normalization': True
                 }],
            'monitors': [
                {'name': 'training_loss',
                 'network': 'validation',
                 'dataset': 'mnist_training'},
                {'name': 'validation_loss',
                 'network': 'validation',
                 'dataset': 'mnist_validation'}],
            }


    '''
    _, ext = os.path.splitext(filename)
    if ext == '.nntxt' or ext == '.prototxt':
        logger.info("Saving {} as prototxt".format(filename))
        proto = create_proto(contents, include_params)
        with open(filename, 'w') as file:
            text_format.PrintMessage(proto, file)
    elif ext == '.protobuf':
        logger.info("Saving {} as protobuf".format(filename))
        proto = create_proto(contents, include_params)
        with open(filename, 'wb') as file:
            file.write(proto.SerializeToString())
    elif ext == '.nnp':
        logger.info("Saving {} as nnp".format(filename))
        try:
            tmpdir = tempfile.mkdtemp()
            save('{}/network.nntxt'.format(tmpdir),
                 contents, include_params=False)

            with open('{}/nnp_version.txt'.format(tmpdir), 'w') as file:
                file.write('{}\n'.format(nnp_version()))

            save_parameters('{}/parameter.protobuf'.format(tmpdir))

            with zipfile.ZipFile(filename, 'w') as nnp:
                nnp.write('{}/nnp_version.txt'.format(tmpdir),
                          'nnp_version.txt')
                nnp.write('{}/network.nntxt'.format(tmpdir), 'network.nntxt')
                nnp.write('{}/parameter.protobuf'.format(tmpdir),
                          'parameter.protobuf')
        finally:
            shutil.rmtree(tmpdir)
