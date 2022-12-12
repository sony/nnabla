# Copyright 2017,2018,2019,2020,2021 Sony Corporation.
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

'''
Save network structure into file.

'''

import os
import re
import types

import nnabla as nn
import numpy
from collections import OrderedDict
from nnabla.logger import logger
from nnabla.parameter import get_parameters
from nnabla.utils import nnabla_pb2
from nnabla.utils.get_file_handle import FileHandlerContext, get_default_file_savers, save_files


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------


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


def _format_opti_config_for_states_checkpoint(ctx, contents):
    ctx.optimizers = OrderedDict()
    for opti_i in contents['optimizers']:
        opt = types.SimpleNamespace(**opti_i)
        ctx.optimizers[opt.name] = types.SimpleNamespace(optimizer=opt)


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


def _create_network(ctx, net, variable_batch_size):
    names = dict(net['names'])
    names.update(net['outputs'])
    g = nn.graph_def.create_graph_from_variable(net['name'], list(net['outputs'].values()), names=names,
                                                parameter_scope=nn.parameter.get_current_parameter_scope())
    n = g.default_graph().as_proto(variable_batch_size=variable_batch_size)
    n.batch_size = net['batch_size']
    ctx.proto_graphs[n.name] = g
    return n


def _create_optimizer(ctx, opti_d, save_solver_in_proto):
    o = nnabla_pb2.Optimizer()
    dataset = None

    datasets = ctx.datasets
    name = opti_d['name']
    solver = opti_d['solver']
    # ctx.networks might be missing when optimizer is used in transfer learning
    network = ctx.networks[opti_d['network']] if ctx.networks else None
    dataset_names = opti_d['dataset']
    weight_decay = opti_d['weight_decay']
    lr_decay = opti_d['lr_decay']
    lr_decay_interval = opti_d['lr_decay_interval']
    update_interval = opti_d['update_interval']

    o.name = name
    o.network_name = network.name if network else b'None'

    proto_network = ctx.proto_graphs[opti_d['network']
                                     ].default_graph() if network else None

    # Allow a list or tuple or a string for dataset names.
    if isinstance(dataset_names, tuple):
        dataset_names = list(dataset_names)
    if isinstance(dataset_names, list):
        for dataset_name in dataset_names:
            if dataset_name in datasets:
                o.dataset_name.append(dataset_name)
                dataset = datasets[dataset_name]
            else:
                raise ValueError(
                    "Invalid dataset_name is found in optimizer: {}".format(dataset_name))
    elif isinstance(dataset_names, str):
        dataset_name = dataset_names
        if dataset_name in datasets:
            o.dataset_name.append(dataset_name)
            dataset = datasets[dataset_name]
    if dataset is None:
        # dataset setting in optimizer might be missing when optimizer is used in transfer learning
        # raise ValueError("Dataset is not defined in optimizer.")
        pass
    o.solver.type = re.sub(r'(|Cuda)$', '', str(solver.name))
    if o.solver.type == 'Adadelta':
        o.solver.adadelta_param.lr = solver.info['lr']
        o.solver.adadelta_param.decay = solver.info['decay']
        o.solver.adadelta_param.eps = solver.info['eps']
    elif o.solver.type == 'Adagrad':
        o.solver.adagrad_param.lr = solver.info['lr']
        o.solver.adagrad_param.eps = solver.info['eps']
    elif o.solver.type == 'AdaBelief':
        o.solver.adabelief_param.alpha = solver.info['alpha']
        o.solver.adabelief_param.beta1 = solver.info['beta1']
        o.solver.adabelief_param.beta2 = solver.info['beta2']
        o.solver.adabelief_param.eps = solver.info['eps']
        o.solver.adabelief_param.wd = solver.info['wd']
        o.solver.adabelief_param.amsgrad = solver.info['amsgrad']
        o.solver.adabelief_param.weight_decouple = solver.info['weight_decouple']
        o.solver.adabelief_param.fixed_decay = solver.info['fixed_decay']
        o.solver.adabelief_param.rectify = solver.info['rectify']
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
    elif o.solver.type == 'RMSpropGraves':
        o.solver.rmsprop_graves_param.lr = solver.info['lr']
        o.solver.rmsprop_graves_param.decay = solver.info['decay']
        o.solver.rmsprop_graves_param.momentum = solver.info['momentum']
        o.solver.rmsprop_graves_param.eps = solver.info['eps']
    elif o.solver.type == 'Sgd':
        o.solver.sgd_param.lr = solver.info['lr']
    o.solver.weight_decay = weight_decay
    o.solver.lr_decay = lr_decay
    o.solver.lr_decay_interval = lr_decay_interval
    o.update_interval = update_interval
    for var_name, data_name in opti_d.get('data_variables', {}).items():
        d = o.data_variable.add()
        d.variable_name = var_name
        d.data_name = data_name
    if proto_network:
        for loss_name in opti_d.get('loss_variables', proto_network.outputs):
            d = o.loss_variable.add()
            d.variable_name = loss_name
    solver_params = solver.get_parameters()
    network_keys = proto_network.parameters.keys(
    ) if proto_network else nn.get_parameters().keys()
    for param in network_keys:
        d = o.parameter_variable.add()
        d.variable_name = param
        d.learning_rate_multiplier = 1.0 if param in solver_params else 0.0
    for g_var in opti_d.get('generator_variables', []):
        d = o.generator_variable.add()
        d.variable_name = g_var
        d.type = 'Constant'
        d.multiplier = 0
    if save_solver_in_proto:
        solver.set_states_to_protobuf(o)
    return o


def _create_monitor(ctx, monitor):
    datasets = ctx.datasets
    if monitor['network'] not in ctx.networks:
        raise ValueError(
            "{} is not found in networks.".format(monitor['network']))
    proto_network = ctx.proto_graphs[monitor['network']].default_graph()
    m = nnabla_pb2.Monitor()
    m.name = monitor['name']
    m.network_name = monitor['network']
    if isinstance(monitor['dataset'], (list, tuple)):
        for dataset_name in monitor['dataset']:
            if dataset_name in datasets:
                m.dataset_name.append(dataset_name)
                dataset = datasets[dataset_name]
            else:
                raise ValueError(
                    "Invalid dataset name is found in monitor definition: {}".format(dataset_name))
    elif isinstance(monitor['dataset'], str):
        dataset_name = monitor['dataset']
        if dataset_name in datasets:
            m.dataset_name.append(dataset_name)
            dataset = datasets[dataset_name]
    if dataset is None:
        raise ValueError("Dataset is not defined in monitor definition.")
    for var_name, data_name in monitor.get('data_variables', {}).items():
        d = m.data_variable.add()
        d.variable_name = var_name
        d.data_name = data_name
    for out in monitor.get('monitor_variables', proto_network.outputs):
        d = m.monitor_variable.add()
        d.type = 'Error'
        d.variable_name = out
    for g_var in monitor.get('generator_variables', []):
        d = m.generator_variable.add()
        d.variable_name = g_var
        d.type = 'Constant'
        d.multiplier = 0
    return m


def _create_executor(ctx, executor):
    '''
    '''
    name, network, remap = \
        executor['name'], ctx.networks[executor['network']], \
        executor.get('remp', {})

    generator_variables = executor.get('generator_variables', [])
    no_image_normalization = executor.get('no_image_normalization')

    proto_network = ctx.proto_graphs[executor['network']].default_graph()

    e = nnabla_pb2.Executor()
    e.name = name
    e.network_name = network.name
    if no_image_normalization is not None:
        e.no_image_normalization = no_image_normalization
    for vname in executor.get('data', proto_network.inputs):
        if vname not in proto_network.variables:
            raise ValueError("{} is not found in networks!".format(vname))
        dv = e.data_variable.add()
        dv.variable_name = vname
        dv.data_name = remap.get(vname, vname)
    for vname in executor.get('output', proto_network.outputs):
        if vname not in proto_network.variables:
            raise ValueError("{} is not found in networks!".format(vname))
        ov = e.output_variable.add()
        ov.variable_name = vname
        ov.data_name = remap.get(vname, vname)
    for param in proto_network.parameters.keys():
        d = e.parameter_variable.add()
        d.variable_name = param
    for vname in generator_variables:
        d = e.generator_variable.add()
        d.type = 'Constant'
        d.multiplier = 0
        d.variable_name = vname
    return e
# ----------------------------------------------------------------------
# Helper functions (END)
# ----------------------------------------------------------------------


def create_proto(contents, include_params=False, variable_batch_size=True, save_solver_in_proto=False):
    class Context:
        pass

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
    ctx = Context()
    ctx.proto_graphs = {}
    networks = {}
    if 'networks' in contents:
        proto_nets = []
        for net in contents['networks']:
            networks[net['name']] = _create_network(
                ctx, net, variable_batch_size)
            proto_nets.append(networks[net['name']])
        proto.network.extend(proto_nets)
    ctx.networks = networks
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
    ctx.datasets = datasets
    if 'optimizers' in contents:
        proto_optimizers = []
        for o in contents['optimizers']:
            proto_optimizers.append(
                _create_optimizer(ctx, o, save_solver_in_proto))
        proto.optimizer.extend(proto_optimizers)
    if 'monitors' in contents:
        proto_monitors = []
        for m in contents['monitors']:
            proto_monitors.append(_create_monitor(ctx, m))
        proto.monitor.extend(proto_monitors)
    if 'executors' in contents:
        proto_executors = []
        for e in contents['executors']:
            proto_executors.append(
                _create_executor(ctx, e))
        proto.executor.extend(proto_executors)

    if include_params:
        params = get_parameters(grad_only=False)
        for variable_name, variable in params.items():
            parameter = proto.parameter.add()
            parameter.variable_name = variable_name
            parameter.shape.dim.extend(variable.shape)
            parameter.data.extend(numpy.array(variable.d).flatten().tolist())
            parameter.need_grad = variable.need_grad

    return proto


def save(filename, contents, include_params=False, variable_batch_size=True, extension=".nnp", parameters=None, include_solver_state=False, solver_state_format='.h5'):
    '''Save network definition, inference/training execution
    configurations etc.

    Args:
        filename (str or file object): Filename to store information. The file
            extension is used to determine the saving file format.
            ``.nnp``: (Recommended) Creating a zip archive with nntxt (network
            definition etc.) and h5 (parameters).
            ``.nntxt``: Protobuf in text format.
            ``.protobuf``: Protobuf in binary format (unsafe in terms of
             backward compatibility).
        contents (dict): Information to store.
        include_params (bool): Includes parameter into single file. This is
            ignored when the extension of filename is nnp.
        variable_batch_size (bool):
            By ``True``, the first dimension of all variables is considered
            as batch size, and left as a placeholder
            (more specifically ``-1``). The placeholder dimension will be
            filled during/after loading.
        extension: if files is file-like object, extension is one of ".nntxt", ".prototxt", ".protobuf", ".h5", ".nnp".
        include_solver_state (bool): Indicate whether to save solver state or not. 
        solver_state_format (str):
            '.h5' or '.protobuf', default '.h5', indicate in which format will solver state be saved,
            notice that this option only works when save network definition in .nnp format
            and include_solver_state is True.

    Example:
        The following example creates a two inputs and two
        outputs MLP, and save the network structure and the initialized
        parameters.

        .. code-block:: python

            import nnabla as nn
            import nnabla.functions as F
            import nnabla.parametric_functions as PF
            from nnabla.utils.save import save

            batch_size = 16
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
    ctx = FileHandlerContext()
    ext = extension
    if isinstance(filename, str):
        _, ext_c = os.path.splitext(filename)
        ext = ext_c if ext_c else ext
    include_params = False if ext == '.nnp' else include_params
    save_solver_in_proto = include_solver_state and ext != '.nnp'
    ctx.proto = create_proto(contents, include_params,
                             variable_batch_size, save_solver_in_proto)
    ctx.parameters = parameters
    if include_solver_state and ext == '.nnp':
        if 'optimizers' not in contents:
            raise KeyError('optimizers should be specified in \
                contents when include_solver_state is True')
        _format_opti_config_for_states_checkpoint(ctx, contents)
        file_savers = get_default_file_savers(
            solver_state_format=solver_state_format)
    else:
        file_savers = get_default_file_savers()
    save_files(ctx, file_savers, filename, ext)
    logger.info("Model file is saved as ({}): {}".format(ext, filename))
