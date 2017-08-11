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


def _add_variable(var, variables, params, prefix):
    if var not in variables.values():
        vname = prefix
        if var in params:
            vname = params[var]
        count = 2
        vname_base = vname
        while vname in variables:
            vname = '{}_{}'.format(vname_base, count)
            count += 1
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
    g.default_context.backend = ctx.backend
    g.default_context.array_class = ctx.array_class
    g.default_context.device_id = ctx.device_id
    g.default_context.compute_backend = ctx.compute_backend
    return g


def _create_training_config(max_epoch, iter_per_epoch, save_best):
    t = nnabla_pb2.TrainingConfig()
    t.max_epoch = max_epoch
    t.iter_per_epoch = iter_per_epoch
    t.save_best = save_best
    return t


def _create_dataset(name, uri, cache_dir, variables, shuffle, batch_size):
    d = nnabla_pb2.Dataset()
    d.name = name
    d.uri = uri
    if cache_dir is not None:
        d.cache_dir = cache_dir
    d.shuffle = shuffle
    d.batch_size = batch_size
    d.variable.extend(variables)
    return d


def _create_network(net):
    n = nnabla_pb2.Network()
    n.name = net['name']
    n.batch_size = net['batch_size']

    params = {v: k for k, v in get_parameters(grad_only=False).items()}
    variables = OrderedDict()
    functions = OrderedDict()

    def _network_recursive(func, seen):
        if func is None:
            return
        seen.add(func)
        for i in func.inputs:
            if i.parent in seen:
                continue
            _network_recursive(i.parent, seen)

        # Collect information.
        function_type = func.info.type_name
        function_name = function_name_base = function_type
        count = 2
        while function_name in functions:
            function_name = '{}_{}'.format(function_name_base, count)
            count += 1
        functions[function_name] = {
            'type': function_type,
            'args': func.info.args,
            'inputs': [],
            'outputs': []
        }
        for i in func.inputs:
            vname = _add_variable(i, variables, params,
                                  '{}_Input'.format(function_name))
            functions[function_name]['inputs'].append(vname)
        for o in func.outputs:
            vname = _add_variable(o, variables, params,
                                  '{}_Output'.format(function_name))
            functions[function_name]['outputs'].append(vname)

    seen = set()
    _network_recursive(net['variable'].parent, seen)

    for name, variable in variables.items():
        v = n.variable.add()
        v.name = name
        shape = list(numpy.array(variable.d).shape)
        if variable in params:
            v.type = 'Parameter'
        else:
            v.type = 'Buffer'
            if len(shape) > 0:
                shape[0] = -1
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


def _create_optimizer(name, solver, network, dataset):
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


def _create_monitor(name, monitor, network, dataset):
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


def _create_executor(name, network, input_names):
    e = nnabla_pb2.Executor()
    e.name = name
    e.network_name = network.name
    inputs, outputs, params = _get_net_variables(network)
    count = 0
    for inp in inputs:
        d = e.data_variable.add()
        d.variable_name = inp
        d.data_name = input_names[count]
        count += 1
    for out in outputs:
        d = e.output_variable.add()
        d.variable_name = out
        d.data_name = input_names[count]
        count += 1
    for param in params:
        d = e.parameter_variable.add()
        d.variable_name = param

    return e


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
                                                      datasets[o['dataset']]))
        proto.optimizer.extend(proto_optimizers)
    if 'monitors' in contents:
        proto_monitors = []
        for m in contents['monitors']:
            proto_monitors.append(_create_monitor(m['name'], m['monitor'],
                                                  networks[m['network']],
                                                  datasets[m['dataset']]))
        proto.monitor.extend(proto_monitors)
    if 'executors' in contents:
        proto_executors = []
        for e in contents['executors']:
            proto_executors.append(
                _create_executor(e['name'], networks[e['network']],
                                 e['variables']))
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
    '''save

    Save network information into protocol buffer file.

    This function store information in 'contents' arg into filename.

    Filename

    If extension of the filename is '.nntxt' contents store with
    readable text format, or if extension of the filename is '.protobuf',
    contents store with binary-encoded text.

    Format of contents.

    Root of contents

    ================ ==================
    Key              Type
    ================ ==================
    global_config    dict
    training_config  dict
    networks         list of networks
    datasets         list of datasets
    optimizers       list of optimizes
    monitors         list of monitors
    executors        list of executors
    ================ ==================


    global_config

    ================ ================== =================================
    Key              Type               Description
    ================ ================== =================================
    default_context  Context            Instance of nnabla.Context
    ================ ================== =================================

    training_config

    ================ ================== =================================
    Key              Type               Description
    ================ ================== =================================
    max_epoch        int                Training limit.
    iter_per_epoch   int                Number of iteration in epoch.
    save_best        bool               Save parameter if result is best.
    ================ ================== =================================

    network

    ================ ================== =================================
    Key              Type               Description
    ================ ================== =================================
    name             str                Name of the network
    batch_size       int                Batch size
    variable         Variable           Output instance of nnabla.variable.
    ================ ================== =================================

    dataset

    ================ ================== =================================
    Key              Type               Description
    ================ ================== =================================
    name             str                Name of the dataset
    uri              str                Data location.
    cache_dir        str                Optional: Cache file location.
    variables        tuple of str       Variable names in this dataset.
    shuffle          bool               Is shuffled or not.
    batch_size       int                Batch size
    ================ ================== =================================

    optimizer

    ================ ================== =================================
    Key              Type               Description
    ================ ================== =================================
    name             str                Name of the optimizer
    solver           Solver             Instance of nnabla.Solver
    network          str                Name of network to optimize.
    dataset          str                Name of dataset to use.
    ================ ================== =================================

    monitor

    ================ ================== =================================
    Key              Type               Description
    ================ ================== =================================
    name             str                Name of the monitor.
    monitor          Monitor            Instance of nnabla.Monitor
    network          str                Name of network to monitor.
    dataset          str                Name of dataset to use.
    ================ ================== =================================

    executor

    ================ ================== =================================
    Key              Type               Description
    ================ ================== =================================
    name             str                Name of the executor.
    network          str                Name of network to execute.
    variables        tuple of str       Input variable names.
    ================ ================== =================================

    Args:
        filename (str): Filename to store infomation.
        contents (dict): Information to store.
        include_params (bool): Includes parameter into single file.
'''

    _, ext = os.path.splitext(filename)
    print(filename, ext)
    if ext == '.nntxt' or ext == '.prototxt':
        logger.info("Saveing {} as prototxt".format(filename))
        proto = create_proto(contents, include_params)
        with open(filename, 'w') as file:
            text_format.PrintMessage(proto, file)
    elif ext == '.protobuf':
        logger.info("Saveing {} as protobuf".format(filename))
        proto = create_proto(contents, include_params)
        with open(filename, 'wb') as file:
            file.write(proto.SerializeToString())
    elif ext == '.nnp':
        logger.info("Saveing {} as nnp".format(filename))
        tmpdir = tempfile.mkdtemp()
        save('{}/network.nntxt'.format(tmpdir), contents, include_params=False)
        save_parameters('{}/parameter.h5'.format(tmpdir))
        with zipfile.ZipFile(filename, 'w') as nnp:
            nnp.write('{}/network.nntxt'.format(tmpdir), 'network.nntxt')
            nnp.write('{}/parameter.h5'.format(tmpdir), 'parameter.h5')
        shutil.rmtree(tmpdir)
