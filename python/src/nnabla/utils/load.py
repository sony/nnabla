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
Load saved network from nntxt.

'''

import itertools
import os
import re
from collections import OrderedDict

import nnabla as nn
import nnabla.functions as F
import nnabla.solver as S
import numpy
from nnabla.core.graph_optimizer import IdentityRemover
from nnabla.initializer import (
    NormalInitializer, UniformInitializer, ConstantInitializer, RangeInitializer)
from nnabla.logger import logger
from nnabla.utils import nnabla_pb2
from nnabla.utils.communicator_util import current_communicator, single_or_rankzero
from nnabla.utils.create_cache import CreateCache
from nnabla.utils.data_iterator import data_iterator_csv_dataset, data_iterator_cache
from nnabla.utils.get_file_handle import get_initial_file_loader, load_files
from nnabla.utils.learning_rate_scheduler import (
    PolynomialScheduler, CosineScheduler, ExponentialScheduler, StepScheduler, LinearWarmupScheduler)


# The mainly difference between legacy implementation and refactor-ed implementation:
#  - legacy shares variable dictionary with each networks, while new has not implemented.
#  - legacy parameters are included in variables, refactor-ed is included in parameters
#  - legacy [-1, x, y, z] will be replaced to [b, x, y, z]. But refactor-ed implementation
#     keep proto variable's shape as [-1, x, y, z], only computation graph's variable's
#     shape is replaced as [b, x, y, z]


def _networks(info, proto_graph):
    def is_required(pf):
        if pf.name.startswith('$EC$/'):
            return True
        return False

    # TODO: sharing variable among networks
    info.proto_graph = proto_graph
    proto_graph.expand_loop_control()
    for network in proto_graph.networks.values():
        network.execute_on_proto(IdentityRemover(
            info.renamed_variables, is_required=is_required))
        network(ctx=info.default_context, batch_size=info.batch_size)
    return proto_graph.networks


def _get_generator(proto):
    if proto.type == 'Normal':
        return NormalInitializer(sigma=proto.multiplier)
    elif proto.type == 'Uniform':
        return UniformInitializer(lim=(-proto.multiplier, proto.multiplier))
    elif proto.type == 'Range':
        return RangeInitializer(start=0, step=proto.multiplier)
    elif proto.type == 'Constant':
        return ConstantInitializer(value=proto.multiplier)
    else:
        raise ValueError('Generator type "' +
                         proto.type + '" is not supported.')


def _get_matching_variable_names(variable, variable_names):
    if variable in variable_names:
        return [variable]
    r = re.compile('{[^}]*}')
    key = r.sub(r'\\[[\\d+]\\]', variable, re.U)
    r2 = re.compile(key)
    variable_names = [
        v_name for v_name in variable_names if re.match(r2, v_name)]
    if not variable_names:
        raise ValueError('Variable "' +
                         variable + '" is not found.')
    return variable_names


def _check_context(ctx):
    try:
        x = nn.Variable()
        F.relu(x)
    except:
        logger.warning('Fallback to CPU context.')
        import nnabla_ext.cpu
        ctx = nnabla_ext.cpu.context()
    return ctx


def _create_optimizer(ctx, o, optimizer_states_checkpoint):
    class Optimizer:
        pass

    optimizer = Optimizer()

    optimizer.proto = o
    optimizer.comm = current_communicator()
    comm_size = optimizer.comm.size if optimizer.comm else 1
    optimizer.start_iter = (o.start_iter - 1) // comm_size + \
        1 if o.start_iter > 0 else 0
    optimizer.end_iter = (o.end_iter - 1) // comm_size + \
        1 if o.end_iter > 0 else 0
    optimizer.name = o.name
    optimizer.order = o.order
    optimizer.update_interval = o.update_interval if o.update_interval > 0 else 1
    optimizer.network = None
    if optimizer_states_checkpoint:
        for k, v in optimizer_states_checkpoint.items():
            if optimizer.name == k:
                optimizer.solver_checkpoint = v

    with nn.context_scope(ctx):
        if o.solver.type == 'Adagrad':
            optimizer.solver = S.Adagrad(
                o.solver.adagrad_param.lr, o.solver.adagrad_param.eps)
            init_lr = o.solver.adagrad_param.lr
        elif o.solver.type == 'Adadelta':
            optimizer.solver = S.Adadelta(
                o.solver.adadelta_param.lr, o.solver.adadelta_param.decay, o.solver.adadelta_param.eps)
            init_lr = o.solver.adadelta_param.lr
        elif o.solver.type == 'AdaBelief':
            optimizer.solver = S.AdaBelief(o.solver.adabelief_param.alpha, o.solver.adabelief_param.beta1,
                                           o.solver.adabelief_param.beta2, o.solver.adabelief_param.eps,
                                           o.solver.adabelief_param.wd,
                                           o.solver.adabelief_param.amsgrad,
                                           o.solver.adabelief_param.weight_decouple,
                                           o.solver.adabelief_param.fixed_decay,
                                           o.solver.adabelief_param.rectify)
            init_lr = o.solver.adabelief_param.alpha
        elif o.solver.type == 'Adam':
            optimizer.solver = S.Adam(o.solver.adam_param.alpha, o.solver.adam_param.beta1,
                                      o.solver.adam_param.beta2, o.solver.adam_param.eps)
            init_lr = o.solver.adam_param.alpha
        elif o.solver.type == 'AdamW':
            optimizer.solver = S.AdamW(o.solver.adamw_param.alpha, o.solver.adamw_param.beta1,
                                       o.solver.adamw_param.beta2, o.solver.adamw_param.eps,
                                       o.solver.adamw_param.wd)
            init_lr = o.solver.adamw_param.alpha
        elif o.solver.type == 'Adamax':
            optimizer.solver = S.Adamax(o.solver.adamax_param.alpha, o.solver.adamax_param.beta1,
                                        o.solver.adamax_param.beta2, o.solver.adamax_param.eps)
            init_lr = o.solver.adamax_param.alpha
        elif o.solver.type == 'AdaBound':
            optimizer.solver = S.AdaBound(o.solver.adabound_param.alpha, o.solver.adabound_param.beta1,
                                          o.solver.adabound_param.beta2, o.solver.adabound_param.eps,
                                          o.solver.adabound_param.final_lr, o.solver.adabound_param.gamma)
            init_lr = o.solver.adabound_param.alpha
        elif o.solver.type == 'AMSGRAD':
            optimizer.solver = S.AMSGRAD(o.solver.amsgrad_param.alpha, o.solver.amsgrad_param.beta1,
                                         o.solver.amsgrad_param.beta2, o.solver.amsgrad_param.eps)
            init_lr = o.solver.amsgrad_param.alpha
        elif o.solver.type == 'AMSBound':
            optimizer.solver = S.AMSBound(o.solver.amsbound_param.alpha, o.solver.amsbound_param.beta1,
                                          o.solver.amsbound_param.beta2, o.solver.amsbound_param.eps,
                                          o.solver.amsbound_param.final_lr, o.solver.amsbound_param.gamma)
            init_lr = o.solver.amsbound_param.alpha
        elif o.solver.type == 'Eve':
            p = o.solver.eve_param
            optimizer.solver = S.Eve(
                p.alpha, p.beta1, p.beta2, p.beta3, p.k, p.k2, p.eps)
            init_lr = p.alpha
        elif o.solver.type == 'Lars':
            optimizer.solver = S.Lars(o.solver.lars_param.lr, o.solver.lars_param.momentum,
                                      o.solver.lars_param.coefficient, o.solver.lars_param.eps)
            init_lr = o.solver.lars_param.lr
        elif o.solver.type == 'Momentum':
            optimizer.solver = S.Momentum(
                o.solver.momentum_param.lr, o.solver.momentum_param.momentum)
            init_lr = o.solver.momentum_param.lr
        elif o.solver.type == 'Nesterov':
            optimizer.solver = S.Nesterov(
                o.solver.nesterov_param.lr, o.solver.nesterov_param.momentum)
            init_lr = o.solver.nesterov_param.lr
        elif o.solver.type == 'RMSprop':
            optimizer.solver = S.RMSprop(
                o.solver.rmsprop_param.lr, o.solver.rmsprop_param.decay, o.solver.rmsprop_param.eps)
            init_lr = o.solver.rmsprop_param.lr
        elif o.solver.type == 'RMSpropGraves':
            optimizer.solver = S.RMSpropGraves(
                o.solver.rmsprop_graves_param.lr, o.solver.rmsprop_graves_param.decay,
                o.solver.rmsprop_graves_param.momentum, o.solver.rmsprop_graves_param.eps)
            init_lr = o.solver.rmsprop_graves_param.lr
        elif o.solver.type == 'Sgd' or o.solver.type == 'SGD':
            optimizer.solver = S.Sgd(o.solver.sgd_param.lr)
            init_lr = o.solver.sgd_param.lr
        elif o.solver.type == 'SgdW':
            optimizer.solver = S.SgdW(o.solver.sgdw_param.lr, o.solver.sgdw_param.momentum,
                                      o.solver.sgdw_param.wd)
            init_lr = o.solver.sgdw_param.lr
        else:
            raise ValueError('Solver "' + o.solver.type +
                             '" is not supported.')

    optimizer.weight_decay = o.solver.weight_decay

    # keep following 2 lines for backward compatibility
    optimizer.lr_decay = o.solver.lr_decay if o.solver.lr_decay > 0.0 else 1.0
    optimizer.lr_decay_interval = o.solver.lr_decay_interval if o.solver.lr_decay_interval > 0 else 1

    optimizer.comm = current_communicator()
    comm_size = optimizer.comm.size if optimizer.comm else 1
    optimizer.scheduler = ExponentialScheduler(init_lr, 1.0, 1)

    if o.solver.lr_scheduler_type == 'Polynomial':
        if o.solver.polynomial_scheduler_param.power != 0.0:
            optimizer.scheduler = PolynomialScheduler(
                init_lr, o.solver.polynomial_scheduler_param.max_iter // comm_size, o.solver.polynomial_scheduler_param.power)
    elif o.solver.lr_scheduler_type == 'Cosine':
        optimizer.scheduler = CosineScheduler(
            init_lr, o.solver.cosine_scheduler_param.max_iter // comm_size)
    elif o.solver.lr_scheduler_type == 'Exponential':
        if o.solver.exponential_scheduler_param.gamma != 1.0:
            optimizer.scheduler = ExponentialScheduler(
                init_lr, o.solver.exponential_scheduler_param.gamma, o.solver.exponential_scheduler_param.iter_interval // comm_size if o.solver.exponential_scheduler_param.iter_interval > comm_size else 1)
    elif o.solver.lr_scheduler_type == 'Step':
        if o.solver.step_scheduler_param.gamma != 1.0 and len(o.solver.step_scheduler_param.iter_steps) > 0:
            optimizer.scheduler = StepScheduler(
                init_lr, o.solver.step_scheduler_param.gamma, [step // comm_size for step in o.solver.step_scheduler_param.iter_steps])
    elif o.solver.lr_scheduler_type == 'Custom':
        # ToDo
        raise NotImplementedError()
    elif o.solver.lr_scheduler_type == '':
        if o.solver.lr_decay_interval != 0 or o.solver.lr_decay != 0.0:
            optimizer.scheduler = ExponentialScheduler(
                init_lr, o.solver.lr_decay if o.solver.lr_decay > 0.0 else 1.0, o.solver.lr_decay_interval // comm_size if o.solver.lr_decay_interval > comm_size else 1)
    else:
        raise ValueError('Learning Rate Scheduler "' + o.solver.lr_scheduler_type +
                         '" is not supported.')

    if o.solver.lr_warmup_scheduler_type == 'Linear':
        if o.solver.linear_warmup_scheduler_param.warmup_iter >= comm_size:
            optimizer.scheduler = LinearWarmupScheduler(
                optimizer.scheduler, o.solver.linear_warmup_scheduler_param.warmup_iter // comm_size)

    return optimizer


def _context(proto):
    comm = current_communicator()
    if not proto.backends:
        logger.warning('Old-style context. Updating to new format.')
        # Update from old Context
        backends = [x.strip() for x in proto.backend.split('|')]
        compute_backends = [x.strip()
                            for x in proto.compute_backend.split('|')]
        if 'cuda' in backends:
            device_id = str(proto.device_id)
            if comm:
                device_id = str(comm.local_rank)

            if 'cudnn' in compute_backends:
                try:
                    import nnabla_ext.cudnn
                    ctx = nnabla_ext.cudnn.context(device_id=device_id)
                except ImportError:
                    logger.warning('Fallback to CPU context.')
                    import nnabla_ext.cpu
                    ctx = nnabla_ext.cpu.context()
            elif 'default' in compute_backends:
                try:
                    import nnabla_ext.cuda
                    ctx = nnabla_ext.cuda.context(device_id=device_id)
                except ImportError:
                    logger.warning('Fallback to CPU context.')
                    import nnabla_ext.cpu
                    ctx = nnabla_ext.cpu.context()
            else:
                raise ValueError(
                    'Invalid compute_backend {}'.format(proto.compute_backend))
        elif 'cpu' in backends:
            import nnabla_ext.cpu
            ctx = nnabla_ext.cpu.context()
        else:
            raise ValueError('Invalid context {}'.format(proto))
        ctx.array_class = str(proto.array_class)
        return ctx
    ctx = nn.Context()
    ctx.backend = proto.backends
    ctx.array_class = str(proto.array_class)

    if comm:
        ctx.device_id = str(comm.local_rank)
    else:
        ctx.device_id = str(proto.device_id)

    return ctx


def _global_config(proto, default_context=None):
    class GlobalConfig:
        pass
    config = GlobalConfig()
    if proto is not None:
        config.default_context = _context(proto.global_config.default_context)
        nn.set_default_context(config.default_context)
    else:
        config.default_context = default_context

    return config


def _training_config(proto):
    class TrainingConfig:
        pass
    config = TrainingConfig()
    config.max_epoch = proto.training_config.max_epoch
    config.iter_per_epoch = proto.training_config.iter_per_epoch
    config.save_best = proto.training_config.save_best
    config.monitor_interval = proto.training_config.monitor_interval if proto.training_config.monitor_interval > 0 else 10
    return config


def _create_dataset(
        uri,
        batch_size,
        shuffle, no_image_normalization,
        cache_dir,
        overwrite_cache,
        create_cache_explicitly,
        prepare_data_iterator,
        dataset_index):
    class Dataset:
        pass
    dataset = Dataset()
    dataset.uri = uri
    dataset.cache_dir = cache_dir
    dataset.normalize = not no_image_normalization

    comm = current_communicator()

    # use same random state for each process until slice is called
    # different random state is used for each dataset
    rng = numpy.random.RandomState(dataset_index)
    use_memory_cache = comm.size == 1 if comm else True

    if prepare_data_iterator:
        if cache_dir == '':
            cache_dir = None

        # Disable implicit cache creation when MPI is available.
        if cache_dir and (create_cache_explicitly or comm):
            cache_index = os.path.join(cache_dir, "cache_index.csv")
            if not os.path.exists(cache_index) or overwrite_cache:
                create_cache_flag = False
                if single_or_rankzero():
                    create_cache_flag = True
                    os.makedirs(cache_dir, exist_ok=True)

                if comm:
                    comm.barrier()
                    # Check whether cache directory needs to be created on each node.
                    if comm.local_rank == 0 and comm.rank != 0 and not os.path.exists(cache_dir):
                        os.makedirs(cache_dir, exist_ok=True)
                        create_cache_flag = True

                if create_cache_flag:
                    log_uri = (uri.split("/", 3)
                               [-1] if uri.startswith("s3") else uri)
                    if comm:
                        logger.log(
                            99, f'Creating cache data for "{log_uri}" on node rank:{comm.rank} local_rank:{comm.local_rank}')
                    else:
                        logger.log(99, f'Creating cache data for "{log_uri}"')
                    if os.path.exists(uri):
                        cc = CreateCache(uri, rng=rng, shuffle=shuffle)
                        cc.create(cache_dir, normalize=False)
                    else:
                        with data_iterator_csv_dataset(uri, batch_size, shuffle, rng=rng, normalize=False, cache_dir=cache_dir, with_memory_cache=False) as di:
                            pass
                if comm:
                    comm.barrier()

            rng = numpy.random.RandomState(dataset_index)
            dataset.data_iterator = (lambda: data_iterator_cache(
                cache_dir, batch_size, shuffle, rng=rng, normalize=dataset.normalize, with_memory_cache=use_memory_cache))
        elif not cache_dir or overwrite_cache or not os.path.exists(cache_dir) or len(os.listdir(cache_dir)) == 0:
            if comm:
                logger.critical(
                    'Implicit cache creation does not support with MPI')
                import sys
                sys.exit(-1)
            else:
                if cache_dir:
                    try:
                        os.makedirs(cache_dir)
                    except OSError:
                        pass  # python2 does not support exists_ok arg
                dataset.data_iterator = (lambda: data_iterator_csv_dataset(
                    uri, batch_size, shuffle, rng=rng, normalize=dataset.normalize, cache_dir=cache_dir))
        else:
            dataset.data_iterator = (lambda: data_iterator_cache(
                cache_dir, batch_size, shuffle, rng=rng, normalize=dataset.normalize, with_memory_cache=use_memory_cache))
    else:
        dataset.data_iterator = None
    return dataset


def _datasets(proto, prepare_data_iterator=True):
    datasets = OrderedDict()
    for i, d in enumerate(proto.dataset):
        datasets[d.name] = _create_dataset(
            d.uri,
            d.batch_size,
            d.shuffle,
            d.no_image_normalization,
            d.cache_dir,
            d.overwrite_cache,
            d.create_cache_explicitly,
            prepare_data_iterator,
            i)
    return datasets


def _optimizers(info):
    proto, default_context = info.proto, info.default_context
    if hasattr(info, 'optimizer_states_checkpoint'):
        optimizer_states_checkpoint = info.optimizer_states_checkpoint
    else:
        optimizer_states_checkpoint = None
    optimizers = OrderedDict()

    for o in proto.optimizer:
        ctx = default_context if not o.solver.context.backends else _context(
            o.solver.context)
        optimizer = _create_optimizer(ctx, o, optimizer_states_checkpoint)
        optimizers[o.name] = optimizer

    return optimizers


def _monitors(info):
    proto = info.proto

    class Monitor:
        pass
    monitors = OrderedDict()

    for m in proto.monitor:
        monitor = Monitor()
        monitor.proto = m
        monitor.network = None
        monitors[m.name] = monitor

    return monitors


def _executors(info):
    proto = info.proto

    class Executor:
        pass
    executors = OrderedDict()

    for e in proto.executor:
        executor = Executor()

        executor.proto = e
        executor.network = None
        executor.num_evaluations = e.num_evaluations if e.num_evaluations > 0 else 1
        executor.repeat_evaluation_type = e.repeat_evaluation_type
        executor.need_back_propagation = e.need_back_propagation
        executor.no_image_normalization = e.no_image_normalization

        executors[e.name] = executor

    return executors


##########################################################################
# API
#
def restore_optimizer_state(optimizer):
    optimizer.solver.set_states_from_protobuf(optimizer.proto)
    if hasattr(optimizer, 'solver_checkpoint'):
        ext, handler = optimizer.solver_checkpoint
        if ext == '.protobuf':
            optimizer.solver.set_states_from_protobuf(handler)
        elif ext == '.h5':
            from nnabla.utils.get_file_handle import load_solve_state_from_h5
            optimizer.solver.set_states(
                load_solve_state_from_h5(None, handler))
            handler.seek(0)


def base_load(filenames, prepare_data_iterator=True, batch_size=None, exclude_parameter=False, parameter_only=False, extension=".nntxt", context=None):
    class Info:
        pass
    info = Info()

    info.prepare_data_iterator = prepare_data_iterator
    info.batch_size = batch_size
    info.exclude_parameter = exclude_parameter
    info.parameter_only = parameter_only
    info.proto = nnabla_pb2.NNablaProtoBuf()

    # first stage file loaders
    file_loaders = get_initial_file_loader()

    # using global parameter scope, keep consistency with legacy implementation.
    # To avoid to surprise previous developers, but it is better using
    # stand-alone OrderedDict() instance.
    info.parameter_scope = nn.parameter.get_current_parameter_scope()
    load_files(info, file_loaders, filenames, extension)

    default_context = None
    if context:
        if context == 'cpu':
            import nnabla_ext.cpu
            default_context = nnabla_ext.cpu.context()
        else:
            cs = context.split(':')
            if cs[0] == 'cudnn':
                if len(cs) == 1:
                    devid = 0
                else:
                    devid = int(cs[1])
            import nnabla_ext.cudnn
            default_context = nnabla_ext.cudnn.context(device_id=devid)
        if default_context is None:
            logger.warning('Invalid context [{}]'.format(context))
        elif info.proto.HasField('global_config'):
            info.global_config = _global_config(info.proto)
            info.global_config.default_context = default_context

    if default_context is None:
        if info.proto.HasField('global_config'):
            info.global_config = _global_config(info.proto)
            default_context = info.global_config.default_context
            if 'cuda' in default_context.backend:
                import nnabla_ext.cudnn
            elif 'cuda:float' in default_context.backend:
                try:
                    import nnabla_ext.cudnn
                except:
                    pass
        else:
            import nnabla_ext.cpu
            default_context = nnabla_ext.cpu.context()
            info.global_config = _global_config(
                None, default_context=default_context)

    default_context = _check_context(default_context)
    logger.log(99, 'Using context "{}"'.format(default_context))
    comm = current_communicator()
    if comm:
        default_context.device_id = str(comm.local_rank)
    if info.proto.HasField('training_config'):
        info.training_config = _training_config(info.proto)

    info.default_context = default_context
    info.datasets = _datasets(
        info.proto, prepare_data_iterator if prepare_data_iterator is not None else info.training_config.max_epoch > 0)

    info.renamed_variables = {}
    info.networks = _networks(info, nn.graph_def.ProtoGraph.from_proto(info.proto, param_scope=info.parameter_scope,
                                                                       rng=numpy.random.RandomState(0)))

    info.optimizers = _optimizers(info)
    info.monitors = _monitors(info)
    info.executors = _executors(info)

    return info


def link_components(info):
    networks, datasets, renamed = info.networks, info.datasets, info.renamed_variables

    def _link_optimizer(info, networks, datasets, renamed):
        # optimizer
        for optimizer in info.optimizers.values():
            o = optimizer.proto
            optimizer.network = networks[o.network_name]
            optimizer.data_iterators = OrderedDict()
            for d in o.dataset_name:
                optimizer.data_iterators[d] = datasets[d].data_iterator

            optimizer.dataset_assign = OrderedDict()
            for d in o.data_variable:
                optimizer.dataset_assign[
                    optimizer.network.variables[renamed.get(d.variable_name, d.variable_name)]] = d.data_name

            optimizer.generator_assign = OrderedDict()
            for g in o.generator_variable:
                optimizer.generator_assign[optimizer.network.variables[
                    renamed.get(g.variable_name, g.variable_name)]] = _get_generator(g)

            # for debugging
            # optimizer.net_variables = optimizer.network.variables
            # optimizer.net_variables.update(optimizer.network.parameters)

            optimizer.loss_variables = []
            for l in o.loss_variable:
                optimizer.loss_variables.append(
                    optimizer.network.variables[renamed.get(l.variable_name, l.variable_name)])

            optimizer.parameter_learning_rate_multipliers = OrderedDict()
            for p in o.parameter_variable:
                param_variable_names = _get_matching_variable_names(
                    p.variable_name, list(itertools.chain(optimizer.network.parameters.keys(),
                                                          optimizer.network.variables.keys())))
                for v_name in param_variable_names:
                    if v_name in optimizer.network.parameters:
                        optimizer.parameter_learning_rate_multipliers[
                            optimizer.network.parameters[v_name]] = p.learning_rate_multiplier
                    elif v_name in optimizer.network.variables:
                        optimizer.parameter_learning_rate_multipliers[
                            optimizer.network.variables[v_name]] = p.learning_rate_multiplier

            parameters = {}
            for v, local_lr in optimizer.parameter_learning_rate_multipliers.items():
                if local_lr > 0.0:
                    parameters[v.name] = v.variable_instance
                else:
                    v.variable_instance.need_grad = False
            optimizer.solver.set_parameters(parameters)
            optimizer.parameters = OrderedDict(
                sorted(parameters.items(), key=lambda x: x[0]))

            # restore solver checkpoint if exists
            restore_optimizer_state(optimizer)

            for v in optimizer.loss_variables:
                v.variable_instance.grad.fill(1.0 / v.variable_instance.size)

            if len(optimizer.loss_variables) == 1:
                optimizer.target = optimizer.loss_variables[0].variable_instance
            else:
                optimizer.target = F.sink(
                    *[v.variable_instance for v in optimizer.loss_variables], one_input_grad=False)

    def _link_monitor(info, networks, datasets, renamed):
        # monitor
        for monitor in info.monitors.values():
            m = monitor.proto
            monitor.network = networks[m.network_name]

            # for debugging
            # monitor.net_variables = monitor.network.variables
            # monitor.net_variables.update(monitor.network.parameters)

            monitor.data_iterators = OrderedDict()
            for d in m.dataset_name:
                monitor.data_iterators[d] = datasets[d].data_iterator

            monitor.dataset_assign = OrderedDict()
            for d in m.data_variable:
                monitor.dataset_assign[monitor.network.variables[
                    renamed.get(d.variable_name, d.variable_name)]] = d.data_name

            monitor.generator_assign = OrderedDict()
            for g in m.generator_variable:
                monitor.generator_assign[monitor.network.variables[
                    renamed.get(g.variable_name, g.variable_name)]] = _get_generator(g)

            monitor.monitor_variables = []
            for e in m.monitor_variable:
                monitor.monitor_variables.append(
                    monitor.network.variables[
                        renamed.get(e.variable_name, e.variable_name)])

            monitor.target = F.sink(
                *[v.variable_instance for v in monitor.monitor_variables])

    def _link_executor(info, networks, renamed):
        # executor
        for executor in info.executors.values():
            e = executor.proto
            executor.network = networks[e.network_name]
            executor.dataset_assign = OrderedDict()
            for d in e.data_variable:
                executor.dataset_assign[executor.network.variables[
                    renamed.get(d.variable_name, d.variable_name)]] = d.data_name

            executor.generator_assign = OrderedDict()
            for g in e.generator_variable:
                executor.generator_assign[executor.network.variables[
                    renamed.get(g.variable_name, g.variable_name)]] = _get_generator(g)

            executor.output_assign = OrderedDict()
            for o in e.output_variable:
                executor.output_assign[executor.network.variables[
                    renamed.get(o.variable_name, o.variable_name)]] = [o.type, o.data_name]

            executor.parameters = OrderedDict()
            for p in e.parameter_variable:
                param_variable_names = _get_matching_variable_names(
                    p.variable_name, list(itertools.chain(executor.network.parameters.keys(),
                                                          executor.network.variables.keys())))
                for v_name in param_variable_names:
                    if v_name in executor.network.parameters:
                        executor.parameters[
                            executor.network.parameters[v_name]] = v_name
                    if v_name in executor.network.variables:
                        executor.parameters[
                            executor.network.variables[v_name]] = v_name

            executor.forward_target = F.sink(*[v.variable_instance
                                               for v in executor.output_assign.keys()])

            if executor.need_back_propagation:
                executor.loss_variables = []
                for l in e.loss_variable:
                    executor.loss_variables.append(executor.network.variables[
                        l.variable_name])

                executor.parameter_learning_rate_multipliers = OrderedDict()
                for p in e.parameter_variable:
                    param_variable_names = _get_matching_variable_names(
                        p.variable_name, list(itertools.chain(executor.network.parameters.keys(),
                                                              executor.network.variables.keys())))
                    for v_name in param_variable_names:
                        if v_name in executor.network.parameters:
                            executor.parameter_learning_rate_multipliers[
                                executor.network.parameters[v_name]] = p.learning_rate_multiplier
                        elif v_name in executor.network.variables:
                            executor.parameter_learning_rate_multipliers[
                                executor.network.variables[v_name]] = p.learning_rate_multiplier

                executor.backward_target = F.sink(
                    *[v.variable_instance for v in executor.loss_variables])

    _link_optimizer(info, networks, datasets, renamed)
    _link_monitor(info, networks, datasets, renamed)
    _link_executor(info, networks, renamed)


def load(filenames, prepare_data_iterator=True, batch_size=None, exclude_parameter=False, parameter_only=False, extension=".nntxt", context=None):
    '''load
    Load network information from files.

    Args:
        filenames (list): file-like object or List of filenames.
        extension: if filenames is file-like object, extension is one of ".nntxt", ".prototxt", ".protobuf", ".h5", ".nnp".
    Returns:
        dict: Network information.
    '''
    info = base_load(filenames, prepare_data_iterator, batch_size,
                     exclude_parameter, parameter_only, extension, context)
    link_components(info)

    return info
