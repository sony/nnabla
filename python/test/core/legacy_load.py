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
Load saved network from nntxt.

'''

from collections import OrderedDict
import google.protobuf.text_format as text_format
import itertools
import numpy
import os
import re
import shutil
import tempfile

from nnabla.initializer import (
    NormalInitializer, UniformInitializer, ConstantInitializer, RangeInitializer,
    calc_normal_std_he_forward, calc_normal_std_he_backward, calc_normal_std_glorot, calc_uniform_lim_glorot)
from nnabla.logger import logger
from nnabla.parameter import get_parameter_or_create

from nnabla.utils import nnabla_pb2
from nnabla.utils.data_iterator import data_iterator_csv_dataset, data_iterator_cache
from nnabla.utils.create_cache import CreateCache
from nnabla.utils.load_function import _create_function_instance
from nnabla.utils.nnp_format import nnp_version
from nnabla.utils.communicator_util import current_communicator, single_or_rankzero
from nnabla.utils.learning_rate_scheduler import (
    PolynomialScheduler, CosineScheduler, ExponentialScheduler, StepScheduler, LinearWarmupScheduler)

from legacy_network import Network
from nnabla.utils.progress import progress
from legacy_get_file_handle import get_file_handle_load, get_buf_type

import nnabla as nn
import nnabla.function as F
import nnabla.solver as S


##########################################################################
# Utilities for handling exceptional function arguments.
#


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
    if numpy.prod(shape) != numpy.prod(inputs[0].shape):
        shape = (batch_size,) + shape
        if numpy.prod(shape) != numpy.prod(inputs[0].shape):
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


##########################################################################
# Private functions.
#


def _create_function(ctx, network, f, variable_index):

    variable_index_name = ''.join(
        ['_' + f.repeat_id[index] + '[' + str(i) + ']' for index, i in enumerate(variable_index)])
    variable_index_low_level_name = ''.join(
        ['_' + f.repeat_id[index] + '[' + str(i) + ']' for index, i in enumerate(variable_index[:-1])])
    function_name = f.name + variable_index_name

    if f.type == "RepeatStart":
        # RepeatStart takes input variable and t-1 variable
        assert(len(f.input) == 2)
        if variable_index[-1] == 0:
            # Input variable if t == 0
            input_variable_names = [f.input[0] if f.input[
                0] in network.variables else f.input[0] + variable_index_low_level_name]
        else:
            # t-1 variable if t > 0
            input_variable_names = [f.input[1] + variable_index_low_level_name +
                                    '_' + f.repeat_param.repeat_id + '[' + str(variable_index[-1] - 1) + ']']

    elif f.type == "RepeatEnd":
        assert(len(f.input) == 1)
        input_variable_names = [f.input[0] + variable_index_name + '_' +
                                f.repeat_param.repeat_id + '[' + str(f.repeat_param.times - 1) + ']']

    elif f.type == "RecurrentInput":
        if variable_index[-1] > 0:
            # Create single split function for single RecurrentInput
            return None, None, None
        function_name = f.name + variable_index_low_level_name
        variable_index_name = variable_index_low_level_name
        input_variable_names = [v_name if v_name in network.variables else v_name +
                                variable_index_low_level_name for v_name in f.input]

    elif f.type == "RecurrentOutput":
        assert(len(f.input) == 1)
        input_variable_names = [f.input[0] + variable_index_name + '_' + f.recurrent_param.repeat_id +
                                '[' + str(v_index) + ']' for v_index in range(f.recurrent_param.length)]

    elif f.type == "Delay":
        assert(len(f.input) == 2)  # Delay takes t-1 variable and initial value
        if variable_index[-1] == 0:
            # Initial value if t == 0
            input_variable_names = [f.input[1] if f.input[
                1] in network.variables else f.input[1] + variable_index_low_level_name]
        else:
            # t-1 variable if t > 0
            input_variable_names = [f.input[0] + variable_index_low_level_name + '_' +
                                    f.recurrent_param.repeat_id + '[' + str(variable_index[-1] - 1) + ']']
    else:
        v_names = []
        for v_name in f.input:
            for index, i in enumerate(variable_index):
                v_name = v_name.replace(
                    '{' + f.repeat_id[index] + '}', '[' + str(i) + ']')
            v_names.append(v_name)
        input_variable_names = [v_name if v_name in network.variables else
                                v_name + variable_index_name if v_name + variable_index_name in network.variables else
                                v_name + variable_index_low_level_name for v_name in v_names]
    inputs = [network.variables[v_name] for v_name in input_variable_names]

    if f.type == "RecurrentInput":
        assert(len(inputs) == 1)
        assert(len(f.output) == 1)
        output_variable_names = [f.output[0] + variable_index_low_level_name + '_' + f.recurrent_param.repeat_id + '[' + str(v_index) + ']'
                                 for v_index in range(inputs[0].shape[f.recurrent_param.axis])]
    else:
        output_variable_names = [v_name + variable_index_name if v_name +
                                 variable_index_name in network.variables else v_name for v_name in f.output]

    outputs = [network.variables[v_name] for v_name in output_variable_names]

    persistent = True
    if f.type == "Reshape":
        shape = resolve_reshape_params(inputs, f, network.batch_size)
        function_instance = F.Reshape(
            ctx, shape=shape, inplace=True)
    elif f.type == "RepeatStart":
        function_instance = F.Identity(ctx)
        persistent = False
    elif f.type == "RepeatEnd":
        function_instance = F.Identity(ctx)
        persistent = False
    elif f.type == "RecurrentOutput":
        function_instance = F.Stack(ctx, axis=f.recurrent_param.axis)
    elif f.type == "RecurrentInput":
        function_instance = F.Split(ctx, axis=f.recurrent_param.axis)
    elif f.type == "Delay":
        function_instance = F.Identity(ctx)
        persistent = False
    elif f.type == "Broadcast":
        shape = resolve_broadcast_params(inputs, f, network.batch_size)
        function_instance = F.Broadcast(ctx, shape)
    else:
        function_instance = _create_function_instance(ctx, f)

    # Prepare link structure
    class Function:
        pass
    function = Function()
    function.name = function_name
    function.function_instance = function_instance
    function.inputs = list(inputs)
    function.outputs = list(outputs)
    function.persistent = persistent

    return function, input_variable_names, output_variable_names


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
                shape[0], numpy.prod(shape[1:])), rng=rng)(shape) * v.initializer.multiplier)
        elif v.initializer.type == 'NormalAffineHeBackward':
            initializer = (lambda shape: NormalInitializer(calc_normal_std_he_backward(
                shape[0], numpy.prod(shape[1:])), rng=rng)(shape) * v.initializer.multiplier)
        elif v.initializer.type == 'NormalAffineGlorot':
            initializer = (lambda shape: NormalInitializer(calc_normal_std_glorot(
                shape[0], numpy.prod(shape[1:])), rng=rng)(shape) * v.initializer.multiplier)
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
                shape[0], numpy.prod(shape[1:])), rng=rng)(shape) * v.initializer.multiplier)
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


def _network(proto, default_context, batch_size, all_variables, rng):
    network = Network()
    network.name = proto.name
    # Read Repeat Info
    network.repeat_info = {}
    for r in proto.repeat_info:
        network.repeat_info[r.id] = r.times

    network.variables = OrderedDict()

    if batch_size is None:
        network.batch_size = proto.batch_size
    else:
        network.batch_size = batch_size

    for v in proto.variable:
        for variable_index in itertools.product(*map(tuple, map(range, [network.repeat_info[id] for id in v.repeat_id]))):
            name = v.name
            for index, i in enumerate(variable_index):
                if ('{' + v.repeat_id[index] + '}' in name):
                    name = name.replace(
                        '{' + v.repeat_id[index] + '}', '[' + str(i) + ']')
                else:
                    name += '_' + v.repeat_id[index] + '[' + str(i) + ']'
            if name in all_variables:
                variable = all_variables[name]
            else:
                shape = tuple(
                    [d if d >= 1 else network.batch_size for d in v.shape.dim])
                variable = _create_variable(v, name, shape, rng)
                all_variables[name] = variable
            network.variables[name] = variable
            logger.debug('{}'.format(
                (name, variable.shape, v.initializer.type if v.initializer.type else '-', v.initializer.multiplier)))

    network.functions = OrderedDict()
    network.function_inputs = OrderedDict()
    network.function_outputs = OrderedDict()
    network.variable_inputs = OrderedDict()
    network.variable_outputs = OrderedDict()
    for f in proto.function:
        ctx = default_context if not f.context.backends else _context(
            f.context)

        for variable_index in itertools.product(*map(tuple, map(range, [network.repeat_info[id] for id in f.repeat_id]))):
            function, input_variable_names, output_variable_names = _create_function(
                ctx, network, f, variable_index)
            if function is not None:
                network.functions[function.name] = function
                for v_name in output_variable_names:
                    network.variable_inputs[
                        network.variables[v_name]] = [function]
                for v_name in input_variable_names:
                    if not network.variables[v_name] in network.variable_outputs:
                        network.variable_outputs[
                            network.variables[v_name]] = []
                    network.variable_outputs[
                        network.variables[v_name]].append(function)

    network.setup(optimize=True)
    return network


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


def _create_optimizer(ctx, o, networks, datasets):
    class Optimizer:
        pass

    optimizer = Optimizer()

    optimizer.comm = current_communicator()
    comm_size = optimizer.comm.size if optimizer.comm else 1
    optimizer.start_iter = (o.start_iter - 1) // comm_size + \
        1 if o.start_iter > 0 else 0
    optimizer.end_iter = (o.end_iter - 1) // comm_size + \
        1 if o.end_iter > 0 else 0
    optimizer.name = o.name
    optimizer.order = o.order
    optimizer.update_interval = o.update_interval if o.update_interval > 0 else 1
    optimizer.network = networks[o.network_name]
    optimizer.data_iterators = OrderedDict()
    for d in o.dataset_name:
        optimizer.data_iterators[d] = datasets[d].data_iterator

    optimizer.dataset_assign = OrderedDict()
    for d in o.data_variable:
        optimizer.dataset_assign[
            optimizer.network.variables[d.variable_name]] = d.data_name

    optimizer.generator_assign = OrderedDict()
    for g in o.generator_variable:
        optimizer.generator_assign[optimizer.network.variables[
            g.variable_name]] = _get_generator(g)

    # for debugging
    optimizer.net_variables = optimizer.network.variables

    optimizer.loss_variables = []
    for l in o.loss_variable:
        optimizer.loss_variables.append(
            optimizer.network.variables[l.variable_name])

    optimizer.parameter_learning_rate_multipliers = OrderedDict()
    for p in o.parameter_variable:
        param_variable_names = _get_matching_variable_names(
            p.variable_name, optimizer.network.variables.keys())
        for v_name in param_variable_names:
            optimizer.parameter_learning_rate_multipliers[
                optimizer.network.variables[v_name]] = p.learning_rate_multiplier

    with nn.context_scope(ctx):
        if o.solver.type == 'Adagrad':
            optimizer.solver = S.Adagrad(
                o.solver.adagrad_param.lr, o.solver.adagrad_param.eps)
            init_lr = o.solver.adagrad_param.lr
        elif o.solver.type == 'Adadelta':
            optimizer.solver = S.Adadelta(
                o.solver.adadelta_param.lr, o.solver.adadelta_param.decay, o.solver.adadelta_param.eps)
            init_lr = o.solver.adadelta_param.lr
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

    parameters = {v.name: v.variable_instance for v,
                  local_lr in optimizer.parameter_learning_rate_multipliers.items() if local_lr > 0.0}
    optimizer.solver.set_parameters(parameters)
    optimizer.parameters = OrderedDict(
        sorted(parameters.items(), key=lambda x: x[0]))

    optimizer.weight_decay = o.solver.weight_decay

    # keep following 2 lines for backward compatibility
    optimizer.lr_decay = o.solver.lr_decay if o.solver.lr_decay > 0.0 else 1.0
    optimizer.lr_decay_interval = o.solver.lr_decay_interval if o.solver.lr_decay_interval > 0 else 1
    optimizer.solver.set_states_from_protobuf(o)

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

    optimizer.forward_sequence = optimizer.network.get_forward_sequence(
        optimizer.loss_variables)
    optimizer.backward_sequence = optimizer.network.get_backward_sequence(
        optimizer.loss_variables, optimizer.parameter_learning_rate_multipliers)

    return optimizer


def _context(proto):
    comm = current_communicator()
    if not proto.backends:
        logger.warn('Old-style context. Updating to new format.')
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
                    logger.warn('Fallback to CPU context.')
                    import nnabla_ext.cpu
                    ctx = nnabla_ext.cpu.context()
            elif 'default' in compute_backends:
                try:
                    import nnabla_ext.cuda
                    ctx = nnabla_ext.cuda.context(device_id=device_id)
                except ImportError:
                    logger.warn('Fallback to CPU context.')
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


def _global_config(proto):
    class GlobalConfig:
        pass
    config = GlobalConfig()
    config.default_context = _context(proto.global_config.default_context)
    nn.set_default_context(config.default_context)

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
                if single_or_rankzero():
                    logger.log(99, 'Creating cache data for "' + uri + '"')

                    try:
                        os.makedirs(cache_dir)
                    except OSError:
                        pass  # python2 does not support exists_ok arg

                    if os.path.exists(uri):
                        cc = CreateCache(uri, rng=rng, shuffle=shuffle)
                        cc.create(cache_dir, normalize=False)
                    else:
                        with data_iterator_csv_dataset(uri, batch_size, shuffle, rng=rng, normalize=False, cache_dir=cache_dir, with_memory_cache=False) as di:
                            pass

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


def _networks(proto, default_context, batch_size, network_names=None):
    # Load networks
    networks = OrderedDict()
    all_variables = {}

    # Random generator for using the same init parameters in all devices
    rng = numpy.random.RandomState(0)

    for np in proto.network:
        if not network_names or np.name in network_names:
            networks[np.name] = _network(
                np, default_context, batch_size, all_variables, rng)

    return networks


def _optimizers(proto, default_context, networks, datasets):
    optimizers = OrderedDict()

    for o in proto.optimizer:
        ctx = default_context if not o.solver.context.backends else _context(
            o.solver.context)
        optimizer = _create_optimizer(ctx, o, networks, datasets)
        optimizers[o.name] = optimizer

    return optimizers


def _monitors(proto, default_context, networks, datasets):
    class Monitor:
        pass
    monitors = OrderedDict()

    for m in proto.monitor:
        monitor = Monitor()

        monitor.network = networks[m.network_name]

        # for debugging
        monitor.net_variables = monitor.network.variables

        monitor.data_iterators = OrderedDict()
        for d in m.dataset_name:
            monitor.data_iterators[d] = datasets[d].data_iterator

        monitor.dataset_assign = OrderedDict()
        for d in m.data_variable:
            monitor.dataset_assign[monitor.network.variables[
                d.variable_name]] = d.data_name

        monitor.generator_assign = OrderedDict()
        for g in m.generator_variable:
            monitor.generator_assign[monitor.network.variables[
                g.variable_name]] = _get_generator(g)

        monitor.monitor_variables = []
        for e in m.monitor_variable:
            monitor.monitor_variables.append(
                monitor.network.variables[e.variable_name])

        monitor.forward_sequence = monitor.network.get_forward_sequence(
            monitor.monitor_variables)

        monitors[m.name] = monitor

    return monitors


def _executors(executors_proto, networks):
    class Executor:
        pass
    executors = OrderedDict()

    for e in executors_proto.executor:
        executor = Executor()

        executor.network = networks[e.network_name]
        executor.num_evaluations = e.num_evaluations if e.num_evaluations > 0 else 1
        executor.repeat_evaluation_type = e.repeat_evaluation_type
        executor.need_back_propagation = e.need_back_propagation
        executor.no_image_normalization = e.no_image_normalization

        executor.dataset_assign = OrderedDict()
        for d in e.data_variable:
            executor.dataset_assign[executor.network.variables[
                d.variable_name]] = d.data_name

        executor.generator_assign = OrderedDict()
        for g in e.generator_variable:
            executor.generator_assign[executor.network.variables[
                g.variable_name]] = _get_generator(g)

        executor.output_assign = OrderedDict()
        for o in e.output_variable:
            executor.output_assign[executor.network.variables[
                o.variable_name]] = [o.type, o.data_name]

        executor.parameters = OrderedDict()
        for p in e.parameter_variable:
            param_variable_names = _get_matching_variable_names(
                p.variable_name, executor.network.variables.keys())
            for v_name in param_variable_names:
                executor.parameters[
                    executor.network.variables[v_name]] = v_name

        executor.forward_sequence = executor.network.get_forward_sequence(
            [o for o in executor.output_assign.keys()])

        if executor.need_back_propagation:
            executor.loss_variables = []
            for l in e.loss_variable:
                executor.loss_variables.append(executor.network.variables[
                    l.variable_name])

            executor.parameter_learning_rate_multipliers = OrderedDict()
            for p in e.parameter_variable:
                param_variable_names = _get_matching_variable_names(
                    p.variable_name, executor.network.variables.keys())
                for v_name in param_variable_names:
                    executor.parameter_learning_rate_multipliers[
                        executor.network.variables[v_name]] = p.learning_rate_multiplier

            executor.backward_sequence = executor.network.get_backward_sequence(
                executor.loss_variables, executor.parameter_learning_rate_multipliers)

        executors[e.name] = executor

    return executors


def _load_optimizer_checkpoint(opti_proto, opti_h5_files, info):
    if opti_proto.optimizer:
        opti_proto_dict = {}
        for o in opti_proto.optimizer:
            opti_proto_dict[o.name] = o
        for o in info.optimizers.values():
            if o.name in opti_proto_dict:
                o.solver.set_states_from_protobuf(opti_proto_dict[o.name])
    elif opti_h5_files:
        for o in info.optimizers.values():
            key = '{}_{}_optimizer.h5.optimizer'.format(
                o.name,
                re.sub(r'(|Cuda)$', '', str(o.solver.name))
            )
            if key in opti_h5_files:
                name_ext = opti_h5_files[key]
                filepath = os.path.splitext(name_ext)[0]
                os.rename(name_ext, filepath)
                o.solver.load_states(filepath)


##########################################################################
# API
#
def load(filenames, prepare_data_iterator=True, batch_size=None, exclude_parameter=False, parameter_only=False, extension=".nntxt"):
    '''load
    Load network information from files.

    Args:
        filenames (list): file-like object or List of filenames.
        extension: if filenames is file-like object, extension is one of ".nntxt", ".prototxt", ".protobuf", ".h5", ".nnp".
    Returns:
        dict: Network information.
    '''
    class Info:
        pass
    info = Info()

    proto = nnabla_pb2.NNablaProtoBuf()

    # optimizer checkpoint
    opti_proto = nnabla_pb2.NNablaProtoBuf()
    OPTI_BUF_EXT = ['.optimizer']
    opti_h5_files = {}
    tmpdir = tempfile.mkdtemp()

    if isinstance(filenames, list) or isinstance(filenames, tuple):
        pass
    elif isinstance(filenames, str) or hasattr(filenames, 'read'):
        filenames = [filenames]

    for filename in filenames:
        if isinstance(filename, str):
            _, ext = os.path.splitext(filename)
        else:
            ext = extension

        # TODO: Here is some known problems.
        #   - Even when protobuf file includes network structure,
        #     it will not loaded.
        #   - Even when prototxt file includes parameter,
        #     it will not loaded.

        if ext in ['.nntxt', '.prototxt']:
            if not parameter_only:
                with get_file_handle_load(filename, ext) as f:
                    try:
                        text_format.Merge(f.read(), proto)
                    except:
                        logger.critical('Failed to read {}.'.format(filename))
                        logger.critical(
                            '2 byte characters may be used for file name or folder name.')
                        raise
            if len(proto.parameter) > 0:
                if not exclude_parameter:
                    nn.load_parameters(filename, extension=ext)
        elif ext in ['.protobuf', '.h5']:
            if not exclude_parameter:
                nn.load_parameters(filename, extension=ext)
            else:
                logger.info('Skip loading parameter.')

        elif ext == '.nnp':
            with get_file_handle_load(filename, ext) as nnp:
                for name in nnp.namelist():
                    _, ext = os.path.splitext(name)
                    if name == 'nnp_version.txt':
                        pass  # TODO currently do nothing with version.
                    elif ext in ['.nntxt', '.prototxt']:
                        if not parameter_only:
                            with nnp.open(name, 'r') as f:
                                text_format.Merge(f.read(), proto)
                        if len(proto.parameter) > 0:
                            if not exclude_parameter:
                                with nnp.open(name, 'r') as f:
                                    nn.load_parameters(f, extension=ext)
                    elif ext in ['.protobuf', '.h5']:
                        if not exclude_parameter:
                            with nnp.open(name, 'r') as f:
                                nn.load_parameters(f, extension=ext)
                        else:
                            logger.info('Skip loading parameter.')
                    elif ext in OPTI_BUF_EXT:
                        buf_type = get_buf_type(name)
                        if buf_type == 'protobuf':
                            with nnp.open(name, 'r') as f:
                                with get_file_handle_load(f, '.protobuf') as opti_p:
                                    opti_proto.MergeFromString(
                                        opti_p.read())
                        elif buf_type == 'h5':
                            nnp.extract(name, tmpdir)
                            opti_h5_files[name] = os.path.join(
                                tmpdir, name)

    default_context = None
    if proto.HasField('global_config'):
        info.global_config = _global_config(proto)
        default_context = info.global_config.default_context
        if 'cuda' in default_context.backend:
            import nnabla_ext.cudnn
        elif 'cuda:float' in default_context.backend:
            try:
                import nnabla_ext.cudnn
            except:
                pass
        try:
            x = nn.Variable()
            y = nn.Variable()
            func = F.ReLU(default_context, inplace=True)
            func.setup([x], [y])
            func.forward([x], [y])
        except:
            logger.warn('Fallback to CPU context.')
            import nnabla_ext.cpu
            default_context = nnabla_ext.cpu.context()
    else:
        import nnabla_ext.cpu
        default_context = nnabla_ext.cpu.context()

    comm = current_communicator()
    if comm:
        default_context.device_id = str(comm.local_rank)
    if proto.HasField('training_config'):
        info.training_config = _training_config(proto)

    info.datasets = _datasets(
        proto, prepare_data_iterator if prepare_data_iterator is not None else info.training_config.max_epoch > 0)

    info.networks = _networks(proto, default_context, batch_size)

    info.optimizers = _optimizers(
        proto, default_context, info.networks, info.datasets)
    _load_optimizer_checkpoint(opti_proto, opti_h5_files, info)
    shutil.rmtree(tmpdir)

    info.monitors = _monitors(
        proto, default_context, info.networks, info.datasets)

    info.executors = _executors(proto, info.networks)

    return info
