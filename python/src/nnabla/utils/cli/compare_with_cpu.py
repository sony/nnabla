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

import os
from collections import OrderedDict

import nnabla as nn
import nnabla.utils.load as load
import numpy as np
from contextlib2 import ExitStack  # Backport from python3
from nnabla.logger import logger
from nnabla.parameter import get_parameters
from nnabla.utils.cli.utility import let_data_to_variable
from nnabla.utils.progress import configure_progress, progress


def add_result(title, result_dict, result_array):
    result_row = [line[0] for line in result_array]
    col_index = len(result_array[0])
    for k in result_dict.keys():
        if k not in result_row:
            result_row.append(k)
            row = ['' for _ in range(len(result_array[0]))]
            row[0] = k
            result_array.append(row)
    result_array[0].append(title)
    for k, v in result_dict.items():
        row_index = result_row.index(k)
        result_array[row_index].append('%f' % v)

    return result_array


def calc_norm_diff(data1, data2):
    std1 = np.std(data1.flatten())
    std2 = np.std(data2.flatten())
    std = max(std1, std2)
    diff_std = np.std(data1.flatten() - data2.flatten())
    norm_diff = (diff_std / std) if std > 0 else 0
    return norm_diff, std1, std2, diff_std


def compare_optimizer(config, parameters, config_cpu, parameters_cpu, result_array):
    loaded_data = {}
    for opt, opt_cpu in zip(config.optimizers.values(), config_cpu.optimizers.values()):
        o = opt.optimizer
        o_cpu = opt_cpu.optimizer
        opts = [o, o_cpu]

        result_name = "optimizer '%s' with network '%s'" % (
            o.name, o.network.name)
        result_dict = OrderedDict()

        logger.log(99, 'Comparing ' + result_name + ' ...')
        logger.log(
            99, 'process(func, variable), norm_diff, current_context_std, cpu_std, diff_std')
        # Start comparison with same parameters
        for p, p_cpu in zip(parameters.values(), parameters_cpu.values()):
            p_cpu.d = p.d

        # Load dataset
        di = opt.data_iterator
        if di not in loaded_data:
            loaded_data[di] = di.next()
        data = loaded_data[di]

        for v, d in o.dataset_assign.items():
            let_data_to_variable(v.variable_instance, data[
                                 di.variables.index(d)],
                                 data_name=d, variable_name=v.name)
        for v, d in o_cpu.dataset_assign.items():
            let_data_to_variable(v.variable_instance, data[
                                 di.variables.index(d)],
                                 data_name=d, variable_name=v.name)

        # Generate data
        generated = {}
        for v, generator in o.generator_assign.items():
            generated[v.name] = generator(v.shape)
            dest_context = config.global_config.default_context if not o.forward_sequence or v not in o.forward_sequence[
                0].inputs else None
            let_data_to_variable(v.variable_instance,
                                 data=generated[v.name], ctx=dest_context,
                                 variable_name=v.name)
        for v, generator in o_cpu.generator_assign.items():
            dest_context = config.global_config.default_context if not o.forward_sequence or v not in o.forward_sequence[
                0].inputs else None
            let_data_to_variable(v.variable_instance,
                                 data=generated[v.name], ctx=dest_context,
                                 variable_name=v.name)

        last_max_diff = 1e-5

        # Forward
        for func, func_cpu in zip(o.forward_sequence, o_cpu.forward_sequence):
            o.network.forward_function(func)
            o_cpu.network.forward_function(func_cpu)
            large_diff = False
            for v, v_cpu in zip(func.outputs, func_cpu.outputs):
                name = 'forward_function (%s, %s)' % (func.name, v.name)
                if v.variable_instance.d.shape != v_cpu.variable_instance.d.shape:
                    logger.log(99, 'Variable shape is different in %s (current_context=%s, cpu=%s)' % (
                        v.name, str(v.variable_instance.d.shape), str(v_cpu.variable_instance.d.shape)))
                norm_diff, std1, std2, diff_std = calc_norm_diff(
                    v.variable_instance.d, v_cpu.variable_instance.d)
                logger.log(99, '%s, %f, %f, %f, %f' %
                           (name, norm_diff, std1, std2, diff_std))
                result_dict[name] = norm_diff
                if norm_diff > last_max_diff:
                    if norm_diff > last_max_diff * 10:
                        logger.log(99, '  current_context(data)=' +
                                   str(v.variable_instance.d.flatten()))
                        logger.log(99, '  cpu(data)=' +
                                   str(v_cpu.variable_instance.d.flatten()))
                        large_diff = True
                    last_max_diff = norm_diff
            if large_diff:
                logger.log(99, '  x_data:')
                for v, v_cpu in zip(func.inputs, func_cpu.inputs):
                    logger.log(99, '    current_context(%s.d)=%s' %
                               (v.name, str(v.variable_instance.d.flatten())))
                    logger.log(99, '    cpu(%s.d)=%s' % (
                        v_cpu.name, str(v_cpu.variable_instance.d.flatten())))

        # Backward
        o.network.prepare_backward(o.backward_sequence)
        o_cpu.network.prepare_backward(o_cpu.backward_sequence)
        for seq, seq_cpu in zip(o.backward_sequence.sequence, o_cpu.backward_sequence.sequence):
            o.network.backward_function(seq)
            o_cpu.network.backward_function(seq_cpu)
            large_diff = False
            for v, v_cpu in zip(seq.func.inputs, seq_cpu.func.inputs):
                if v.variable_instance.need_grad:
                    name = 'backward_function (%s, %s)' % (
                        seq.func.name, v.name)
                    norm_diff, std1, std2, diff_std = calc_norm_diff(
                        v.variable_instance.g, v_cpu.variable_instance.g)
                    logger.log(99, '%s, %f, %f, %f, %f' %
                               (name, norm_diff, std1, std2, diff_std))
                    result_dict[name] = norm_diff
                    if norm_diff > last_max_diff:
                        if norm_diff > last_max_diff * 10:
                            logger.log(99, '  current_context(diff)=' + str(
                                v.variable_instance) + str(v.variable_instance.g.flatten()))
                            logger.log(99, '  cpu(diff)=' + str(v_cpu.variable_instance) +
                                       str(v_cpu.variable_instance.g.flatten()))
                            large_diff = True
                        last_max_diff = norm_diff
            if large_diff:
                logger.log(99, '  x_data:')
                for v, v_cpu in zip(seq.func.inputs, seq_cpu.func.inputs):
                    logger.log(99, '    current_context(%s.d)=%s' %
                               (v.name, str(v.variable_instance.d.flatten())))
                    logger.log(99, '    cpu(%s.d)=%s' % (
                        v_cpu.name, str(v_cpu.variable_instance.d.flatten())))
                logger.log(99, '  y_diff:')
                for v, v_cpu in zip(seq.func.outputs, seq_cpu.func.outputs):
                    logger.log(99, '    current_context(%s.g)=%s' %
                               (v.name, str(v.variable_instance.g.flatten())))
                    logger.log(99, '    cpu(%s.g)=%s' % (
                        v_cpu.name, str(v_cpu.variable_instance.g.flatten())))

        # Update (weight decay)
        if o.weight_decay > 0:
            o.solver.weight_decay(o.weight_decay)
            o_cpu.solver.weight_decay(o_cpu.weight_decay)

        # Update
        o.solver.update()
        o_cpu.solver.update()
        for i, ((v, lr), (v_cpu, lr_cpu)) in enumerate(zip(o.parameter_learning_rate_multipliers.items(), o_cpu.parameter_learning_rate_multipliers.items())):
            if lr > 0:
                name = 'update (%s, %s)' % (o.solver.name, v.name)
                norm_diff, std1, std2, diff_std = calc_norm_diff(
                    v.variable_instance.d, v_cpu.variable_instance.d)
                logger.log(99, '%s, %f, %f, %f, %f' %
                           (name, norm_diff, std1, std2, diff_std))
                result_dict[name] = norm_diff

        result_array = add_result(result_name, result_dict, result_array)

    return result_array


def compare_with_cpu_command(args):
    configure_progress(os.path.join(args.outdir, 'progress.txt'))

    class TrainConfig:
        pass

    class OptConfig:
        pass

    class MonConfig:
        pass

    # Load config with current context
    files = []
    files.append(args.config)

    with nn.parameter_scope('current'):
        info = load.load(files)
        parameters = get_parameters(grad_only=False)

    config = TrainConfig()
    config.global_config = info.global_config
    config.training_config = info.training_config

    config.optimizers = OrderedDict()
    for name, opt in info.optimizers.items():
        o = OptConfig()
        o.optimizer = opt
        o.data_iterator = None
        config.optimizers[name] = o

    config.monitors = OrderedDict()
    for name, mon in info.monitors.items():
        m = MonConfig()
        m.monitor = mon
        m.data_iterator = None
        config.monitors[name] = m

    # Load config with cpu context
    files = []
    files.append(args.config2)

    with nn.parameter_scope('cpu'):
        info_cpu = load.load(files)
        cpu_parameters = get_parameters(grad_only=False)

    config_cpu = TrainConfig()
    config_cpu.global_config = info_cpu.global_config
    config_cpu.training_config = info_cpu.training_config

    config_cpu.optimizers = OrderedDict()
    for name, opt in info_cpu.optimizers.items():
        o = OptConfig()
        o.optimizer = opt
        o.data_iterator = None
        config_cpu.optimizers[name] = o

    config_cpu.monitors = OrderedDict()
    for name, mon in info_cpu.monitors.items():
        m = MonConfig()
        m.monitor = mon
        m.data_iterator = None
        config_cpu.monitors[name] = m

    result_array = [['1-Correl']]

    # Profile Optimizer
    with ExitStack() as stack:
        for name, o in config.optimizers.items():
            o.data_iterator = stack.enter_context(
                o.optimizer.data_iterator())
        for name, o in config_cpu.optimizers.items():
            o.data_iterator = stack.enter_context(
                o.optimizer.data_iterator())
        result_array = compare_optimizer(
            config, parameters, config_cpu, cpu_parameters, result_array)

    # Write profiling result
    import csv
    with open(args.outdir + os.sep + 'compare_with_cpu.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(result_array)

    logger.log(99, 'Compare with CPU Completed.')
    progress(None)
    return True


def add_compare_with_cpu_command(subparsers):
    # Compare with CPU
    subparser = subparsers.add_parser(
        'compare_with_cpu', help='Compare performance between two nntxt.')
    subparser.add_argument(
        '-c', '--config', help='path to nntxt', required=True)
    subparser.add_argument(
        '-c2', '--config2', help='path to cpu nntxt', required=True)
    subparser.add_argument(
        '-o', '--outdir', help='output directory', required=True)
    subparser.set_defaults(func=compare_with_cpu_command)
