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

from __future__ import print_function

import os
import time
from collections import OrderedDict
from functools import partial

import nnabla.utils.callback as callback
import nnabla.utils.load as load
import numpy as np
from contextlib2 import ExitStack  # Backport from python3
from nnabla.ext_utils import import_extension_module
from nnabla.logger import logger
from nnabla.utils.cli.utility import let_data_to_variable
from nnabla.utils.progress import configure_progress, progress


def profile(config, name, func, result_dict, synchromize):
    # Warm-up
    func()
    synchromize()

    # Profile
    start_0 = time.time()
    result = 0
    count = 0
    while time.time() < start_0 + 1.0 or count < 100:
        start = time.time()
        func()
        synchromize()
        stop = time.time()
        result += stop - start
        count += 1

    t = result * 1000 / count

    logger.log(99, '%s %f(ms)' % (name, t))
    result_dict[name] = t
    return result_dict


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


def profile_optimizer(config, result_array, synchronize):
    # Profile Training
    for opt in config.optimizers.values():
        o = opt.optimizer
        result_name = "optimizer '%s' with network '%s'" % (
            o.name, o.network.name)
        result_dict = OrderedDict()

        logger.log(99, 'Profiling ' + result_name + ' ...')
        # Clear weight
        for name, p in o.parameters.items():
            if name[-2:] in ('/W', '/b'):
                p.data.zero()

        # Load dataset
        def load_dataset():
            loaded_data = {}
            data = OrderedDict()
            for di in opt.data_iterators:
                if di not in loaded_data:
                    loaded_data[di] = di.next()
                data.update(zip(di.variables, loaded_data[di]))
            return data
        profile(config, 'load_dataset', load_dataset, result_dict, synchronize)

        # Let data
        data = load_dataset()
        for v, d in o.dataset_assign.items():
            def let_data():
                if d not in data:
                    raise ValueError(
                        'Data "' + d + '" is not found in dataset.')
                let_data_to_variable(v.variable_instance, data=data[d],
                                     data_name=d, variable_name=v.name)
            profile(config, 'let_data (%s to %s)' %
                    (d, v.name), let_data, result_dict, synchronize)

        # Generate data
        for v, generator in o.generator_assign.items():
            def generate_data():
                let_data_to_variable(v.variable_instance,
                                     data=generator(
                                         v.variable_instance.d.shape),
                                     variable_name=v.name)
            profile(config, 'generate_data (%s)' %
                    v.name, generate_data, result_dict, synchronize)

        '''
        # Setup (detail)
        for func in o.forward_sequence:
            def setup():
                o.network.setup_function(func)
            profile(config, 'setup_function (%s : %s)' % (
                func.name, func.function_instance.name), setup, result_dict, synchronize)
        '''
        # Warm-up
        # o.network.forward(o.forward_sequence)
        # o.network.prepare_backward(o.backward_sequence)
        # o.network.backward(o.backward_sequence)
        o.forward_sequence = []
        o.backward_sequence = []
        o.target.forward(clear_no_need_grad=True,
                         function_pre_hook=lambda f: o.forward_sequence.append(f))
        o.target.backward(
            clear_buffer=True, function_pre_hook=lambda f: o.backward_sequence.append(f))

        # Forward (detail)
        for func in o.forward_sequence:
            if func.name == 'Sink':
                continue
            profile(config, 'forward_function (%s : %s)' % (func.name, func.name),
                    partial(func.forward, inputs=func.inputs,
                            outputs=func.outputs),
                    result_dict, synchronize)

        # Backward (detail)
        def empty_func():
            pass   # keep this for backward compatible
        profile(config, 'prepare_backward',
                empty_func, result_dict, synchronize)

        for func in o.backward_sequence:
            if func.name == 'Sink':
                continue
            profile(config, 'backward_function (%s : %s)' % (func.name, func.name),
                    partial(func.backward, inputs=func.inputs,
                            outputs=func.outputs),
                    result_dict, synchronize)

        # Forward (all)
        def forward_all():
            o.target.forward(clear_no_need_grad=True)
        profile(config, 'forward_all', forward_all, result_dict, synchronize)

        # Backward (all)
        def backward_all():
            o.target.backward(clear_buffer=True)
        profile(config, 'backward_all', backward_all, result_dict, synchronize)

        # Backward (wo param zero_grad)
        # Backward (all)
        def backward_all_wo_zero_grad():
            for name, p in o.parameters.items():
                if name[-2:] in ('/W', '/b'):
                    p.grad.zero()
            o.target.backward(clear_buffer=True)
        profile(config, 'backward_all(wo param zero_grad)',
                backward_all_wo_zero_grad, result_dict, synchronize)

        # Update (weight decay)
        if o.weight_decay > 0:
            def weight_decay():
                o.solver.weight_decay(o.weight_decay)
            profile(config, 'weight_decay (%s)' %
                    o.solver.name, weight_decay, result_dict, synchronize)

        # Update
        def update():
            o.solver.update()
        profile(config, 'update (%s)' %
                o.solver.name, update, result_dict, synchronize)

        # Monitor loss
        def monitor_loss():
            for l in o.loss_variables:
                np.mean(l.variable_instance.d)
        profile(config, 'monitor_loss', monitor_loss, result_dict, synchronize)

        result_array = add_result(result_name, result_dict, result_array)

    return result_array


def profile_command(args):
    callback.update_status(args)

    configure_progress(os.path.join(args.outdir, 'progress.txt'))

    class TrainConfig:
        pass
    config = TrainConfig()
    info = load.load(args.config)

    config.global_config = info.global_config
    config.training_config = info.training_config

    class OptConfig:
        pass
    config.optimizers = OrderedDict()
    for name, opt in info.optimizers.items():
        o = OptConfig()
        o.optimizer = opt
        o.data_iterators = []
        config.optimizers[name] = o

    class MonConfig:
        pass
    config.monitors = OrderedDict()
    for name, mon in info.monitors.items():
        m = MonConfig()
        m.monitor = mon
        m.data_iterators = []
        config.monitors[name] = m

    try:
        ext_module = import_extension_module(
            config.global_config.default_context.backend[0].split(':')[0])

        def synchronize(): return ext_module.synchronize(
            device_id=config.global_config.default_context.device_id)

    except (ImportError, ModuleNotFoundError):
        def synchronize(): return None

    result_array = [['time in ms']]

    callback.update_status('processing', True)

    # Profile Optimizer
    with ExitStack() as stack:
        # Create data_iterator instance only once for each dataset in optimizers
        optimizer_data_iterators = {}
        for name, o in config.optimizers.items():
            for di in o.optimizer.data_iterators.values():
                if di not in optimizer_data_iterators:
                    di_instance = stack.enter_context(di())
                    optimizer_data_iterators[di] = di_instance
                else:
                    di_instance = optimizer_data_iterators[di]
                o.data_iterators.append(di_instance)
        result_array = profile_optimizer(config, result_array, synchronize)

    # Write profiling result
    import csv
    with open(args.outdir + os.sep + 'profile.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(result_array)

    logger.log(99, 'Profile Completed.')
    progress(None)
    callback.update_status('finished')
    return True


def add_profile_command(subparsers):
    # Profile
    subparser = subparsers.add_parser(
        'profile', help='Profiling performance with NNP.')
    subparser.add_argument(
        '-c', '--config', help='path to nntxt', required=True)
    subparser.add_argument(
        '-o', '--outdir', help='output directory', required=True)
    subparser.set_defaults(func=profile_command)
