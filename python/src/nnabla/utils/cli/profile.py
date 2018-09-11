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
from contextlib2 import ExitStack  # Backport from python3
import numpy as np
import os
import copy
import time

import nnabla as nn
import nnabla.function as F
from nnabla.logger import logger
from nnabla.parameter import save_parameters
from nnabla.utils.progress import configure_progress, progress
from nnabla.utils.cli.utility import let_data_to_variable
import nnabla.utils.load as load
# Console only start
import nnabla.utils.console.status as status
# Console only end


def profile(config, name, func, result_dict):
    # for sync CPU/GPU
    identity = F.Identity(config.global_config.default_context)
    tmp_in = nn.Variable((1,))
    tmp_out = nn.Variable((1,))
    identity.setup([tmp_in], [tmp_out])

    tmp_in.d = [0.]
    identity.forward([tmp_in], [tmp_out])

    # Profile
    start = time.time()
    count = 0
    while time.time() < start + 1.0 or count < 100:
        func()
        count += 1

    # sync CPU/GPU
    identity.forward([tmp_in], [tmp_out])
    data = tmp_out.d

    t = (time.time() - start) * 1000 / count
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


def profile_optimizer(config, result_array):
    # Profile Training
    for opt in config.optimizers.values():
        o = opt.optimizer
        result_name = "optimizer '%s' with network '%s'" % (
            o.name, o.network.name)
        result_dict = OrderedDict()

        logger.log(99, 'Profiling ' + result_name + ' ...')

        # Load dataset
        def load_dataset():
            loaded_datas = {}
            di = opt.data_iterator
            loaded_datas[di] = di.next()
            return loaded_datas
        profile(config, 'load_dataset', load_dataset, result_dict)

        # Let data
        loaded_datas = load_dataset()
        for v, d in o.dataset_assign.items():
            def let_data():
                try:
                    data = loaded_datas[opt.data_iterator][
                        opt.data_iterator.variables.index(d)]
                except:
                    print(opt.data_iterator.variables)
                    raise ValueError(
                        'Data "' + d + '" is not found in dataset.')
                let_data_to_variable(v.variable_instance, data=data)
            profile(config, 'let_data (%s to %s)' %
                    (d, v.name), let_data, result_dict)

        # Generate data
        for v, generator in o.generator_assign.items():
            def generate_data():
                let_data_to_variable(v.variable_instance,
                                     data=generator(v.shape))
            profile(config, 'generate_data (%s)' %
                    v.name, generate_data, result_dict)

        # Setup (detail)
        # CSXENA-5470 disable profiling setup method.
        # for func in o.forward_sequence:
        # def setup():
        # o.network.setup_function(func)
        # profile(config, 'setup_function (%s : %s)' % (
        # func.name, func.function_instance.name), setup, result_dict)

        # Forward (detail)
        for func in o.forward_sequence:
            def forward():
                o.network.forward_function(func)
            in_place_str = ' : in_place' if func.function_instance.inplace_data(
                0) > 0 else ''
            profile(config, 'forward_function (%s : %s%s)' % (
                func.name, func.function_instance.name, in_place_str), forward, result_dict)

        # Backward (detail)
        def prepare_backward():
            o.network.prepare_backward(o.backward_sequence)
        profile(config, 'prepare_backward', prepare_backward, result_dict)
        for seq in o.backward_sequence.sequence:
            o.network.prepare_backward(o.backward_sequence)

            def backward():
                o.network.backward_function(seq)
            in_place_str = ' : in_place' if seq.func.function_instance.inplace_grad(
                0) > 0 else ''
            profile(config, 'backward_function (%s : %s%s)' % (
                seq.func.name, seq.func.function_instance.name, in_place_str), backward, result_dict)

        # Forward (all)
        def forward_all():
            o.network.forward(o.forward_sequence)
        profile(config, 'forward_all', forward_all, result_dict)

        # Backward (all)
        def backward_all():
            o.network.backward(o.backward_sequence)
        profile(config, 'backward_all', backward_all, result_dict)

        # Backward (all)
        def backward_all_wo_zero_grad():
            o.network.backward(o.backward_sequence, parameter_zero_grad=False)
        profile(config, 'backward_all(wo param zero_grad)',
                backward_all_wo_zero_grad, result_dict)

        # Update (weight decay)
        if o.weight_decay > 0:
            def weight_decay():
                o.solver.weight_decay(o.weight_decay)
            profile(config, 'weight_decay (%s)' %
                    o.solver.name, weight_decay, result_dict)

        # Update
        def update():
            o.solver.update()
        profile(config, 'update (%s)' % o.solver.name, update, result_dict)

        # Monitor loss
        def monitor_loss():
            for l in o.loss_variables:
                np.mean(l.variable_instance.d)
        profile(config, 'monitor_loss', monitor_loss, result_dict)

        result_array = add_result(result_name, result_dict, result_array)

    return result_array


def profile_command(args):
    # Console only start
    status.init(args)
    # Console only end

    configure_progress(os.path.join(args.outdir, 'progress.txt'))
    files = []
    files.append(args.config)

    class TrainConfig:
        pass
    config = TrainConfig()
    info = load.load(files)

    config.global_config = info.global_config
    config.training_config = info.training_config

    class OptConfig:
        pass
    config.optimizers = OrderedDict()
    for name, opt in info.optimizers.items():
        o = OptConfig()
        o.optimizer = opt
        o.data_iterator = None
        config.optimizers[name] = o

    class MonConfig:
        pass
    config.monitors = OrderedDict()
    for name, mon in info.monitors.items():
        m = MonConfig()
        m.monitor = mon
        m.data_iterator = None
        config.monitors[name] = m

    result_array = [['time in ms']]

    # Console only start
    status.start_process()
    status.dump(status='processing')
    # Console only end

    # Profile Optimizer
    with ExitStack() as stack:
        for name, o in config.optimizers.items():
            o.data_iterator = stack.enter_context(
                o.optimizer.data_iterator())
        result_array = profile_optimizer(config, result_array)

    # Write profiling result
    import csv
    with open(args.outdir + os.sep + 'profile.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(result_array)

    logger.log(99, 'Profile Completed.')
    progress(None)
    # Console only start
    status.dump(status='finished')
    # Console only end


def add_profile_command(subparsers):
    # Profile
    subparser = subparsers.add_parser(
        'profile', help='Profiling performance with NNP.')
    subparser.add_argument(
        '-c', '--config', help='path to nntxt', required=True)
    subparser.add_argument(
        '-o', '--outdir', help='output directory', required=True)
    subparser.set_defaults(func=profile_command)
