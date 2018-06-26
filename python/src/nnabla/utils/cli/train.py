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

from six.moves import range
from collections import OrderedDict
from contextlib2 import ExitStack  # Backport from python3
import numpy as np
import os
import copy

from nnabla.logger import logger
from nnabla import available_contexts
from nnabla.parameter import save_parameters
from nnabla.utils.progress import configure_progress, progress
from nnabla.utils.cli.utility import let_data_to_variable
import nnabla.utils.load as load


def _update(iter, config, cost):
    loaded_datas = {}
    is_first_optimizer = True
    for opt in config.optimizers.values():
        o = opt.optimizer
        # Load dataset
        di = opt.data_iterator
        if o.data_iterator not in loaded_datas:
            loaded_datas[o.data_iterator] = di.next()
        datas = loaded_datas[o.data_iterator]
        for v, d in o.dataset_assign.items():
            dest_context = config.global_config.default_context if not o.forward_sequence or v not in o.forward_sequence[
                0].inputs else None
            let_data_to_variable(v.variable_instance, datas[
                                 di.variables.index(d)], ctx=dest_context)

        # Generate data
        for v, generator in o.generator_assign.items():
            dest_context = config.global_config.default_context if not o.forward_sequence or v not in o.forward_sequence[
                0].inputs else None
            let_data_to_variable(v.variable_instance,
                                 data=generator(v.shape), ctx=dest_context)

        # Monitor loss before forward to prepare input data while processing on
        # GPU
        if cost.variables:
            for l in cost.variables:
                cost.sum_iter += np.mean(l.variable_instance.d)
            if is_first_optimizer:
                is_first_optimizer = False
                progress("Training : cost={0:0.6f}".format(cost.sum_iter),
                         (iter % config.training_config.iter_per_epoch) * 1.0 / config.training_config.iter_per_epoch)
                cost.sum_epoch += cost.sum_iter
                cost.sum_iter = 0.0

        # Forward
        o.network.forward(o.forward_sequence)

        # Backward
        o.network.backward(o.backward_sequence, iter % o.update_interval == 0)

        # Update
        if iter % o.update_interval == o.update_interval - 1:
            if o.weight_decay > 0:
                o.solver.weight_decay(o.weight_decay)
            o.solver.update()

        if o.lr_decay != 1.0 and iter % o.lr_decay_interval == o.lr_decay_interval - 1:
            o.solver.set_learning_rate(o.solver.learning_rate() * o.lr_decay)

        # Reserve monitor loss
        cost.variables = o.loss_variables

    # Monitor loss at the end of iteration
    if iter % config.training_config.iter_per_epoch == config.training_config.iter_per_epoch - 1 and cost.variables:
        for l in cost.variables:
            cost.sum_iter += np.mean(l.variable_instance.d)
        cost.sum_epoch += cost.sum_iter
        cost.variables = None
        cost.sum_iter = 0.0

    return cost


def _evaluate(args, config, monitoring_report, best_error):
    error_str = ''
    valid_error = 0.0
    for name, mon in config.monitors.items():
        m = mon.monitor
        error_sum_monitor = 0.0
        error_count = 0
        di = mon.data_iterator
        dp_epoch = di.epoch
        while dp_epoch == di.epoch:
            # Set data to variable
            datas = di.next()
            for v, d in m.dataset_assign.items():
                dest_context = config.global_config.default_context if not m.forward_sequence or v not in m.forward_sequence[
                    0].inputs else None
                let_data_to_variable(v.variable_instance, datas[
                                     di.variables.index(d)], ctx=dest_context)

            # Generate data
            for v, generator in m.generator_assign.items():
                dest_context = config.global_config.default_context if not m.forward_sequence or v not in m.forward_sequence[
                    0].inputs else None
                let_data_to_variable(v.variable_instance,
                                     data=generator(v.shape), ctx=dest_context)

            # Sum error before forward to prepare input data while processing
            # on GPU
            if error_count > 0:
                for v in m.monitor_variables:
                    error_sum_monitor += np.mean(v.variable_instance.d)
                progress('Evaluating "{0}"'.format(
                    name) + ' : error={0:0.6f}'.format(
                    error_sum_monitor / error_count),
                    di.position * 1.0 / di.size)
            error_count += 1

            # Forward recursive
            m.network.forward(m.forward_sequence)

        # Sum error at the end of dataset
        for v in m.monitor_variables:
            error_sum_monitor += np.mean(v.variable_instance.d)

        error = error_sum_monitor / error_count
        monitoring_report.append('  {}: {}\n'.format(name, error))
        if error_str != '':
            error_str += ', '
        else:
            error_str = ' {'
        error_str += '{}={:.6f}'.format(name, error)
        if name == 'valid_error':
            valid_error = error
    if error_str != '':
        error_str += '}'

    # Save Parameters
    if (not config.training_config.save_best) or \
       (not best_error) or \
       (best_error is not None and valid_error <= best_error):
        best_error = valid_error
        save_parameters(os.path.join(args.outdir, 'parameters.h5'))

    return best_error, error_str


def train(args, config):
    max_iter = config.training_config.max_epoch * \
        config.training_config.iter_per_epoch
    logger.log(99, 'Training epoch 1 of {} begin'.format(
        config.training_config.max_epoch))

    class Cost:
        pass
    cost = Cost()
    cost.sum_epoch = 0.0
    cost.sum_iter = 0.0
    cost.variables = None

    best_error = None

    for iter in range(max_iter):
        cost = _update(iter, config, cost)

        if (iter + 1) % config.training_config.iter_per_epoch == 0:
            # End of epoch
            epoch = iter // config.training_config.iter_per_epoch + 1
            cost_avg_epoch = cost.sum_epoch / config.training_config.iter_per_epoch
            monitoring_report = []

            # Evaluation
            error_str = ''
            if epoch % 10 == 0 or epoch <= 5:
                best_error, error_str = _evaluate(
                    args, config, monitoring_report, best_error)

            # Write to monitoring_report.yml
            f = open(os.path.join(args.outdir, 'monitoring_report.yml'), 'a')
            f.write('{}:\n'.format(epoch - 1))
            f.write('  cost: {}\n'.format(cost_avg_epoch))
            for str in monitoring_report:
                f.write(str)
            f.close()
            cost.sum_epoch = 0

            logger.log(99, 'epoch {} of {} cost={:.6f} {}'.format(
                epoch, config.training_config.max_epoch, cost_avg_epoch, error_str))


def train_command(args):
    configure_progress(os.path.join(args.outdir, 'progress.txt'))
    files = []
    files.append(args.config)
    if args.param:
        files.append(args.param)

    class TrainConfig:
        pass
    config = TrainConfig()
    info = load.load(files)

    logger.log(99, 'Train with contexts {}'.format(available_contexts))

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

    # Training
    max_iter = config.training_config.max_epoch * \
        config.training_config.iter_per_epoch
    if max_iter > 0:

        data_iterators = {'optimizer': {}, 'monitor': {}}
        with ExitStack() as stack:
            for name, o in config.optimizers.items():
                o.data_iterator = stack.enter_context(
                    o.optimizer.data_iterator())
            for name, m in config.monitors.items():
                m.data_iterator = stack.enter_context(
                    m.monitor.data_iterator())
            train(args, config)

    else:
        # save parameters without training (0 epoch learning)
        save_parameters(os.path.join(
            args.outdir, 'parameters.h5'))

    logger.log(99, 'Training Completed.')
    progress(None)


def add_train_command(subparsers):
    # Train
    subparser = subparsers.add_parser('train', help='Training with NNP.')
    subparser.add_argument(
        '-c', '--config', help='path to nntxt', required=True)
    subparser.add_argument(
        '-p', '--param', help='path to parameter file', required=False)
    subparser.add_argument(
        '-o', '--outdir', help='output directory', required=True)
    subparser.set_defaults(func=train_command)
