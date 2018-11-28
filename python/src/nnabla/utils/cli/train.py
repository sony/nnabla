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
import glob
import os
import time
import zipfile

import nnabla as nn
from nnabla.logger import logger
from nnabla import available_contexts
from nnabla.parameter import save_parameters
from nnabla.utils.progress import configure_progress, progress
from nnabla.utils.cli.utility import let_data_to_variable
from nnabla.utils.nnp_format import nnp_version
from nnabla.utils.communicator_util import current_communicator, single_or_rankzero

import nnabla.utils.load as load

# Console only start
import nnabla.utils.console.status as status
from nnabla.utils.console.utils import get_info_from_sdcproj
# Console only end


_save_parameter_info = {}


def _all_reduce(comm, var, division, inplace):
    import threading
    _finish = False

    def _wait():
        import time
        import sys
        count = 0
        while not _finish:
            if count > 10000:
                logger.log(99, "STALLED MPI RANK {}".format(comm.rank))
                sys.exit(-1)
            time.sleep(0.01)
            count += 1

    th = threading.Thread(target=_wait)
    th.start()

    comm.all_reduce(var, division=division, inplace=inplace)
    _finish = True
    th.join()


def _save_parameters(args, suffix, epoch, force=False):
    global _save_parameter_info

    if suffix not in _save_parameter_info:
        _save_parameter_info[suffix] = {}
        _save_parameter_info[suffix]['epoch'] = 0
        _save_parameter_info[suffix]['time'] = 0

    current_time = time.time()
    timediff = current_time - _save_parameter_info[suffix]['time']
    epochdiff = epoch - _save_parameter_info[suffix]['epoch']

    globname = os.path.join(args.outdir, 'results_{}_*.nnp'.format(suffix))
    exists = glob.glob(globname)

    base = os.path.join(args.outdir, 'results_{}_{}'.format(suffix, epoch))
    filename = base + '.nnp'

    if not os.path.exists(filename) and \
       (force or timediff > 180.0 or epochdiff > 10):

        version_filename = base + '_version.txt'

        with open(version_filename, 'w') as file:
            file.write('{}\n'.format(nnp_version()))

        param_filename = base + '_param.protobuf'
        save_parameters(param_filename)

        with zipfile.ZipFile(filename, 'w') as nnp:
            nnp.write(version_filename, 'nnp_version.txt')
            nnp.write(_save_parameter_info['config'], os.path.basename(
                _save_parameter_info['config']))
            nnp.write(param_filename, 'parameter.protobuf')

        os.unlink(version_filename)
        os.unlink(param_filename)

        for exist in exists:
            os.unlink(exist)

        _save_parameter_info[suffix]['epoch'] = epoch
        _save_parameter_info[suffix]['time'] = current_time

        # Console only start
        status.save_train_snapshot()
        # Console only end


def _update(iter, config, cost):
    comm = current_communicator()

    loaded_datas = {}
    is_first_optimizer = True

    def _sum_cost():
        if comm:
            # logger.log(99, "Calc cost with communicator")
            var = [nn.NdArray()]
            var[0].data = cost.sum_iteration
            _all_reduce(comm, var, division=False, inplace=True)
            cost.sum_epoch += var[0].data
            cost.num_iteration += comm.size
        else:
            cost.sum_epoch += cost.sum_iteration
            cost.num_iteration += 1

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
                cost.sum_iteration += np.mean(l.variable_instance.d)
                l.variable_instance.data.zero()
            if is_first_optimizer:
                is_first_optimizer = False
                _sum_cost()
                if single_or_rankzero():
                    progress("Training : cost={0:0.6f}".format(cost.sum_iteration),
                             (iter % config.training_config.iter_per_epoch) * 1.0 / config.training_config.iter_per_epoch)
                cost.sum_iteration = 0.0

        # Forward
        o.network.forward(o.forward_sequence)

        # Backward
        o.network.backward(o.backward_sequence, iter % o.update_interval == 0)

        # Update
        if iter % o.update_interval == o.update_interval - 1:
            if o.weight_decay > 0:
                o.solver.weight_decay(o.weight_decay)

            if o.comm:  # Updated param with communicator
                params = [x.grad for x in o.parameters.values()]
                _all_reduce(o.comm, params, division=True, inplace=True)

            if o.scheduler is not None:
                o.solver.set_learning_rate(o.scheduler.get_learning_rate(iter))
            o.solver.update()

        # Sync w sometimes
        if iter % 10 == 9:  # TODO: change the interval
            if o.comm:
                params = [x.data for x in o.parameters.values()]
                _all_reduce(o.comm, params, division=True, inplace=True)

        # Reserve monitor loss
        cost.variables = o.loss_variables

    # Monitor loss at the end of iteration
    if iter % config.training_config.iter_per_epoch == config.training_config.iter_per_epoch - 1 and cost.variables:
        for l in cost.variables:
            cost.sum_iteration += np.mean(l.variable_instance.d)
            l.variable_instance.data.zero()
        _sum_cost()
        cost.variables = None
        cost.sum_iteration = 0.0

    return cost


def _evaluate(args, config, monitoring_report, best_error, epoch):
    comm = current_communicator()
    error_str = ''
    valid_error = 0.0

    def _sum_error(sum, error):
        ret = None
        if comm:
            # logger.log(99, "Calc error with communicator")
            var = [nn.NdArray()]
            var[0].data = error
            _all_reduce(comm, var, division=False, inplace=True)
            ret = sum + var[0].data
        else:
            ret = sum + error
        return ret

    for name, mon in config.monitors.items():
        m = mon.monitor
        error_sum_monitor = 0.0
        error_count = 0
        di = mon.data_iterator

        for i in range(di.size // di.batch_size):
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
                error_sum = 0.0
                for v in m.monitor_variables:
                    error_sum += np.mean(v.variable_instance.d)
                    v.variable_instance.data.zero()
                error_sum_monitor = _sum_error(error_sum_monitor, error_sum)
                if single_or_rankzero():
                    progress('Evaluating "{0}"'.format(
                        name) + ' : error={0:0.6f}'.format(
                        error_sum_monitor / error_count),
                        di.position * 1.0 / di.size)
            error_count += comm.size if comm else 1

            # Forward recursive
            m.network.forward(m.forward_sequence)

        # Sum error at the end of dataset
        error_sum = 0.0
        for v in m.monitor_variables:
            error_sum += np.mean(v.variable_instance.d)
            v.variable_instance.data.zero()
        error_sum_monitor = _sum_error(error_sum_monitor, error_sum)

        if error_count == 0:
            error = 0
        else:
            error = error_sum_monitor / error_count
        monitoring_report.append('  {}: {}\n'.format(name, error))

        # Console only start
        status.set_val(['monitoring_report', epoch, name], error)
        status.set_val(['last', name], error)  # save last value
        # Console only end

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
    if single_or_rankzero():
        if (not config.training_config.save_best) or \
           (not best_error) or \
           (best_error is not None and valid_error <= best_error):
            best_error = valid_error
            # Console only start
            status.set_val('best.valid_error', best_error)
            status.set_val('best.epoch', epoch)
            # Console only end
            _save_parameters(args, 'best', epoch, True)

    return best_error, error_str


def _get_current_parameter(args):

    globname = os.path.join(args.outdir, 'results_current_*.nnp')
    exists = glob.glob(globname)

    if len(exists) > 0:
        ex_list = {}

        for ex in exists:
            n = int(ex.rsplit('_', 1)[1].rsplit('.', 1)[0])
            ex_list[n] = ex

        last_epoch = sorted(ex_list.keys())[0]
        last_parameter = ex_list[last_epoch]
        logger.log(99, "Load parameter from [{}]".format(
            os.path.basename(last_parameter)))
        load.load([last_parameter], parameter_only=True)
        return last_epoch

    return 0


def _calc_estimate_time(timeinfo, max_iter, last_iter, iter):
    timeinfo.past_time = time.time() - timeinfo.start_time
    timeinfo.estimate_time = timeinfo.past_time * \
        (max_iter - last_iter) / (iter - last_iter)
    timeinfo.remain_time = timeinfo.estimate_time - timeinfo.past_time
    timeinfo.last_past_time = timeinfo.past_time
    return timeinfo


def _train(args, config):
    global _save_parameter_info
    comm = current_communicator()

    last_epoch = 0
    if args.resume:
        last_epoch = _get_current_parameter(args)
        logger.log(99, "Resume from epoch {}".format(last_epoch + 1))

    # Console only start
    status.set_val('epoch.max', config.training_config.max_epoch)
    status.set_val('epoch.current', last_epoch + 1
                   if last_epoch < config.training_config.max_epoch
                   else config.training_config.max_epoch)
    # Console only end

    max_iteration = config.training_config.max_epoch * \
        config.training_config.iter_per_epoch
    if single_or_rankzero():
        logger.log(99, 'Training epoch {} of {} begin'.format(last_epoch + 1,
                                                              config.training_config.max_epoch))

    class Cost:
        pass
    cost = Cost()
    cost.sum_epoch = 0.0
    cost.num_iteration = 0
    cost.sum_iteration = 0.0
    cost.variables = None

    best_error = None

    class TimeInfo:
        pass
    timeinfo = TimeInfo()
    timeinfo.last_past_time = None

    if max_iteration > 0:
        last_iteration = last_epoch * config.training_config.iter_per_epoch
        if last_iteration < max_iteration:

            timeinfo.start_time = time.time()

            # Console only start
            status.start_process(start_time=timeinfo.start_time)
            status.dump(status='processing')
            # Console only end

            for iteration in range(last_iteration, max_iteration):

                cost = _update(iteration, config, cost)
                if (iteration - last_iteration) > 0:
                    timeinfo = _calc_estimate_time(
                        timeinfo, max_iteration, last_iteration, iteration)
                    # Console only start
                    status.update_time_train(prediction=timeinfo.estimate_time)
                    # Console only end
                    if config.timelimit > 0 and timeinfo.estimate_time > config.timelimit:
                        logger.log(99, 'Expected training time ({:.3f}s) will exceed time limit ({}s).'.format(
                            timeinfo.estimate_time, config.timelimit))
                        return False, False

                if (iteration + 1) % config.training_config.iter_per_epoch == 0:
                    last_past_time = -1
                    # End of epoch
                    epoch = iteration // config.training_config.iter_per_epoch + 1
                    cost_avg_epoch = cost.sum_epoch / cost.num_iteration
                    cost.sum_epoch = 0.0
                    cost.num_iteration = 0
                    monitoring_report = []

                    # Evaluation
                    error_str = ''
                    if epoch % config.training_config.monitor_interval == 0 or epoch <= 5:
                        best_error, error_str = _evaluate(
                            args, config, monitoring_report, best_error, epoch)

                    if single_or_rankzero():
                        # Write to monitoring_report.yml
                        f = open(os.path.join(
                            args.outdir, 'monitoring_report.yml'), 'a')
                        f.write('{}:\n'.format(epoch - 1))
                        f.write('  cost: {}\n'.format(cost_avg_epoch))
                        for s in monitoring_report:
                            f.write(s)
                        f.close()

                        # Console only start
                        status.set_val(['monitoring_report', epoch, 'cost'],
                                       cost_avg_epoch)
                        # Console only end

                        _save_parameters(args, 'current', epoch)

                        # Console only start
                        status.set_val('epoch.current', epoch)
                        status.dump()
                        # Console only end

                        logger.log(99, 'epoch {} of {} cost={:.6f} {} time=({:.1f}s /{:.1f}s)'.format(
                            epoch, config.training_config.max_epoch, cost_avg_epoch, error_str,
                            timeinfo.past_time, timeinfo.estimate_time))

                        if epoch < config.training_config.max_epoch:
                            elapsed_time = timeinfo.start_time - status.get_val('start_time') + timeinfo.past_time
                            avg_time_per_epoch = timeinfo.past_time / (epoch - last_epoch)

                            if args.time_limit is not None and (elapsed_time + avg_time_per_epoch) >= float(args.time_limit):
                                f = open(os.path.join(
                                    args.outdir, 'force_restart'), 'a')
                                f.write('remain_time: {}\n'.format(timeinfo.remain_time))
                                f.close()

                                _save_parameters(args, 'current', epoch, True)

                                return False, True

            if single_or_rankzero():
                _save_parameters(args, 'current', epoch, True)
    return True, False


def train_command(args):
    # Console only start
    status.init(args)
    # Console only end

    if single_or_rankzero():
        configure_progress(os.path.join(args.outdir, 'progress.txt'))

    info = load.load([args.config], exclude_parameter=True)

    # Check dataset uri is empty.
    dataset_error = False
    for dataset in info.datasets.values():
        if dataset.uri.strip() == '':
            dataset_error = True
    if dataset_error:
        logger.log(99, 'Fatal error. Dataset URI is empty.')
        return

    class TrainConfig:
        pass
    config = TrainConfig()
    config.timelimit = -1
    if args.param:
        load.load([args.param], parameter_only=True)

    # Console only start
    config.timelimit = get_info_from_sdcproj(args)
    # Console only end

    config.global_config = info.global_config
    config.training_config = info.training_config

    if single_or_rankzero():
        logger.log(99, 'Train with contexts {}'.format(available_contexts))

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
    comm = current_communicator()
    config.training_config.iter_per_epoch //= comm.size if comm else 1
    max_iteration = config.training_config.max_epoch * \
        config.training_config.iter_per_epoch

    global _save_parameter_info
    _save_parameter_info = {}
    _, config_ext = os.path.splitext(args.config)
    if config_ext == '.prototxt' or config_ext == '.nntxt':
        _save_parameter_info['config'] = args.config
    elif config_ext == '.nnp':
        with zipfile.ZipFile(args.config, 'r') as nnp:
            for name in nnp.namelist():
                _, ext = os.path.splitext(name)
                if ext == '.nntxt' or ext == '.prototxt':
                    nnp.extract(name, args.outdir)
                    _save_parameter_info['config'] = os.path.join(
                        args.outdir, name)

    result = False
    restart = False
    if max_iteration > 0:
        data_iterators = {'optimizer': {}, 'monitor': {}}
        rng = np.random.RandomState(comm.rank if comm else 0)
        with ExitStack() as stack:
            for name, o in config.optimizers.items():
                o.data_iterator = stack.enter_context(
                    o.optimizer.data_iterator())
                if comm and comm.size > 1:
                    o.data_iterator = o.data_iterator.slice(
                        rng, comm.size, comm.rank)
            for name, m in config.monitors.items():
                m.data_iterator = stack.enter_context(
                    m.monitor.data_iterator())
                if comm and comm.size > 1:
                    m.data_iterator = m.data_iterator.slice(
                        rng, comm.size, comm.rank)
            result, restart = _train(args, config)
    else:
        # save parameters without training (0 epoch learning)
        logger.log(99, '0 epoch learning. (Just save parameter.)')
        if single_or_rankzero():
            _save_parameters(args, 'current', 0, True)
        result = True

    if single_or_rankzero() and not restart:
        if result:
            logger.log(99, 'Training Completed.')
            # Console only start
            status.dump(status='finished')
            # Console only end
        else:
            logger.log(99, 'Training Incompleted.')
            # Console only start
            status.dump(status='failed')
            # Console only end
    if single_or_rankzero():
        progress(None)


def add_train_command(subparsers):
    # Train
    subparser = subparsers.add_parser('train', help='Training with NNP.')
    subparser.add_argument(
        '-a', '--assign', help='csv file that defines parameter assignment.')
    subparser.add_argument(
        '-r', '--resume', help='resume from last saved parameter.', action='store_true')
    subparser.add_argument(
        '-c', '--config', help='path to nntxt', required=True)
    subparser.add_argument(
        '-p', '--param', help='path to parameter file', required=False)
    subparser.add_argument(
        '-o', '--outdir', help='output directory', required=True)
    # Console only start
    subparser.add_argument(
        '-s', '--sdcproj', help='path to sdcproj', required=False)
    subparser.add_argument(
        '-j', '--job_url_list', help='path to job url list', required=False)
    subparser.add_argument(
        '-t', '--time_limit', help='exec time limit', required=False)
    # Console only end
    subparser.set_defaults(func=train_command)
