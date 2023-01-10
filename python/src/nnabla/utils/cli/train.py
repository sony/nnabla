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

import glob
import os
import time
import zipfile
from collections import OrderedDict

import nnabla as nn
import nnabla.utils.callback as callback
import nnabla.utils.load as load
import numpy as np
from contextlib2 import ExitStack  # Backport from python3
from nnabla import available_contexts
from nnabla.config import nnabla_config
from nnabla.logger import logger
from nnabla.parameter import save_parameters
from nnabla.utils.cli.utility import NodeTimeInfoCollector
from nnabla.utils.cli.utility import get_cpu_gpu_average_load
from nnabla.utils.cli.utility import let_data_to_variable
from nnabla.utils.cli.utility import lms_scheduler
from nnabla.utils.cli.utility import load_train_state
from nnabla.utils.cli.utility import measure_cpu_gpu_instant_load
from nnabla.utils.cli.utility import save_optimizer_states
from nnabla.utils.cli.utility import str_to_num
from nnabla.utils.communicator_util import current_communicator, single_or_rankzero
from nnabla.utils.nnp_format import nnp_version
from nnabla.utils.progress import configure_progress, progress
from six.moves import range

try:
    _OPTIMIZER_CHECKPOINT_INTERVAL = int(
        nnabla_config.get('CHECKPOINT', 'optimizer_interval'))
except Exception:
    _OPTIMIZER_CHECKPOINT_INTERVAL = 5


_save_parameter_info = {}


nodeTimeCollector = NodeTimeInfoCollector()


def _all_reduce(comm, var, division, inplace):
    comm.all_reduce(var, division=division, inplace=inplace)


def _save_parameters(args, suffix, epoch, train_config, force=False):
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
    base_candidate = callback.result_base(base, suffix, args.outdir)
    if base_candidate is None:
        if suffix is None or suffix == 'best':
            base = os.path.join(args.outdir, 'results')
    else:
        base = base_candidate

    filename = base + '.nnp'

    if force or (not os.path.exists(filename) and (timediff > 180.0 or epochdiff > 10)):

        # Remove existing nnp before saving new file.
        for exist in exists:
            os.unlink(exist)

        version_filename = base + '_version.txt'

        with open(version_filename, 'w') as file:
            file.write('{}\n'.format(nnp_version()))

        param_filename = f'{base}_param{nnabla_config.get("MISC", "nnp_param_format")}'
        save_parameters(param_filename)

        need_save_opti = train_config.optimizers and epoch % _OPTIMIZER_CHECKPOINT_INTERVAL == 0
        if need_save_opti:
            opti_filenames = save_optimizer_states(
                base, '.h5', train_config)

        with zipfile.ZipFile(filename, 'w') as nnp:
            nnp.write(version_filename, 'nnp_version.txt')
            nnp.write(_save_parameter_info['config'], os.path.basename(
                _save_parameter_info['config']))
            nnp.write(
                param_filename, f'parameter{nnabla_config.get("MISC", "nnp_param_format")}')
            if need_save_opti:
                for f in opti_filenames:
                    nnp.write(f, f[len(base) + 1:])

        os.unlink(version_filename)
        os.unlink(param_filename)
        if need_save_opti:
            for f in opti_filenames:
                os.unlink(f)

        _save_parameter_info[suffix]['epoch'] = epoch
        _save_parameter_info[suffix]['time'] = current_time

        callback.save_train_snapshot()


def _update(iter, config, cost, scheduler):
    comm = current_communicator()

    loaded_data = {}
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

    def _get_reserved_variable(shape, reserved_variable_name, iter, iter_per_epoch, max_epoch):
        if reserved_variable_name == "%iter":
            value = iter
        elif reserved_variable_name == "%max_iter":
            value = max_epoch * iter_per_epoch
        elif reserved_variable_name == "%epoch":
            value = iter // iter_per_epoch
        elif reserved_variable_name == "%epochf":
            value = iter * 1.0 / iter_per_epoch
        elif reserved_variable_name == "%max_epoch":
            value = max_epoch
        elif reserved_variable_name == "%progress":
            value = (iter * 1.0 / iter_per_epoch) / max_epoch
        else:
            raise ValueError("Unknown reserved variable {}".format(
                reserved_variable_name))
        return value

    for opt in config.optimizers.values():
        o = opt.optimizer
        if (o.start_iter == 0 or iter + 1 >= o.start_iter) and (o.end_iter == 0 or iter + 1 <= o.end_iter):
            # Load dataset
            data = OrderedDict()
            for di in opt.data_iterators:
                if di not in loaded_data:
                    loaded_data[di] = di.next()
                data.update(zip(di.variables, loaded_data[di]))
            for v, d in o.dataset_assign.items():
                # TODO: here we consume a bit more memory for loading all edge nodes to cuda
                # dest_context = config.global_config.default_context if not o.forward_sequence or v not in o.forward_sequence[
                #     0].inputs else None
                dest_context = config.global_config.default_context
                if d not in data and d[0] == "%":
                    value = _get_reserved_variable(
                        v.variable_instance.shape, d, iter, config.training_config.iter_per_epoch, config.training_config.max_epoch)
                    v.variable_instance.data.fill(value)
                elif d in data:
                    let_data_to_variable(v.variable_instance, data[
                                         d], ctx=dest_context,
                                         data_name=d, variable_name=v.name)
                else:
                    raise ValueError('Variable "{}" is not found in dataset "{}", optimizer "{}"'.format(
                        d, ', '.join(o.data_iterators.keys()), o.name))

            # Generate data
            for v, generator in o.generator_assign.items():
                # TODO: here we consume a bit more memory for loading all edge nodes to cuda
                # dest_context = config.global_config.default_context if not o.forward_sequence or v not in o.forward_sequence[
                #     0].inputs else None
                dest_context = config.global_config.default_context
                let_data_to_variable(v.variable_instance,
                                     data=generator(v.variable_instance.d.shape), ctx=dest_context,
                                     variable_name=v.name)

            # Monitor loss before forward to prepare input data while processing on
            # GPU
            if cost.variables:
                for l in cost.variables:
                    cost.sum_iteration += np.mean(l.variable_instance.d)
                    # l.variable_instance.data.zero()
                if is_first_optimizer:
                    is_first_optimizer = False
                    _sum_cost()
                    if single_or_rankzero():
                        progress("Training : cost={0:0.6f}".format(cost.sum_iteration),
                                 (iter % config.training_config.iter_per_epoch) * 1.0 / config.training_config.iter_per_epoch)
                    cost.sum_iteration = 0.0

            with scheduler:
                # This size is omitted if multiple variables in loss_variables
                # otherwise, this size is omitted since passing through F.sink()
                l_size = o.loss_variables[0].variable_instance.size

                with nodeTimeCollector.collect_cost_time(comm, iter):
                    # Forward
                    o.target.forward(clear_no_need_grad=True)

                    # Equivalency with previous version
                    if iter % o.update_interval == 0:
                        o.solver.zero_grad()

                    # Backward
                    if o.comm and iter % o.update_interval == o.update_interval - 1:
                        params = [x.grad for x in o.parameters.values()]
                        o.target.backward(grad=1.0 / l_size,
                                          clear_buffer=True, communicator_callbacks=comm.all_reduce_callback(params, 2 << 20, division=True))
                    else:
                        o.target.backward(grad=1.0 / l_size, clear_buffer=True)

                # Update
                if iter % o.update_interval == o.update_interval - 1:
                    if o.weight_decay > 0:
                        o.solver.weight_decay(o.weight_decay)

                    if o.scheduler is not None:
                        o.solver.set_learning_rate(
                            o.scheduler.get_learning_rate(iter))
                    o.solver.update()

            # Reserve monitor loss
            cost.variables = o.loss_variables

    # Monitor loss at the end of epoch
    if iter % config.training_config.iter_per_epoch == config.training_config.iter_per_epoch - 1 and cost.variables:
        for l in cost.variables:
            cost.sum_iteration += np.mean(l.variable_instance.d)
            # l.variable_instance.data.zero()
        _sum_cost()
        cost.variables = None
        cost.sum_iteration = 0.0

    return cost


def _evaluate(args, config, monitoring_report, best_error, epoch, scheduler):
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
        data_size = max([di.size for di in mon.data_iterators])
        batch_size = max([di.batch_size for di in mon.data_iterators])

        for i in range(data_size // batch_size):
            # Load dataset
            data = OrderedDict()
            for di in mon.data_iterators:
                data.update(zip(di.variables, di.next()))

            # Set data to variable
            for v, d in m.dataset_assign.items():
                # NOTICE: trivially increase cuda memory usage for loading all edge nodes
                dest_context = config.global_config.default_context
                let_data_to_variable(v.variable_instance, data[
                                     d], ctx=dest_context,
                                     data_name=d, variable_name=v.name)

            # Generate data
            for v, generator in m.generator_assign.items():
                # NOTICE: trivially increase cuda memory usage for loading all edge nodes
                dest_context = config.global_config.default_context
                let_data_to_variable(v.variable_instance,
                                     data=generator(v.variable_instance.d.shape), ctx=dest_context,
                                     variable_name=v.name)

            # Sum error before forward to prepare input data while processing
            # on GPU
            if error_count > 0:
                error_sum = 0.0
                for v in m.monitor_variables:
                    error_sum += np.mean(v.variable_instance.d)
                    # v.variable_instance.data.zero()
                error_sum_monitor = _sum_error(error_sum_monitor, error_sum)
                if single_or_rankzero():
                    progress('Evaluating "{0}"'.format(
                        name) + ' : error={0:0.6f}'.format(
                        error_sum_monitor / error_count),
                        di.position * 1.0 / di.size)
            error_count += comm.size if comm else 1

            with scheduler:
                # Forward recursive
                m.target.forward(clear_no_need_grad=True, clear_buffer=True)

        # Sum error at the end of dataset
        error_sum = 0.0
        for v in m.monitor_variables:
            error_sum += np.mean(v.variable_instance.d)
            # v.variable_instance.data.zero()
        error_sum_monitor = _sum_error(error_sum_monitor, error_sum)

        if error_count == 0:
            error = 0
        else:
            error = error_sum_monitor / error_count

        if np.isnan(error) or np.isinf(error):
            logger.log(99, 'Validation error is Nan')
            error = 0.0

        monitoring_report.append('  {}: {}\n'.format(name, error))

        callback.update_status((['monitoring_report', epoch, name], error))
        callback.update_status((['last', name], error))  # save last value

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
            callback.update_status(('best.valid_error', best_error))
            callback.update_status(('best.epoch', epoch))
            _save_parameters(args, 'best', epoch, config, True)

    return best_error, error_str


def _get_current_parameter(args, config):
    def convert_to_info(config):
        class Info:
            pass
        ret = Info()
        ret.optimizers = OrderedDict()
        for name, opt in config.optimizers.items():
            ret.optimizers[name] = opt.optimizer
        return ret

    best_error, best_epoch = callback.get_best_from_status(args)

    globname = os.path.join(args.outdir, 'results_current_*.nnp')
    exists = glob.glob(globname)

    if len(exists) > 0:
        ex_list = {}

        info = convert_to_info(config)
        for ex in exists:
            n = int(ex.rsplit('_', 1)[1].rsplit('.', 1)[0])
            ex_list[n] = ex

        last_epoch = sorted(ex_list.keys(), reverse=True)[0]
        last_parameter = ex_list[last_epoch]
        logger.log(99, "Load parameter from [{}]".format(
            os.path.basename(last_parameter)))
        # load.load([last_parameter], parameter_only=True)
        load_train_state(last_parameter, info)
        return last_epoch, best_epoch, best_error

    return 0, best_epoch, best_error


def _calc_estimate_time(timeinfo, max_iter, last_iter, iter):
    timeinfo.past_time = time.time() - timeinfo.start_time
    timeinfo.estimate_time = timeinfo.past_time * \
        (max_iter - last_iter) / (iter - last_iter)
    timeinfo.remain_time = timeinfo.estimate_time - timeinfo.past_time
    timeinfo.last_past_time = timeinfo.past_time
    return timeinfo


def _calc_epoch_span(timeinfo):
    cur_time = time.time()
    span = cur_time - timeinfo.last_epoch_start_time
    timeinfo.last_epoch_start_time = cur_time
    return span


def _format_cgload_log(cg_load):
    narr = np.array(cg_load).T
    if narr.shape[0] == 4:
        log_str = 'average load:{{cpu:{:.1f}%, gpu:{:.1f}%}}'.format(
            np.mean(narr[1]), np.mean(narr[3]))
    else:
        log_str = 'average load:{{cpu:{:.1f}%}}'.format(
            np.mean(narr[1]))
    return log_str


def _train(args, config):
    global _save_parameter_info
    comm = current_communicator()
    _CGLOAD_LOG_INTERVAL = 20

    best_epoch = None
    best_error = None
    last_epoch = 0
    if args.resume:
        last_epoch, best_epoch, best_error = _get_current_parameter(
            args, config)
        if best_epoch is not None:
            logger.log(
                99, "Best error {} recorded at epoch {} in previous training.".format(best_error,
                                                                                      best_epoch))
            if best_epoch > last_epoch:
                logger.log(
                    99, "Resumed epoch is {} but this training keep this result.".format(last_epoch))
        logger.log(99, "Resume from epoch {}".format(last_epoch + 1))

    callback.update_status(('epoch.max', config.training_config.max_epoch))
    callback.update_status(('epoch.current', last_epoch + 1
                            if last_epoch < config.training_config.max_epoch
                            else config.training_config.max_epoch))

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

    class TimeInfo:
        pass
    timeinfo = TimeInfo()
    timeinfo.past_time = 0
    timeinfo.estimate_time = 0
    timeinfo.last_past_time = None

    ctx = config.global_config.default_context

    if args.enable_ooc:
        logger.log(99, 'OOC enabled.')
        logger.log(99, f'    OOC GPU memory size {args.ooc_gpu_memory_size}.')
        logger.log(99, f'    OOC Window length   {args.ooc_window_length}.')
    scheduler = lms_scheduler(
        ctx,
        use_lms=args.enable_ooc,
        gpu_memory_size=args.ooc_gpu_memory_size,
        window_length=args.ooc_window_length)
    val_scheduler = lms_scheduler(
        ctx,
        use_lms=args.enable_ooc,
        gpu_memory_size=args.ooc_gpu_memory_size,
        window_length=args.ooc_window_length)

    if max_iteration > 0:
        last_iteration = last_epoch * config.training_config.iter_per_epoch
        if last_iteration < max_iteration:

            timeinfo.start_time = time.time()
            timeinfo.last_epoch_start_time = timeinfo.start_time

            callback.update_status('processing', True, timeinfo.start_time)

            for iteration in range(last_iteration, max_iteration):

                # instant load measurement
                measure_cpu_gpu_instant_load()

                cost = _update(iteration, config, cost, scheduler)

                if np.isnan(cost.sum_epoch) or np.isinf(cost.sum_epoch):
                    logger.log(99, 'Cost is Nan')
                    return False, False

                timeinfo = _calc_estimate_time(
                    timeinfo, max_iteration, last_iteration, iteration + 1)
                callback.update_time_train(prediction=timeinfo.estimate_time)

                if 0 < config.timelimit < timeinfo.estimate_time:
                    logger.log(99, 'Expected training time ({:.3f}s) will exceed time limit ({}s).'.format(
                        timeinfo.estimate_time, config.timelimit))
                    return False, False

                if (iteration + 1) % config.training_config.iter_per_epoch == 0:
                    last_past_time = -1
                    # End of epoch
                    epoch = iteration // config.training_config.iter_per_epoch + 1
                    cost_avg_epoch = cost.sum_epoch / cost.num_iteration if cost.num_iteration else 0
                    cost.sum_epoch = 0.0
                    cost.num_iteration = 0
                    monitoring_report = []

                    # Evaluation
                    error_str = ''
                    if epoch % config.training_config.monitor_interval == 0 or epoch <= 5:
                        best_error, error_str = _evaluate(
                            args, config, monitoring_report, best_error, epoch, val_scheduler)

                    # Cpu/Gpu average load
                    cg_load_str = ''
                    cgload_log = ''
                    cg_load = get_cpu_gpu_average_load()
                    if cg_load:
                        cg_load_str = 'epoch {} average_load_matrix: {}'.format(
                            epoch, cg_load)
                        span = _calc_epoch_span(timeinfo)
                        if span > _CGLOAD_LOG_INTERVAL:
                            cgload_log = _format_cgload_log(cg_load)

                    if single_or_rankzero():
                        # Write to monitoring_report.yml
                        f = open(os.path.join(
                            args.outdir, 'monitoring_report.yml'), 'a')
                        f.write('{}:\n'.format(epoch - 1))
                        f.write('  cost: {}\n'.format(cost_avg_epoch))
                        for s in monitoring_report:
                            f.write(s)
                        f.close()

                        callback.update_status((['monitoring_report', epoch, 'cost'],
                                                cost_avg_epoch))

                        _save_parameters(args, 'current', epoch, config)

                        callback.update_status(('epoch.current', epoch))
                        callback.update_status()

                        logger.log(99, 'epoch {} of {} cost={:.6f} {} time=({:.1f}s /{:.1f}s) {}'.format(
                            epoch, config.training_config.max_epoch, cost_avg_epoch, error_str,
                            timeinfo.past_time, timeinfo.estimate_time, cgload_log))

                        if cg_load_str:
                            # cpu_gpu_average_load record at epoch level
                            callback.update_status(
                                (['cpu_gpu_epoch_load', epoch], cg_load))
                            progress(cg_load_str, 1)

                        if not callback.check_training_time(args, config, timeinfo, epoch, last_epoch):
                            _save_parameters(
                                args, 'current', epoch, config, True)
                            return False, True

            if single_or_rankzero():
                _save_parameters(args, 'current', epoch, config, True)
    return True, False


def train_command(args):
    if args.ooc_gpu_memory_size is not None:
        ooc_gpu_memory_size = str_to_num(args.ooc_gpu_memory_size)
        if ooc_gpu_memory_size < 0:
            logger.log(
                99, f'Fatal error. invalid ooc_gpu_memory_size [{args.ooc_gpu_memory_size}].')
            return False
        args.ooc_gpu_memory_size = ooc_gpu_memory_size
    if args.ooc_window_length is not None:
        ooc_window_length = str_to_num(args.ooc_window_length)
        if ooc_window_length < 0:
            logger.log(
                99, f'Fatal error. invalid ooc_window_length [{args.ooc_window_length}].')
            return False
        args.ooc_window_length = ooc_window_length

    callback.update_status(args)

    if single_or_rankzero():
        configure_progress(os.path.join(args.outdir, 'progress.txt'))

    info = load.load([args.config], prepare_data_iterator=None,
                     exclude_parameter=True, context=args.context)

    # Check dataset uri is empty.
    dataset_error = False
    for dataset in info.datasets.values():
        if dataset.uri.strip() == '':
            dataset_error = True
    if dataset_error:
        logger.log(99, 'Fatal error. Dataset URI is empty.')
        return False

    class TrainConfig:
        pass
    config = TrainConfig()
    config.timelimit = -1
    if args.param:
        # If this parameter file contains optimizer information
        # we need to info to recovery.
        # load.load([args.param], parameter_only=True)
        load_train_state(args.param, info)

    config.timelimit = callback.get_timelimit(args)

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
        rng = np.random.RandomState(comm.rank if comm else 0)
        with ExitStack() as stack:
            # Create data_iterator instance only once for each dataset in optimizers
            optimizer_data_iterators = {}
            for name, o in config.optimizers.items():
                for di in o.optimizer.data_iterators.values():
                    if di not in optimizer_data_iterators:
                        di_instance = stack.enter_context(di())
                        if comm and comm.size > 1:
                            di_instance = di_instance.slice(
                                rng, comm.size, comm.rank)
                        optimizer_data_iterators[di] = di_instance
                    else:
                        di_instance = optimizer_data_iterators[di]
                    o.data_iterators.append(di_instance)

            # Create data_iterator instance only once for each dataset in monitors
            monitor_data_iterators = {}
            for name, m in config.monitors.items():
                for di in m.monitor.data_iterators.values():
                    if di not in monitor_data_iterators:
                        di_instance = stack.enter_context(di())
                        if comm and comm.size > 1:
                            di_instance = di_instance.slice(
                                rng, comm.size, comm.rank)
                        monitor_data_iterators[di] = di_instance
                    else:
                        di_instance = monitor_data_iterators[di]
                    m.data_iterators.append(di_instance)
            monitor_data_iterators.update(optimizer_data_iterators)

            result, restart = _train(args, config)
    else:
        # save parameters without training (0 epoch learning)
        logger.log(99, '0 epoch learning. (Just save parameter.)')
        if single_or_rankzero():
            _save_parameters(args, None, 0, config, True)
        result = True

    if single_or_rankzero() and not restart:
        if result:
            logger.log(99, 'Training Completed.')
            callback.update_status('finished')
        else:
            logger.log(99, 'Training Incompleted.')
            callback.update_status('failed')
    if single_or_rankzero():
        progress(None)
    return True


def add_train_command(subparsers):
    # Train
    subparser = subparsers.add_parser('train', help='Training with NNP.')
    subparser.add_argument(
        '-r', '--resume', help='Resume from last saved parameter', action='store_true')
    subparser.add_argument(
        '-c', '--config', help='Path to nntxt', required=True)
    subparser.add_argument(
        '-p', '--param', help='Path to parameter file', required=False)
    subparser.add_argument(
        '-o', '--outdir', help='Output directory', required=True)
    subparser.add_argument(
        '-O', '--enable-ooc', help='Enable Out Of Core training', action='store_true')
    subparser.add_argument(
        '-m', '--ooc-gpu-memory-size', help='OOC gpu memory size (INTEGER or NUMeNUM or NUM[KkMmGgTtPp])', default=None)
    subparser.add_argument(
        '-C', '--context', help='Force exec context (cpu or cudnn[:DEVID])', default=None)
    subparser.add_argument(
        '-w', '--ooc-window-length', help='OOC window length (INTEGER or NUMeNUM or NUM[KkMmGgTtPp])', default=None)
    callback.add_train_command_arg(subparser)
    subparser.set_defaults(func=train_command)
