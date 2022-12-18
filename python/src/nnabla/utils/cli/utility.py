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
import re
import timeit
from contextlib import contextmanager

import nnabla as nn
import nnabla.utils.callback as callback
import numpy as np
from nnabla.config import nnabla_config
from nnabla.logger import logger
from nnabla.utils import nnabla_pb2
from nnabla.utils.communicator_util import current_communicator
from nnabla.utils.get_file_handle import get_file_handle_save, get_decorated_file_loader, load_files

cpu_load_backend_ok = True
try:
    import psutil
except Exception:
    # measure cpu load only if psutil installed
    cpu_load_backend_ok = False

gpu_load_backend_ok = True
try:
    from nnabla.utils import nvml
    nvml.nvmlInit()
except Exception:
    # measure gpu load only if nvml installed
    gpu_load_backend_ok = False

try:
    _ANALYSE_GPU_STATUS_INTERVAL = int(
        nnabla_config.get('MULTINODE', 'analyse_gpu_status_interval'))
    _GPU_SLOWING_WARNING_THRESHOLD = float(
        nnabla_config.get('MULTINODE', 'gpu_slowing_warning_threshold'))
    _GPU_SLOWING_ERROR_THRESHOLD = float(
        nnabla_config.get('MULTINODE', 'gpu_slowing_error_threshold'))
except Exception:
    _ANALYSE_GPU_STATUS_INTERVAL = 20
    _GPU_SLOWING_WARNING_THRESHOLD = 1.4
    _GPU_SLOWING_ERROR_THRESHOLD = 2

# load variable
# ============
gpu_m_count = 0
gpu_a_load = {}
if cpu_load_backend_ok:
    p_handler = psutil.Process()
    p_handler_avg = psutil.Process()
    p_handler_avg.cpu_percent()


def is_float(x):
    # x is string
    try:
        float(x)
        return True
    except ValueError:
        return False


def str_to_num(str):
    # Convert str to num.
    #   12345678
    #   123e123  123 ^ 123
    #   1234K    1234 * 1000
    #   1234k    1234 * 1024
    #   1234M    1234 * 1000^2
    #   1234m    1234 * 1024^2
    #   1234G    1234 * 1000^3
    #   1234g    1234 * 1024^3
    #   1234T    1234 * 1000^4
    #   1234t    1234 * 1024^4
    #   1234P    1234 * 1000^5
    #   1234p    1234 * 1024^5
    import re
    m = re.match(
        r'(^[1-9][0-9]*$|^([1-9][0-9]*(\.[0-9]+)?)((e)([0-9]+)|[KkMmGgTtPp]))$', str)
    print(m)
    if m:
        if m.group(4):
            if m.group(5):
                return int(float(m.group(2)) ** int(m.group(6)))
            else:
                return int(float(m.group(2)) *
                           {'K': 1000, 'k': 1024,
                            'M': 1000 ** 2, 'm': 1024 ** 2,
                            'G': 1000 ** 3, 'g': 1024 ** 3,
                            'T': 1000 ** 4, 't': 1024 ** 4,
                            'P': 1000 ** 5, 'p': 1024 ** 5}[m.group(4)])
        else:
            return int(m.group(1))
    else:
        return -1


def compute_full_path(root_path, file_path):
    full_path = os.path.join(root_path, file_path)
    full_path = full_path.replace('\\', os.path.sep)
    full_path = full_path.replace('/', os.path.sep)
    full_path = full_path.replace(os.path.sep + '.' + os.path.sep, os.path.sep)
    return full_path


def let_data_to_variable(variable, data, ctx=None, data_name=None, variable_name=None):
    try:
        if data.dtype <= np.float64:
            variable.data.cast(data.dtype)[...] = data
        else:
            variable.d = data
    except:
        if variable.shape != data.shape:
            logger.critical('Shape does not match between data{} and variable{} ({} != {}).'.format(
                ' "' + data_name + '"' if data_name else '',
                ' "' + variable_name + '"' if variable_name else '',
                data.shape, variable.shape))
        raise
    variable.need_grad = False

    # Copy to device
    if ctx:
        try:
            variable.data.cast(variable.data.dtype, ctx)
        except:
            if ctx.array_class != 'CpuArray':
                # Fallback to cpu
                ctx.array_class = 'CpuArray'
                variable.data.cast(variable.data.dtype, ctx)
            else:
                raise


def collect_and_shape_result(c_load, g_load):
    # c_load : float e.g. 58.5
    # g_load : [[nvidia_device_id, gpu_load]]

    comm = current_communicator()
    if comm:
        res = [[comm.rank, c_load], *g_load[:1]]
        t_load_ndarray = np.array(res).reshape(-1)

        load_var = nn.Variable([len(t_load_ndarray), ])
        load_var.d = t_load_ndarray
        load_list_var = [nn.Variable([len(t_load_ndarray), ])
                         for _ in range(comm.size)]
        comm.all_gather(load_var.data, [a.data for a in load_list_var])
        result_arr = [[*np.round(a.d.astype(float), decimals=1)]
                      for a in load_list_var]
    else:
        res = [[0, c_load], *g_load[:1]]
        t_load_ndarray = np.round(np.array(res).reshape(-1), decimals=1)
        result_arr = [[*t_load_ndarray.astype(float)]]

    result_arr = sorted(result_arr, key=lambda x: x[0])

    return result_arr


def measure_cpu_gpu_instant_load():
    # Get current cpu gpu load, as
    # load = [rank, cpu_load, nvidia_device_id, gpu_load]
    # result_arr: [load, load, ...]

    gpu_load = []
    if gpu_load_backend_ok:
        global gpu_a_load
        global gpu_m_count

        gpu_m_count += 1
        try:
            comm = current_communicator()
            if comm:
                index = comm.local_rank
            elif 'cuda' in str(nn.get_current_context().backend):
                index = 0
            else:
                raise Exception
            handler = nvml.nvmlDeviceGetHandleByIndex(index)
            gpu_load = [
                [index, nvml.nvmlDeviceGetUtilizationRates(handler).gpu]]

            if index in gpu_a_load.keys():
                gpu_a_load[index]['name'] = nvml.nvmlDeviceGetName(
                    handler).decode("utf-8")
                o_load = gpu_a_load[index]['load']
                n_load = gpu_load[0][1]
                gpu_a_load[index]['load'] = (
                    (gpu_m_count - 1) * o_load + n_load) / gpu_m_count
            else:
                gpu_a_load[index] = {
                    'name': nvml.nvmlDeviceGetName(handler).decode("utf-8"),
                    'load': gpu_load[0][1]
                }

        except Exception:
            gpu_load = []

    if cpu_load_backend_ok:
        global p_handler
        cpu_load = p_handler.cpu_percent()
        callback.update_status(
            ('cpu_gpu_load', collect_and_shape_result(cpu_load, gpu_load)))


def get_cpu_gpu_average_load():
    # Get average cpu gpu load, as
    # load = [rank, cpu_load, nvidia_device_id, gpu_load]
    # result_arr: [load, load, ...]

    g_load = []
    if gpu_load_backend_ok:
        global gpu_a_load
        global gpu_m_count

        load_info = {**gpu_a_load}
        gpu_a_load = {}
        gpu_m_count = 0

        # adjust data type then transfer them to numpy ndarray
        g_load = [[float(a), float(load_info[a]['load'])]
                  for a in load_info.keys()]
        g_load = g_load[:1] if len(g_load) else []

    if cpu_load_backend_ok:
        global p_handler_avg
        c_load = p_handler_avg.cpu_percent()
        return collect_and_shape_result(c_load, g_load)


def _create_optimizer_lite(opti):
    proto_o = nnabla_pb2.Optimizer()
    proto_o.name = opti.name
    proto_o.solver.type = re.sub(r'(|Cuda)$', '', str(opti.solver.name))
    opti.solver.set_states_to_protobuf(proto_o)
    return proto_o


def save_optimizer_states(filebase, ext, train_config):
    filelist = []
    if ext == '.protobuf':
        filename = filebase + '_optimizer.protobuf.optimizer'
        proto = nnabla_pb2.NNablaProtoBuf()
        proto_optimizers = []
        for o in train_config.optimizers.values():
            proto_optimizers.append(_create_optimizer_lite(o.optimizer))
        proto.optimizer.extend(proto_optimizers)
        with get_file_handle_save(filename, '.protobuf') as f:
            f.write(proto.SerializeToString())
            filelist.append(filename)
    else:
        for o in train_config.optimizers.values():
            f_name = '{}_{}_optimizer.h5'.format(
                o.optimizer.name,
                re.sub(r'(|Cuda)$', '', str(o.optimizer.solver.name))
            )
            filename = '{}_{}'.format(filebase, f_name)
            o.optimizer.solver.save_states(filename)
            name_ext = '{}.optimizer'.format(filename)
            os.rename(filename, name_ext)
            filelist.append(name_ext)
    return filelist


class NodeTimeInfoCollector:
    def __init__(self):
        self.timelist = []
        self._frequency = _ANALYSE_GPU_STATUS_INTERVAL
        self._warning_threshold = _GPU_SLOWING_WARNING_THRESHOLD
        self._error_threshold = _GPU_SLOWING_ERROR_THRESHOLD

    @contextmanager
    def collect_cost_time(self, comm, iter_n):
        if comm and iter_n % self._frequency == 0:
            t1 = timeit.default_timer()
            yield
            t2 = timeit.default_timer()
            self.timelist.append(t2 - t1)
            self._get_analysis(comm)
        else:
            yield

    def _reap_data(self):
        result = self.timelist[:]
        self.timelist = []
        return result

    def _collect_info_from_multinode(self, comm, _d_ndarray):
        load_var = nn.Variable([len(_d_ndarray), ])
        load_var.d = _d_ndarray
        load_list_var = [nn.Variable([len(_d_ndarray), ])
                         for _ in range(comm.size)]
        comm.all_gather(load_var.data, [a.data for a in load_list_var])

        return load_list_var

    def _reap_multinode_data(self, comm):
        c_data = np.array([comm.rank, sum(self._reap_data())])
        multinode_data = self._collect_info_from_multinode(comm, c_data)
        result_arr = [a.d for a in multinode_data]

        # result_arr is list(<class 'numpy.ndarray'>, ...) like
        return result_arr

    def _get_analysis(self, comm):

        def _analyse_gpu_cost_time(result, threshold):
            aver = np.mean(result, axis=0)[1]
            _node_l = [*filter(lambda n: n[1] > aver * threshold, result)]
            if len(_node_l):
                ranks = ', '.join([str(int(n[0])) for n in _node_l])
                _str = ('Gpu of Rank {} ran slower than average '
                        'by a factor of {} or more'.format(ranks, threshold))
                return _str
            return ''

        result = self._reap_multinode_data(comm)
        if comm.rank == 0:
            error_str = _analyse_gpu_cost_time(result, self._error_threshold)
            if error_str:
                logger.error(error_str)
                raise Exception(error_str)
            else:
                warning_str = _analyse_gpu_cost_time(
                    result, self._warning_threshold)
                if warning_str:
                    logger.warning(warning_str)


def load_train_state(filename, info):
    info.exclude_parameter = False
    info.parameter_only = False
    file_loaders = get_decorated_file_loader()
    info.parameter_scope = nn.parameter.get_current_parameter_scope()
    load_files(info, file_loaders, filename)
    logger.info("Load training resume states: {}".format(filename))


from contextlib import contextmanager
import nnabla.logger as logger
from nnabla.lms import SwapInOutScheduler


@contextmanager
def sechdule_scope(scheduler):
    scheduler.update_pre_hook()
    yield scheduler
    scheduler.update_post_hook()


def lms_scheduler(ctx, use_lms, gpu_memory_size=None, window_length=None):
    _check_list = [x.split(":")[0] for x in ctx.backend]
    if "cudnn" not in _check_list and "cuda" not in _check_list:
        logger.warning(
            "ctx passed to scheduler doesn't have cuda/cudnn backend. lms scheduler will not be used.")
        use_lms = False

    comm = current_communicator()
    if comm:
        logger.log(
            99, f'[OoC] Currently OoC is disabled for Multi-GPU training.')
        use_lms = False

    if use_lms:
        gpu_index = 0
        if 'cuda' in str(ctx.backend):
            gpu_index = int(ctx.device_id)
        else:
            logger.log(99, f'[OoC] OoC is only enabled for GPU training.')
            raise Exception

        if gpu_memory_size is None or gpu_memory_size == 0:
            try:
                handle = nvml.nvmlDeviceGetHandleByIndex(gpu_index)
                total_memory = nvml.nvmlDeviceGetMemoryInfo(handle).total
                gpu_memory_size = int(total_memory * 0.7)
            except:
                logger.log(
                    99, f'[OoC] Could not get GPU memory size using default value(6GB).')
                gpu_memory_size = 6e9  # default 6 GiB
                pass

        if window_length is None or window_length == 0:
            window_length = int(gpu_memory_size * 1.5)

        logger.log(
            99, f'[OoC] gpu_memory_limit: {gpu_memory_size / 1e9}GB, prefetch_window_length: {window_length / 1e9}GB')
        # Change array preference so that lms works well.
        # import nnabla_ext.cuda.init as cuda_init
        # cuda_init.prefer_cpu_pinned_array()
        # cuda_init.prefer_cuda_virtual_array()
        from nnabla.ext_utils import get_extension_context
        be, tc = ctx.backend[0].split(":")
        cpu_ctx = get_extension_context("cpu", device_id="", type_config=tc)
        return SwapInOutScheduler(cpu_ctx, ctx, gpu_memory_size, window_length)
    else:
        class DummyScheduler(object):
            function_pre_hook = None
            function_post_hook = None
            update_pre_hook = None
            update_post_hook = None

            def start_scheduling(self):
                return None

            def end_scheduling(self):
                return None

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

        return DummyScheduler()
