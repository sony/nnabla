# Copyright (c) 2020 Sony Corporation. All Rights Reserved.
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

import pytest
from contextlib import contextmanager, ExitStack
import numpy as np
import nnabla as nn
from functools import partial
from collections import OrderedDict
import itertools
import time
import glob
import zipfile
from legacy_load import load as ref_load

from nnabla.utils import load
from nnabla.utils import save

from nnabla.utils.nnp_format import nnp_version
from nnabla.logger import logger


from nntxt import NNTXT_EQUIVALENCE_CASES, NNTXT_IMPROVEMENT_CASES, CASE_INDEX

from nnabla.testing import assert_allclose
from helper import generate_csv_png, assert_tensor_equal, create_temp_with_dir, generate_case_from_nntxt_str
from nnabla.utils.data_iterator import data_iterator_csv_dataset
from nnabla.utils.communicator_util import current_communicator, single_or_rankzero
from nnabla.utils.cli.utility import let_data_to_variable, save_optimizer_states, load_train_state

# The mainly difference between legacy implementation and refactor-ed implementation:
#  - legacy parameters are included in variables, refactor-ed is included in parameters
#  - legacy [-1, x, y, z] will be replaced to [b, x, y, z]. But refactor-ed implementation
#     keep proto variable's shape as [-1, x, y, z], only computation graph's variable's
#     shape is replaced as [b, x, y, z]
#  - legacy set loss's gradient to 1.0 / tensor.size, but we think this should be incorrect
#     we should use default value 1.0, instead.


class Verifier:
    def __call__(self, pf):
        inputs = ','.join(pf.inputs)
        outputs = ','.join(pf.outputs)
        print("{}:{}, i:{}, o:{}".format(pf.type, pf.name, inputs, outputs))


def _ref_forward(args, index, config, data, variables, output_image=True):
    for e in config.executors:
        for v, d in e.dataset_assign.items():
            vind = variables.index(d)
            if v.variable_instance.d.shape != data[vind].shape:
                let_data_to_variable(v.variable_instance,
                                     np.reshape(
                                         data[vind], v.variable_instance.d.shape),
                                     data_name=d, variable_name=v.name)
            else:
                let_data_to_variable(v.variable_instance,
                                     data[vind].astype(
                                         v.variable_instance.d.dtype),
                                     data_name=d, variable_name=v.name)

        # Generate data
        for v, generator in e.generator_assign.items():
            v.variable_instance.d = generator(v.shape)

        # Forward recursive
        sum = [np.zeros(o.shape, dtype=o.variable_instance.d.dtype)
               for o in e.output_assign.keys()]
        for i in range(e.num_evaluations):
            e.network.forward(e.forward_sequence)
            if e.need_back_propagation:
                e.network.backward(e.backward_sequence)

            for o_index, o in enumerate(e.output_assign.keys()):
                if e.repeat_evaluation_type == "last":
                    sum[o_index] = o.variable_instance.d
                else:
                    sum[o_index] += o.variable_instance.d
        if e.repeat_evaluation_type == "last":
            avg = sum
        else:
            avg = [s / e.num_evaluations for s in sum]

    return avg


def _forward(args, index, config, data, variables, output_image=True):
    for e in config.executors:
        for v, d in e.dataset_assign.items():
            vind = variables.index(d)
            if v.variable_instance.d.shape != data[vind].shape:
                let_data_to_variable(v.variable_instance,
                                     np.reshape(
                                         data[vind], v.variable_instance.d.shape),
                                     data_name=d, variable_name=v.name)
            else:
                let_data_to_variable(v.variable_instance,
                                     data[vind].astype(
                                         v.variable_instance.d.dtype),
                                     data_name=d, variable_name=v.name)

        # Generate data
        for v, generator in e.generator_assign.items():
            v.variable_instance.d = generator(v.variable_instance.d.shape)

        # Forward recursive
        sum = [np.zeros(o.variable_instance.d.shape, dtype=o.variable_instance.d.dtype)
               for o in e.output_assign.keys()]
        for i in range(e.num_evaluations):
            e.forward_target.forward(clear_buffer=True)
            if e.need_back_propagation:
                e.backward_target.backward(clear_buffer=True)

            for o_index, o in enumerate(e.output_assign.keys()):
                if e.repeat_evaluation_type == "last":
                    sum[o_index] = o.variable_instance.d
                else:
                    sum[o_index] += o.variable_instance.d
        if e.repeat_evaluation_type == "last":
            avg = sum
        else:
            avg = [s / e.num_evaluations for s in sum]

    return avg


def common_forward(info, forward_func):
    batch_size = 1

    class ForwardConfig:
        pass

    class Args:
        pass

    args = Args()

    config = ForwardConfig
    if hasattr(info, 'global_config'):
        config.global_config = info.global_config
    config.executors = info.executors.values()
    config.networks = []
    for e in config.executors:
        if e.network.name in info.networks.keys():
            config.networks.append(info.networks[e.network.name])
        else:
            assert False, "{} is not found.".format(e.network.name)

    normalize = True
    for d in info.datasets.values():
        args.dataset = d.uri
        normalize = d.normalize
        break
    for e in config.executors:
        normalize = normalize and not e.no_image_normalization

    data_iterator = (lambda: data_iterator_csv_dataset(
        uri=args.dataset,
        batch_size=config.networks[0].batch_size,
        shuffle=False,
        normalize=normalize,
        with_memory_cache=False,
        with_file_cache=False))

    result = []
    with data_iterator() as di:
        index = 0
        while index < di.size:
            data = di.next()
            avg = forward_func(
                args, index, config, data, di.variables)
            index += len(avg[0])
            result.append(avg[0])

    return np.array(result)


#########################################
# test training scripts
#########################################
class TrainConfig:
    pass


current_path = os.path.dirname(os.path.abspath(__file__))


def print_variable_from_file(var_name, iter_num=0):
    var_name = var_name.replace('/', '-')
    var_a = np.load(os.path.join(current_path, "logdata",
                                 "ref-{}-{}.npy".format(var_name, iter_num)))
    var_b = np.load(os.path.join(current_path, "logdata",
                                 "new-{}-{}.npy".format(var_name, iter_num)))
    print("legacy==> {}".format(var_a.shape))
    print(var_a)
    print("new==> {}".format(var_b.shape))
    print(var_b)
    print("equal!" if np.allclose(var_a, var_b) else "not equal")
    grad_a = np.load(os.path.join(current_path, "logdata",
                                  "ref-{}-{}-g.npy".format(var_name, iter_num)))
    grad_b = np.load(os.path.join(current_path, "logdata",
                                  "new-{}-{}-g.npy".format(var_name, iter_num)))
    print("legacy gradient==> {}".format(grad_a.shape))
    # print(grad_a)
    print("new gradient==> {}".format(grad_b.shape))
    # print(grad_b)
    print("equal!" if np.allclose(grad_a, grad_b) else "not equal")


def load_varaible_from_file(var_name, iter_num=0):
    var_a = np.load(os.path.join(current_path, "logdata",
                                 "legacy-{}-{}.npy".format(var_name, iter_num)))
    var_b = np.load(os.path.join(current_path, "logdata",
                                 "new-{}-{}.npy".format(var_name, iter_num)))
    print("legacy==> {}".format(var_a.shape))
    # print(var_a)
    print("new==> {}".format(var_b.shape))
    # print(var_b)
    print("equal!" if np.allclose(var_a, var_b) else "not equal")
    grad_a = np.load(os.path.join(current_path, "logdata",
                                  "legacy-{}-{}-g.npy".format(var_name, iter_num)))
    grad_b = np.load(os.path.join(current_path, "logdata",
                                  "new-{}-{}-g.npy".format(var_name, iter_num)))
    print("legacy gradient==> {}".format(grad_a.shape))
    # print(grad_a)
    print("new gradient==> {}".format(grad_b.shape))
    # print(grad_b)
    print("equal!" if np.allclose(grad_a, grad_b) else "not equal")


# def get_variable_by_name(prefix, var_name, iter_num=0):
#     return np.load(os.path.join(current_path, "logdata",
#                                  "{}-{}-{}.npy".format(prefix, var_name, iter_num)))

def get_variable_by_name(prefix, net_name, var_name, iter_num=0):
    return np.load(os.path.join(current_path, "logdata",
                                "{}-{}-{}-{}.npy".format(prefix, net_name, var_name.replace('/', '-'), iter_num)))


def _update(config):
    loaded_data = {}
    sum_iteration = 0.0
    for opt in config.optimizers.values():
        o = opt.optimizer
        data = OrderedDict()
        for di in opt.data_iterators:
            if di not in loaded_data:
                loaded_data[di] = di.next()
            data.update(zip(di.variables, loaded_data[di]))
        for v, d in o.dataset_assign.items():
            dest_context = config.global_config.default_context
            if d in data:
                let_data_to_variable(v.variable_instance, data[
                                     d], ctx=dest_context,
                                     data_name=d, variable_name=v.name)
            else:
                raise ValueError('Variable "{}" is not found in dataset "{}", optimizer "{}"'.format(
                    d, ', '.join(o.data_iterators.keys()), o.name))

        for v, generator in o.generator_assign.items():
            dest_context = config.global_config.default_context
            let_data_to_variable(v.variable_instance,
                                 data=generator(v.variable_instance.d.shape), ctx=dest_context,
                                 variable_name=v.name)

        # Notice: here is special place for new version.
        if config.iter % o.update_interval == 0:
            o.solver.zero_grad()

        if config.save_optimizer_variable:
            for pv in o.net_variables.values():
                if pv not in itertools.chain(o.dataset_assign.keys(), o.generator_assign.keys()):
                    pv.variable_instance.persistent = True

        # call on iteration.
        if config.on_iter:
            config.on_iter(config)

        # o.network.forward(o.forward_sequence)
        # o.network.backward(o.backward_sequence, iter %
        #                    o.update_interval == 0)
        if config.iter >= config.start_iteration:
            config.cb.forward(o)
            config.cb.backward(o, config.iter % o.update_interval == 0)

            if config.save_optimizer_variable:
                for k, v in o.net_variables.items():
                    fn_d = os.path.join(current_path, "logdata", "{}-{}-{}.npy".format(
                        config.impl, k.replace('/', '-'), config.iter))
                    fn_g = os.path.join(current_path, "logdata", "{}-{}-{}-g.npy".format(
                        config.impl, k.replace('/', '-'), config.iter))
                    np.save(fn_d, v.variable_instance.d)
                    np.save(fn_g, v.variable_instance.g)

                # params = o.solver.get_parameters()
                # params = nn.get_parameters()
                # for k, v, in params.items():
                #     fn_d = os.path.join(current_path, "logdata", "{}-{}-{}.npy".format(
                #         config.impl, k.replace('/', '-'), config.iter))
                #     fn_g = os.path.join(current_path, "logdata", "{}-{}-{}-g.npy".format(
                #         config.impl, k.replace('/', '-'), config.iter))
                #     np.save(fn_d, v.d)
                #     np.save(fn_g, v.g)
                print("iter: {}".format(config.iter))

            if o.weight_decay > 0:
                o.solver.weight_decay(o.weight_decay)

            if o.scheduler is not None:
                o.solver.set_learning_rate(
                    o.scheduler.get_learning_rate(config.iter))
            o.solver.update()
            variables = o.loss_variables
            for l in variables:
                sum_iteration += np.mean(l.variable_instance.d)

    return sum_iteration


def _evaluate(config):
    for name, mon in config.monitors.items():
        m = mon.monitor
        error_sum_monitor = 0.0
        data_size = max([di.size for di in mon.data_iterators])
        batch_size = max([di.batch_size for di in mon.data_iterators])

        if config.save_evaluation_variable:
            for pv in m.net_variables.values():
                if pv not in itertools.chain(m.dataset_assign.keys(), m.generator_assign.keys()):
                    pv.variable_instance.persistent = True

        for i in range(data_size // batch_size):
            data = OrderedDict()
            for di in mon.data_iterators:
                data.update(zip(di.variables, di.next()))

            for v, d in m.dataset_assign.items():
                dest_context = config.global_config.default_context
                let_data_to_variable(v.variable_instance, data[
                                     d], ctx=dest_context,
                                     data_name=d, variable_name=v.name)

            for v, generator in m.generator_assign.items():
                dest_context = config.global_config.default_context
                let_data_to_variable(v.variable_instance,
                                     data=generator(v.variable_instance.d.shape), ctx=dest_context,
                                     variable_name=v.name)

            if config.iter >= config.start_iteration:
                # m.network.forward(m.forward_sequence)
                config.cb.forward(m)

            error_sum = 0.0
            for v in m.monitor_variables:
                error_sum += np.mean(v.variable_instance.d)
                # v.variable_instance.data.zero()

        error_sum_monitor += error_sum
        if config.save_evaluation_variable:
            for k, v in m.net_variables.items():
                fn_d = os.path.join(current_path, "logdata", "{}-{}-{}-{}.npy".format(
                    config.impl, name, k.replace('/', '-'), config.iter))
                fn_g = os.path.join(current_path, "logdata", "{}-{}-{}-g.npy".format(
                    config.impl, name, k.replace('/', '-'), config.iter))
                np.save(fn_d, v.variable_instance.d)
                np.save(fn_g, v.variable_instance.g)
            print("save evaluation, iter: {}, name: {}".format(config.iter, name))

    return error_sum_monitor


def _train(config):
    error = 0.0
    cost = 0.0

    for iteration in range(config.end_iteration):
        config.iter = iteration
        cost = _update(config)
        error = _evaluate(config)
        yield cost, error


def train(info, config):
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

    # Training
    comm = current_communicator()
    config.training_config.iter_per_epoch //= comm.size if comm else 1
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
        yield from _train(config)


# This code comes from train.py, simulate same logic for testing
def _save_parameters(tmp_nnp_file, train_config, nntxt_str):
    base = os.path.dirname(tmp_nnp_file)
    base = os.path.join(base, 'results')

    version_filename = base + '_version.txt'

    with open(version_filename, 'w') as file:
        file.write('{}\n'.format(nnp_version()))

    # This is for testing, start=>
    nntxt_filename = base + '_network.nntxt'
    with open(nntxt_filename, "w") as file:
        file.write(nntxt_str)
    # <= End.

    param_filename = base + '_param.h5'
    nn.parameter.save_parameters(param_filename)

    opti_filenames = save_optimizer_states(
        base, '.h5', train_config)

    with zipfile.ZipFile(tmp_nnp_file, 'w') as nnp:
        nnp.write(version_filename, 'nnp_version.txt')
        nnp.write(nntxt_filename, "network.nntxt")
        nnp.write(param_filename, 'parameter.h5')
        for f in opti_filenames:
            nnp.write(f, f[len(base) + 1:])

    os.unlink(version_filename)
    os.unlink(param_filename)
    for f in opti_filenames:
        os.unlink(f)
    logger.info("{} is saved.".format(tmp_nnp_file))


@pytest.mark.skip(reason="Skipped in auto regression testing, since human check is needed!")
@pytest.mark.parametrize("nntxt_idx", CASE_INDEX)
def test_expander(nntxt_idx):
    with generate_case_from_nntxt_str(NNTXT_EQUIVALENCE_CASES[nntxt_idx], '.protobuf', 128) as nnp_file:
        g = nn.graph_def.load(nnp_file)
        n = g.default_graph.expand_loop_control()
        n.execute_on_proto(Verifier())


@pytest.mark.parametrize("nntxt_idx", CASE_INDEX)
@pytest.mark.parametrize("parameter_format", ['.protobuf', '.h5'])
@pytest.mark.parametrize("dataset_sample_num", [32])
def test_load_and_infer_equivalence(nntxt_idx, parameter_format, dataset_sample_num):
    '''These cases tends to test equivalence before and after
    refactoring NNP load functions. The scope of refactor includes network part and load function.
    This test firstly generated .nnp from nntxt_str, according to specified parameter_format
    and replace dataset's uri with a temporarily generated random dataset, then performs inferring
    operation similar to what is done in cli/forward.py.
    '''
    with generate_case_from_nntxt_str(NNTXT_EQUIVALENCE_CASES[nntxt_idx], parameter_format, dataset_sample_num) as nnp_file:
        ref_info = ref_load(nnp_file)
        ref_result = partial(
            common_forward, forward_func=_ref_forward)(ref_info)

        info = load.load(nnp_file)
        result = partial(common_forward, forward_func=_forward)(info)

    assert_tensor_equal(result, ref_result)


@pytest.mark.parametrize("nntxt_idx", [0])
@pytest.mark.parametrize("parameter_format", ['.protobuf'])
@pytest.mark.parametrize("dataset_sample_num", [32])
def test_load_and_infer_improvement(nntxt_idx, parameter_format, dataset_sample_num):
    '''This case tests improvement features, comparing legacy implementation,
    legacy cannot load or infer successfully, while refactor-ed is OK.
    '''
    with generate_case_from_nntxt_str(NNTXT_IMPROVEMENT_CASES[nntxt_idx], parameter_format, dataset_sample_num) as nnp_file:
        with pytest.raises(ValueError) as excinfo:
            ref_info = ref_load(nnp_file)
            ref_result = partial(
                common_forward, forward_func=_ref_forward)(ref_info)
            print(excinfo)

        info = load.load(nnp_file)
        result = partial(common_forward, forward_func=_forward)(info)

    # Since legacy implementaion cannot handle this case correctly,
    # comparing result is impossible.
    # assert_tensor_equal(result, ref_result)


@pytest.mark.parametrize("nntxt_idx", CASE_INDEX)
@pytest.mark.parametrize("parameter_format", ['.protobuf'])
@pytest.mark.parametrize("dataset_sample_num", [64])
@pytest.mark.parametrize("batch_size", [16])
def test_load_and_train_equivalence(nntxt_idx, parameter_format, dataset_sample_num, batch_size):
    '''These cases tends to test equivalence before and after refactoring.
    The operation is similar to what is done in cli/train.py.
    '''
    # for debugging
    save_v = False
    output_network_topology = False
    verbose = False
    m_iter = 10

    class Callback:
        pass

    legacy_config = TrainConfig()
    legacy_config.on_iter = None
    legacy_config.save_optimizer_variable = False
    legacy_config.save_evaluation_variable = False
    legacy_config.start_iteration = 0
    legacy_config.end_iteration = 10
    legacy_config.enable_save_variable = save_v
    legacy_cb = Callback()
    legacy_cb.forward = lambda o: o.network.forward(o.forward_sequence)
    legacy_cb.backward = lambda o, b: o.network.backward(
        o.backward_sequence, b)
    legacy_config.cb = legacy_cb
    legacy_config.impl = "legacy"

    new_config = TrainConfig()
    new_config.on_iter = None
    new_config.save_optimizer_variable = False
    new_config.save_evaluation_variable = False
    new_config.start_iteration = 0
    new_config.end_iteration = 10
    new_config.enable_save_variable = save_v
    new_cb = Callback()
    new_cb.forward = lambda x: x.target.forward(clear_no_need_grad=True)
    new_cb.backward = lambda x, b: x.target.backward(clear_buffer=True)
    new_config.cb = new_cb
    new_config.impl = "new"

    with generate_case_from_nntxt_str(NNTXT_EQUIVALENCE_CASES[nntxt_idx], parameter_format, dataset_sample_num, batch_size) as nnp_file:
        ref_result = []
        result = []
        nn.clear_parameters()
        info = ref_load(nnp_file, batch_size=batch_size)
        for cost, error in partial(train, config=legacy_config)(info):
            ref_result.append((cost, error))

        nn.clear_parameters()
        info = load.load(nnp_file, batch_size=batch_size)

        if output_network_topology:
            for n, opt in info.optimizers.items():
                print(n)
                opt.network.execute_on_proto(Verifier())

        for cost, error in partial(train, config=new_config)(info):
            result.append((cost, error))

        for i, ((cost_ref, error_ref), (cost, error)) in enumerate(zip(ref_result, result)):
            if verbose:
                print("{}: cost: {} <--> {}".format(i, cost_ref, cost))
                print("{}: error: {} <--> {}".format(i, error_ref, error))
            assert_allclose(np.array([cost_ref, error_ref]), np.array([cost, error]), rtol=1e-2, atol=1e-3,
                            err_msg="Error: {}".format(nntxt_idx))


def compare_info(ref_info, info):
    pass


@pytest.mark.parametrize("nntxt_idx", CASE_INDEX)
@pytest.mark.parametrize("parameter_format", ['.protobuf'])
@pytest.mark.parametrize("dataset_sample_num", [64])
@pytest.mark.parametrize("batch_size", [16])
@pytest.mark.parametrize("include_params", [False])
@pytest.mark.parametrize("variable_batch_size", [True])
def test_load_and_save_equivalence(nntxt_idx, parameter_format, dataset_sample_num, batch_size,
                                   include_params, variable_batch_size):
    '''These cases tends to test equivalence before and after refactoring.
    '''
    verbose = True
    a_few_iter = 10
    half_iter = 5
    output_network_topology = True
    with generate_case_from_nntxt_str(NNTXT_EQUIVALENCE_CASES[nntxt_idx], parameter_format, dataset_sample_num, batch_size) as nnp_file:
        with create_temp_with_dir("saved.nnp") as saved_nnp_file:
            class Callback:
                pass

            class ModelSaver:
                def __init__(self, info):
                    self.info = info

                def __call__(self, config):
                    if config.iter != half_iter:
                        return

                    info = self.info
                    datasets = []
                    with ExitStack() as stack:
                        for d_name, d in info.datasets.items():
                            ds = {}
                            ds['name'] = d_name
                            ds['uri'] = d.uri
                            ds['cache_dir'] = d.cache_dir
                            di_instance = stack.enter_context(
                                d.data_iterator())
                            ds['variables'] = [
                                var_name for var_name in di_instance.variables]
                            ds['batch_size'] = di_instance.batch_size
                            ds['no_image_normalization'] = not d.normalize
                            ds['shuffle'] = di_instance._shuffle
                            datasets.append(ds)

                    dataset_assign = set()
                    for obj in itertools.chain(info.monitors.values(), info.executors.values(), info.optimizers.values()):
                        for pv in obj.dataset_assign.keys():
                            dataset_assign.add(pv.name)

                    contents = {
                        'global_config': {'default_context': info.global_config.default_context},
                        'training_config':
                        {'max_epoch': info.training_config.max_epoch,
                         'iter_per_epoch': info.training_config.iter_per_epoch,
                         'save_best': info.training_config.save_best},
                        'networks': [
                            {'name': n_name,
                             'batch_size': n.batch_size,
                             'outputs': {out: n.variables[out].variable_instance for out in n.outputs},
                             'names': {inp: n.variables[inp].variable_instance
                                       for inp in itertools.chain(n.inputs, n.outputs)}}
                            for n_name, n in info.networks.items()],
                        'executors': [
                            {'name': e_name,
                             'network': e.network.name,
                             'data': [pv.name for pv in e.dataset_assign.keys()],
                             'generator_variables': [pv.name for pv in e.generator_assign.keys()],
                             'output': [pv.name for pv in e.output_assign.keys()]}
                            for e_name, e in info.executors.items()],
                        'optimizers': [
                            {'name': o_name,
                             'solver': o.solver,
                             'network': o.network.name,
                             'data_variables': {pv.name: d for pv, d in o.dataset_assign.items()},
                             'generator_variables': [pv.name for pv in o.generator_assign.keys()],
                             'loss_variables': [pv.name for pv in o.loss_variables],
                             'dataset': [ds_name for ds_name in o.data_iterators.keys()],
                             'weight_decay': o.weight_decay,
                             'lr_decay': o.lr_decay,
                             'lr_decay_interval': o.lr_decay_interval,
                             'update_interval': o.update_interval}
                            for o_name, o in info.optimizers.items()],
                        'datasets': datasets,
                        'monitors': [
                            {'name': m_name,
                             'network': m.network.name,
                             'data_variables': {pv.name: d for pv, d in m.dataset_assign.items()},
                             'generator_variables': [pv.name for pv in m.generator_assign.keys()],
                             'monitor_variables': [pv.name for pv in m.monitor_variables],
                             'dataset': [ds_name for ds_name in m.data_iterators.keys()]}
                            for m_name, m in info.monitors.items()],
                    }

                    save.save(saved_nnp_file, contents,
                              include_params, variable_batch_size)

            new_config = TrainConfig()
            new_config.start_iteration = 0
            new_config.end_iteration = a_few_iter
            new_config.save_optimizer_variable = False
            new_config.save_evaluation_variable = False
            new_cb = Callback()
            new_cb.forward = lambda x: x.target.forward(
                clear_no_need_grad=True)
            new_cb.backward = lambda x, b: x.target.backward(clear_buffer=True)
            new_config.cb = new_cb
            new_config.impl = "ref"

            ref_result = []
            ref_info = load.load(nnp_file, batch_size=batch_size)

            if output_network_topology:
                for n, opt in ref_info.optimizers.items():
                    print(n)
                    opt.network.execute_on_proto(Verifier())

            new_config.on_iter = ModelSaver(ref_info)
            for cost, error in partial(train, config=new_config)(ref_info):
                ref_result.append((cost, error))

            new_config.on_iter = None
            new_config.start_iteration = half_iter
            new_config.end_iteration = a_few_iter
            new_config.impl = "new"
            result = []
            nn.clear_parameters()
            info = load.load(saved_nnp_file, batch_size=batch_size)

            if output_network_topology:
                for n, opt in info.optimizers.items():
                    print(n)
                    opt.network.execute_on_proto(Verifier())

            for cost, error in partial(train, config=new_config)(info):
                result.append((cost, error))

            compare_info(ref_info, info)

            for i, ((cost_ref, error_ref), (cost, error)) in enumerate(zip(ref_result, result)):
                if verbose:
                    print("{}: cost: {} <--> {}".format(i, cost_ref, cost))
                    print("{}: error: {} <--> {}".format(i, error_ref, error))
                if i > new_config.start_iteration:
                    assert_allclose(np.array([cost_ref, error_ref]), np.array([cost, error]), rtol=1e-2, atol=1e-5,
                                    err_msg="Error: {}".format(nntxt_idx))


@pytest.mark.parametrize("nntxt_idx", CASE_INDEX)
@pytest.mark.parametrize("parameter_format", ['.protobuf'])
@pytest.mark.parametrize("dataset_sample_num", [64])
@pytest.mark.parametrize("batch_size", [16])
def test_resume_suspend_equivalence(nntxt_idx, parameter_format, dataset_sample_num, batch_size):
    '''These cases tends to test equivalence before and after refactoring.
    '''
    verbose = True
    a_few_iter = 10
    half_iter = 5
    output_network_topology = False
    with generate_case_from_nntxt_str(NNTXT_EQUIVALENCE_CASES[nntxt_idx], parameter_format, dataset_sample_num, batch_size) as nnp_file:
        with create_temp_with_dir("saved_parameter.nnp") as saved_parameter_nnp:
            class Callback:
                pass

            class ModelSaver:
                def __init__(self, info):
                    self.info = info

                def __call__(self, config):
                    if config.iter != half_iter:
                        return
                    _save_parameters(saved_parameter_nnp, config,
                                     NNTXT_EQUIVALENCE_CASES[nntxt_idx])

            new_config = TrainConfig()
            new_config.start_iteration = 0
            new_config.end_iteration = a_few_iter
            new_config.save_optimizer_variable = False
            new_config.save_evaluation_variable = False
            new_cb = Callback()
            new_cb.forward = lambda x: x.target.forward(
                clear_no_need_grad=True)
            new_cb.backward = lambda x, b: x.target.backward()
            new_config.cb = new_cb
            new_config.impl = "ref"

            ref_result = []
            ref_info = load.load(nnp_file, batch_size=batch_size)
            print("load.load")

            if output_network_topology:
                for n, opt in ref_info.optimizers.items():
                    print(n)
                    opt.network.execute_on_proto(Verifier())

            new_config.on_iter = ModelSaver(ref_info)
            for cost, error in partial(train, config=new_config)(ref_info):
                ref_result.append((cost, error))

            new_config.on_iter = None
            new_config.start_iteration = half_iter
            new_config.end_iteration = a_few_iter
            new_config.impl = "new"
            result = []
            nn.clear_parameters()
            info = load.load(nnp_file, batch_size=batch_size,
                             exclude_parameter=True)
            print("load.load")

            # Here, `info` is different `config`, but optimizer is same.
            load_train_state(saved_parameter_nnp, info)

            for cost, error in partial(train, config=new_config)(info):
                result.append((cost, error))

            compare_info(ref_info, info)

            for i, ((cost_ref, error_ref), (cost, error)) in enumerate(zip(ref_result, result)):
                if verbose:
                    print("{}: cost: {} <--> {}".format(i, cost_ref, cost))
                    print("{}: error: {} <--> {}".format(i, error_ref, error))
                if i > new_config.start_iteration:
                    assert_allclose(np.array([cost_ref, error_ref]), np.array([cost, error]), rtol=1e-2, atol=1e-5,
                                    err_msg="Error: {}".format(nntxt_idx))
