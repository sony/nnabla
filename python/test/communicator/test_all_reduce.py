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

import pytest
import nnabla as nn
import nnabla.parametric_functions as PF
import numpy as np
from nnabla.testing import assert_allclose

from six.moves import reduce


def check_comm_nccl_opts(comm_nccl_opts):
    if comm_nccl_opts is None:
        pytest.skip(
            "Communicator test is disabled. You can turn it on by an option `--test-communicator`.")
    if len(comm_nccl_opts.devices) < 2:
        pytest.skip(
            "Communicator test is disabled. Use more than 1 gpus.")


def ref_all_reduce(x_data_list, size, division):
    f = reduce(lambda x, y: x + y, np.arange(size)) + size
    results = []
    for x_data in x_data_list:
        result = x_data * f
        if division:
            result /= size
        results.append(result)
    return results


@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("division", [True, False])
def test_all_reduce(seed, inplace, division, comm_nccl_opts):
    check_comm_nccl_opts(comm_nccl_opts)

    comm = comm_nccl_opts.comm
    device_id = int(comm_nccl_opts.device_id)
    n_devices = len(comm_nccl_opts.devices)

    # Variables
    x_list = []
    x_data_list = []
    num_layers = 20
    rng = np.random.RandomState(seed)
    for l in range(num_layers):
        x_data = rng.rand(3, 4)
        x_data_list.append(x_data)
        x = nn.Variable(x_data.shape)
        x.d = x_data * (device_id + 1)
        x_list.append(x)

    # AllReduce
    comm.all_reduce([x.data for x in x_list],
                    division=division, inplace=inplace)

    # Ref AllReduce
    refs = ref_all_reduce(x_data_list, n_devices, division)

    # Check
    for x, ref in zip(x_list, refs):
        assert_allclose(x.d, ref, rtol=1e-3, atol=1e-6)


@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("division", [True, False])
def test_all_reduce_skip_by_zero(seed, inplace, division, comm_nccl_opts):
    '''
    Checking the behavior that all_reduce is skipped if NdArray is set as zeroing
    by NdArray.zero().
    '''
    check_comm_nccl_opts(comm_nccl_opts)

    comm = comm_nccl_opts.comm
    device_id = int(comm_nccl_opts.device_id)
    n_devices = len(comm_nccl_opts.devices)

    xs = [nn.Variable((2, 3, 4), need_grad=True),
          nn.Variable((2, 3), need_grad=True)]

    # Fill data as 1
    for x in xs:
        x.data.fill(1)

    def get_grads(aa):
        return [a.grad for a in aa]

    def zero_grads(aa):
        for a in aa:
            a.grad.zero()

    # A. Allreduce is not performed as all arrays are not updated.
    zero_grads(xs)
    comm.all_reduce(get_grads(xs), division=division, inplace=inplace)
    for g in get_grads(xs):
        assert g.zeroing

    # B. All reduce is performed as any of arrays is updated.
    zero_grads(xs)
    # modify the grad values in rank 0
    if comm.rank == 0:
        for g in get_grads(xs):
            g.data = 1
    comm.all_reduce(get_grads(xs), division=division, inplace=inplace)
    for g in get_grads(xs):
        assert not g.zeroing

    # Construct a graph for allreduce during backward
    import nnabla.functions as F
    y = sum([F.sum(F.relu(x)) for x in xs])

    def execute_allreduce_during_backward(performed):
        y.forward(clear_no_need_grad=True)
        comm_callback = comm.all_reduce_callback(
            get_grads(xs), 1024 * 1024 * 2, division=division)
        zero_grads(xs)
        y.backward(
            None, clear_buffer=True,
            communicator_callbacks=comm_callback
        )
        for g in get_grads(xs):
            assert g.zeroing != performed

    # C-1. performing allreduce during backward
    execute_allreduce_during_backward(True)

    # C-2. not performing allreduce during backward
    for x in xs:
        x.need_grad = False
    execute_allreduce_during_backward(False)

    # C-3. performing allreduce during backward
    # NOTE: It's not supported because callbacks over devices are not
    # consistently callled due to skipping backward operation on
    # variables not requiring gradients.
    # if comm.rank == 0:
    #     for x in xs:
    #         x.need_grad = True
    # execute_allreduce_during_backward(True)
