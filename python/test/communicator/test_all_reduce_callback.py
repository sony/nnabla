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
import nnabla.functions as F
import nnabla.parametric_functions as PF
import numpy as np
from nbla_test_utils import list_context
from nnabla.testing import assert_allclose


@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("pack_size", [2, 8])
@pytest.mark.parametrize("division", [False, True])
def test_all_reduce_callback(seed, pack_size, division, comm_nccl_opts):
    if comm_nccl_opts is None:
        pytest.skip(
            "Communicator test is disabled. You can turn it on by an option `--test-communicator`.")
    if len(comm_nccl_opts.devices) < 2:
        pytest.skip(
            "Communicator test is disabled. Use more than 1 gpus.")

    comm = comm_nccl_opts.comm
    device_id = int(comm_nccl_opts.device_id)
    nn.set_default_context(comm_nccl_opts.ctx)

    nn.clear_parameters()
    x_data_list = []
    num_layers = 20
    rng = np.random.RandomState(seed)
    for l in range(num_layers):
        x_data = rng.rand(3, 4)
        x_data_list.append(x_data)

    # all_reduce_callback
    x_list1 = []
    n1 = nn.Variable([3, 4])
    n1.d = 0
    for l in range(num_layers):
        x = nn.Variable([3, 4], need_grad=True)
        n1 = F.add2(n1, x)
        x.d = x_data_list[l] * (device_id + 1)
        x.g = 0
        x_list1.append(x)
    n1.backward(clear_buffer=True,
                communicator_callbacks=comm.all_reduce_callback([v.grad for v in x_list1], pack_size, division))

    # Ref AllReduce
    x_list2 = []
    n2 = nn.Variable([3, 4])
    n2.d = 0
    for l in range(num_layers):
        x = nn.Variable([3, 4], need_grad=True)
        n2 = F.add2(n2, x)
        x.d = x_data_list[l] * (device_id + 1)
        x.g = 0
        x_list2.append(x)
    n2.backward(clear_buffer=True)
    comm.all_reduce([v.grad for v in x_list2],
                    inplace=False, division=division)

    # Check
    for x, ref in zip(x_list1, x_list2):
        assert_allclose(x.g, ref.g)
