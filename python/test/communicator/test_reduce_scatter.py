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


def ref_reduce_scatter(x_data_list, size, division):
    f = reduce(lambda x, y: x + y, np.arange(size)) + size
    results = []
    for x_data in x_data_list:
        result = x_data * f
        if division:
            result /= size
        results.append(result)
    return results


@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("division", [True, False])
def test_reduce_scatter(seed, division, comm_nccl_opts):
    if comm_nccl_opts is None:
        pytest.skip(
            "Communicator test is disabled. You can turn it on by an option `--test-communicator`.")
    if len(comm_nccl_opts.devices) < 2:
        pytest.skip(
            "Communicator test is disabled. Use more than 1 gpus.")
    comm = comm_nccl_opts.comm
    device_id = int(comm_nccl_opts.device_id)
    devices = comm_nccl_opts.devices
    n_devices = len(comm_nccl_opts.devices)

    # Variables
    rng = np.random.RandomState(seed)
    x_list = []
    x_data_list = []
    for i in devices:
        x_data = rng.rand(3, 4)
        x_data_list.append(x_data)
        x = nn.Variable(x_data.shape)
        x.d = x_data * (device_id + 1)
        x_list.append(x)

    # Reduce Scatter
    x = nn.Variable((3, 4))
    comm.reduce_scatter(
        [x.data for x in x_list], x.data, division=division)

    # Ref
    refs = ref_reduce_scatter(x_data_list, n_devices, division)

    # Check
    assert_allclose(x.d, refs[device_id], rtol=1e-3, atol=1e-6)
