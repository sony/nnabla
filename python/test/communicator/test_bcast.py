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


def ref_bcast(x_data_list, src):
    results = []
    for x_data in x_data_list:
        results.append(x_data * (src + 1))
    return results


@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("src", [0, 1])
@pytest.mark.parametrize("inplace", [True, False])
def test_bcast(seed, src, inplace, comm_nccl_opts):
    if comm_nccl_opts is None:
        pytest.skip(
            "Communicator test is disabled. You can turn it on by an option `--test-communicator`.")
    if len(comm_nccl_opts.devices) < 2:
        pytest.skip(
            "Communicator test is disabled. Use more than 1 gpus.")

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

    # Bcast
    comm.bcast([x.data for x in x_list], src, inplace=inplace)

    # Ref
    refs = ref_bcast(x_data_list, src)

    # Check
    for x, ref in zip(x_list, refs):
        assert_allclose(x.d, ref, rtol=1e-3, atol=1e-6)
