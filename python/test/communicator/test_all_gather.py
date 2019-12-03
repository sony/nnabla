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


def ref_all_gather(x_data, n_devices):
    results = []
    for i in range(n_devices):
        results.append(x_data * i)
    return results


@pytest.mark.parametrize("seed", [313])
def test_all_gather(seed, comm_nccl_opts):
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
    rng = np.random.RandomState(seed)
    x_data = rng.rand(3, 4)
    x = nn.Variable(x_data.shape)
    x.d = x_data * device_id
    y_list = []
    for i in range(n_devices):
        y = nn.Variable(x_data.shape)
        y_list.append(y)

    # AllGahter
    comm.all_gather(x.data, [y.data for y in y_list])

    # Ref
    refs = ref_all_gather(x_data, n_devices)

    # Check
    for y, ref in zip(y_list, refs):
        assert_allclose(y.d, ref, rtol=1e-3, atol=1e-6)
