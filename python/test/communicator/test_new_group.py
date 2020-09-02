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
import nnabla.parametric_functions as PF
import numpy as np


@pytest.mark.skip("Skip until nccl is fixed.")
@pytest.mark.parametrize("seed", [313])
def test_new_group(seed, comm_nccl_opts):
    if comm_nccl_opts is None:
        pytest.skip(
            "Communicator test is disabled. You can turn it on by an option `--test-communicator`.")
    try:
        import nnabla_ext
        import nnabla_ext.cuda
    except:
        pytest.skip("{} is supported in CUDA device".format(
            sys._getframe().f_code.co_name))

    n_devices = comm_nccl_opts.comm.size

    # Reference
    rng = np.random.RandomState(seed)
    device = rng.choice(n_devices)
    groups = {
        "group_i": [device],
        "all": np.arange(n_devices).tolist(),
        "world": np.arange(n_devices).tolist()
    }

    # comm_nccl_opts.comm.new_group
    group_i = comm_nccl_opts.comm.new_group(("group_i", [device]))
    group_all = comm_nccl_opts.comm.new_group(("all", np.arange(n_devices)))

    # comm_nccl_opts.comm.list_groups
    assert comm_nccl_opts.comm.list_groups() == groups

    # comm_nccl_opts.comm.find_group
    assert comm_nccl_opts.comm.find_group("group_i") == groups["group_i"]
    assert comm_nccl_opts.comm.find_group("group_k") == []
