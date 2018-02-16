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
import nnabla.communicators as C
import numpy as np
from nbla_test_utils import list_context
from nnabla.contrib.context import extension_context

############################################
# Communicator has to be instantiated here,
# otherwise, mpirun fails.
############################################

# Communicator
comm = None
try:
    extension_module = "cuda"
    ctx = extension_context(extension_module)
    comm = C.MultiProcessDataParalellCommunicator(ctx)
    comm.init()
    n_devices = comm.size
    mpi_rank = comm.rank
    mpi_local_rank = comm.local_rank
    device_id = mpi_local_rank
    ctx.device_id = str(device_id)
except:
    pass

############################################


def ref_bcast(x_data_list, src):
    results = []
    for x_data in x_data_list:
        results.append(x_data * (src + 1))
    return results


@pytest.mark.skipif(comm == None, reason="Communicator does not exist.")
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("src", [0, 1])
@pytest.mark.parametrize("inplace", [True, False])
def test_bcast(seed, src, inplace):
    try:
        import nnabla_ext
        import nnabla_ext.cuda
    except:
        pytest.skip("{} is supported in CUDA device".format(
            sys._getframe().f_code.co_name))

    n_devices = nnabla_ext.cuda.init.get_device_count()
    if n_devices < 2:
        pytest.skip("Number of cuda devices in this machine is less than 2.")

    if n_devices != comm.size:
        pytest.skip("Number of cuda devices is not same as that of processes.")

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
        assert np.allclose(x.d, ref)
