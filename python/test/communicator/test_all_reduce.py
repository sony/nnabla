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

# Contex
extension_module = "cuda"
ctx = extension_context(extension_module)

# Communicator
try: 
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
@pytest.mark.parametrize("division", [False])
def test_all_reduce(seed, inplace, division):
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

    # AllReduce
    comm.all_reduce([x.data for x in x_list],
                    division=division, inplace=inplace)

    # Ref AllReduce
    refs = ref_all_reduce(x_data_list, n_devices, division)

    # Check
    for x, ref in zip(x_list, refs):
        assert np.allclose(x.d, ref)
