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


def ref_all_gather(x_data, n_devices):
    results = []
    for i in range(n_devices):
        results.append(x_data * i)
    return results


@pytest.mark.skipif(comm == None, reason="Communicator does not exist.")
@pytest.mark.parametrize("seed", [313])
def test_all_gather(seed):
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
        assert np.allclose(y.d, ref)
