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


def pytest_addoption(parser):
    parser.addoption('--test-communicator', action='store_true',
                     default=False, help='Enable NCCL communicator test.')
    parser.addoption('--communicator-gpus', type=str, default=None,
                     help='Comma separated device IDs. e.g --communicator-gpus=0,2.')
    parser.addoption('--type-config', type=str, default='float', action='store',
                     help='Type of computation. e.g. "float", "half"., --type-config=float')


@pytest.fixture(scope='session')
def comm_nccl_opts(request):
    """Common resources for communicator tests.
    """
    if not request.config.getoption('--test-communicator'):
        return None

    import nnabla.communicators as C
    from nnabla.ext_utils import get_extension_context

    try:
        from nnabla_ext import cuda
    except Exception as e:
        raise ImportError(
            "Communicator test requires CUDA extension.\n{}".format(e))

    gpus = request.config.getoption('--communicator-gpus')
    n_devices = cuda.get_device_count()
    if gpus is None:
        devices = list(map(str, range(n_devices)))
    else:
        devices = gpus.split(',')
        # Check numbers
        try:
            for d in devices:
                gid = int(d)
                if gid >= n_devices:
                    raise ValueError('')
        except ValueError as e:
            raise ValueError(
                "GPU IDs must be comma separated integers of available GPUs. Given {}. Available GPUs are {}.".format(gpus, n_devices))

    extension_module = "cuda"
    type_config = request.config.getoption('--type-config')
    ctx = get_extension_context(extension_module, type_config=type_config)
    try:
        comm = C.MultiProcessCommunicator(ctx)
    except Exception as e:
        raise RuntimeError(
            "Communicator could not be created. You may haven't build with distributed support.\n{}".format(e))
    try:
        comm.init()
    except Exception as e:
        raise RuntimeError(
            "Communicator initialization failed. (Maybe MPI init failure.)\n{}".format(e))

    assert len(
        devices) == comm.size, "Number of cuda devices used are not same as that of processes."
    n_devices = comm.size
    mpi_rank = comm.rank
    mpi_local_rank = comm.local_rank
    ctx.device_id = devices[mpi_local_rank]

    class CommOpts:
        pass

    c = CommOpts()
    c.comm = comm
    c.ctx = ctx
    c.device_id = ctx.device_id
    c.devices = devices
    c.mpi_rank = mpi_rank
    c.mpi_local_rank = mpi_local_rank
    return c
