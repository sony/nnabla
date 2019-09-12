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


def test_data_parallel_communicator(comm_nccl_opts):
    pytest.skip("Always skip. Deprecated.")
    if comm_nccl_opts is None:
        pytest.skip(
            "Communicator test is disabled. You can turn it on by an option `--test-communicator`.")
    from nnabla.ext_utils import get_extension_context
    comm = comm_nccl_opts.comm
    device_id = int(comm_nccl_opts.device_id)
    devices = comm_nccl_opts.devices
    n_devices = len(comm_nccl_opts.devices)

    # Contexts and Computation Graph
    extension_module = "cuda"
    ctxs = {}
    for d in devices:
        ctx = get_extension_context(extension_module,
                                    device_id="{}".format(d))
        ctxs[d] = ctx
        with nn.context_scope(ctx):
            x_data = np.random.rand(4, 5)
            x = nn.Variable(x_data.shape)
            with nn.parameter_scope("gpu{}".format(d)):
                with nn.parameter_scope("affine1"):
                    z = PF.affine(x, 10, with_bias=True)
                with nn.parameter_scope("affine2"):
                    y = PF.affine(z, 5)

    # Init w.g
    grads = {}
    for d in devices:
        with nn.parameter_scope("gpu{}".format(d)):
            params = nn.get_parameters()
            grad = []
            for i, elm in enumerate(params.items()):
                k, v = elm
                grad_ = np.random.randn(*v.shape)
                v.g = grad_
                v.grad.cast(np.float32, ctxs[d])
                grad.append(grad_)
            grads[d] = grad

    # Reference
    ref_grads = []
    with nn.parameter_scope("gpu{}".format(d)):
        params = nn.get_parameters()
        for i in range(len(params)):
            ave_grad = 0
            for d in devices:
                ave_grad += grads[d][i]
            ave_grad /= n_devices
            ref_grads.append(ave_grad)

    comm.clear_context_parameters()
    for d in devices:
        with nn.parameter_scope("gpu{}".format(d)):
            comm.add_context_and_parameters(
                (ctxs[d], nn.get_parameters()))
    comm.allreduce(division=True)

    # Check
    atol = 1e-6
    for d in devices:
        with nn.parameter_scope("gpu{}".format(d)):
            params = nn.get_parameters()
            for i, elm in enumerate(params.items()):
                k, v = elm
                assert_allclose(ref_grads[i], v.g, atol=atol)
