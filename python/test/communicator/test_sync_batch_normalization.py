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
import numpy as np
import nnabla as nn
import nnabla.functions as F
from nbla_test_utils import list_context
from nnabla.testing import assert_allclose

ctxs = list_context('SyncBatchNormalization')

from six.moves import range


def create_inputs(rng, axis, device_id):
    x = rng.randn(2, 3, 4).astype(np.float32) * 2
    x += device_id
    shape_stat = [1 for _ in range(x.ndim)]
    shape_stat[axis] = x.shape[axis]
    beta = rng.randn(*shape_stat).astype(np.float32)
    gamma = rng.randn(*shape_stat).astype(np.float32)
    rmean = np.zeros(shape_stat, dtype=np.float32)
    rvar = np.zeros(shape_stat, dtype=np.float32)
    return x, beta, gamma, rmean, rvar


def mask_inputs(inputs, no_scale, no_bias, no_mean, no_variance):
    if no_bias:
        inputs[1] = np.zeros(inputs[1].shape)

    if no_scale:
        inputs[2] = np.ones(inputs[2].shape)

    if no_mean:
        inputs[3] = np.zeros(inputs[3].shape)

    if no_variance:
        inputs[4] = np.ones(inputs[4].shape)

    return inputs


def mask_vinputs(vinputs, no_scale, no_bias, no_mean, no_variance):
    if no_bias:
        vinputs[1] = None

    if no_scale:
        vinputs[2] = None

    if no_mean:
        vinputs[3] = None

    if no_variance:
        vinputs[4] = None

    return vinputs


@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("axis", [1, 2])
@pytest.mark.parametrize("decay_rate", [0.9])
@pytest.mark.parametrize("eps", [1e-5])
@pytest.mark.parametrize("output_stat, batch_stat", [[False, False], [False, True], [True, True]])
@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("no_scale, no_bias", [[False, False], [True, True]])
@pytest.mark.parametrize("no_mean", [True, False])
@pytest.mark.parametrize("no_variance", [True, False])
def test_sync_batch_normalization_forward_backward(seed, axis, decay_rate, eps, batch_stat,
                                                   output_stat, ctx, func_name, comm_nccl_opts,
                                                   no_scale, no_bias, no_mean, no_variance):
    if comm_nccl_opts is None:
        pytest.skip(
            "Communicator test is disabled. You can turn it on by an option `--test-communicator`.")

    comm = comm_nccl_opts.comm
    device_id = int(comm_nccl_opts.device_id)
    n_devices = len(comm_nccl_opts.devices)
    ctx.device_id = comm_nccl_opts.device_id

    def ref_batch_normalization(x, beta, gamma, rmean, rvar, comm, axes, decay_rate,
                                eps, batch_stat, output_stat):

        orig = x - device_id
        inputs = []
        for i in range(n_devices):
            inputs.append(orig + i)
        x = np.concatenate(inputs)

        vx = nn.Variable(x.shape, True)
        vx.d = x
        vbeta = nn.Variable(beta.shape, True)
        vbeta.d = beta
        vgamma = nn.Variable(gamma.shape, True)
        vgamma.d = gamma
        vrmean = nn.Variable(rmean.shape, True)
        vrmean.d = rmean
        vrvar = nn.Variable(rvar.shape, True)
        vrvar.d = rvar
        with nn.context_scope(ctx):
            out = F.batch_normalization(vx, vbeta, vgamma, vrmean, vrvar,
                                        batch_stat=batch_stat, output_stat=output_stat,
                                        axes=axes, decay_rate=decay_rate, eps=eps)
        if output_stat:
            out[0].forward()
            rmean[...] = vrmean.d.copy()
            rvar[...] = vrvar.d.copy()
            return out[0].d[device_id*2:(device_id+1)*2], out[1].d, out[2].d
        out.forward()
        rmean[...] = vrmean.d.copy()
        rvar[...] = vrvar.d.copy()
        return out.d[device_id*2:(device_id+1)*2]

    def ref_batch_normalize_grad(x, beta, gamma, rmean, rvar,
                                 dy,
                                 comm, axes, decay_rate,
                                 eps, batch_stat, output_stat, **kw):
        orig = x - device_id
        inputs = []
        for i in range(n_devices):
            inputs.append(orig + i)
        x = np.concatenate(inputs)

        vx = nn.Variable(x.shape, True)
        vx.d = x
        vx.g = 0
        vbeta = nn.Variable(beta.shape, True)
        vbeta.d = beta
        vbeta.g = 0
        vgamma = nn.Variable(gamma.shape, True)
        vgamma.d = gamma
        vgamma.g = 0
        vrmean = nn.Variable(rmean.shape, True)
        vrmean.d = rmean
        vrvar = nn.Variable(rvar.shape, True)
        vrvar.d = rvar
        with nn.context_scope(ctx):
            out = F.batch_normalization(vx, vbeta, vgamma, vrmean, vrvar,
                                        batch_stat=batch_stat, output_stat=output_stat, axes=axes, decay_rate=decay_rate, eps=eps)
        f = out.parent
        f.forward([vx, vbeta, vgamma, vrmean, vrvar], [out])
        for i in range(n_devices):
            out.g[2*i:2*(i+1)] = dy
        f.backward([vx, vbeta, vgamma, vrmean, vrvar], [out])

        return np.concatenate([vx.g[device_id*2:(device_id+1)*2].flatten(), vbeta.g.flatten(), vgamma.g.flatten()])

    def ref_batch_normalize_grad_with_output_stat(x, beta, gamma, rmean, rvar,
                                                  dy, dmean, dvar,
                                                  comm, axes, decay_rate,
                                                  eps, batch_stat, output_stat, **kw):
        orig = x - device_id
        inputs = []
        for i in range(n_devices):
            inputs.append(orig + i)
        x = np.concatenate(inputs)

        vx = nn.Variable(x.shape, True)
        vx.d = x
        vx.g = 0
        vbeta = nn.Variable(beta.shape, True)
        vbeta.d = beta
        vbeta.g = 0
        vgamma = nn.Variable(gamma.shape, True)
        vgamma.d = gamma
        vgamma.g = 0
        vrmean = nn.Variable(rmean.shape, True)
        vrmean.d = rmean
        vrvar = nn.Variable(rvar.shape, True)
        vrvar.d = rvar
        with nn.context_scope(ctx):
            out = F.batch_normalization(vx, vbeta, vgamma, vrmean, vrvar,
                                        batch_stat=batch_stat, output_stat=output_stat, axes=axes, decay_rate=decay_rate, eps=eps)
        f = out[0].parent
        f.forward([vx, vbeta, vgamma, vrmean, vrvar], out)
        for i in range(n_devices):
            out[0].g[2*i:2*(i+1)] = dy
        out[1].g[...] = dmean
        out[2].g[...] = dvar
        f.backward([vx, vbeta, vgamma, vrmean, vrvar], out)
        return np.concatenate([vx.g[device_id*2:(device_id+1)*2].flatten(), vbeta.g.flatten(), vgamma.g.flatten()])
    from nbla_test_utils import function_tester
    rng = np.random.RandomState(seed)
    inputs = list(create_inputs(rng, axis, device_id))
    inputs = mask_inputs(inputs, no_scale, no_bias, no_mean, no_variance)

    if output_stat:
        ref_grad = ref_batch_normalize_grad_with_output_stat
    else:
        ref_grad = ref_batch_normalize_grad
    axes = [axis]

    # TODO: enable test using function_tester when the backward is supported with batch_stat=True
    if batch_stat is True:
        function_tester(rng, F.sync_batch_normalization, ref_batch_normalization,
                        inputs,
                        ref_grad=ref_grad,
                        func_args=[comm],
                        func_kwargs=dict(
                            axes=axes,
                            decay_rate=decay_rate,
                            eps=eps,
                            batch_stat=batch_stat,
                            output_stat=output_stat),
                        backward=[True, True, True, False, False],
                        ctx=ctx, func_name=func_name, dstep=1e-2, atol_b=1e-2)
    else:
        if no_mean or no_variance:
            return

        # Forward test when batch_stat is False
        vinputs = []
        for i in inputs:
            vinputs.append(nn.Variable(i.shape, True))
            vinputs[-1].d = i

        vinputs = mask_vinputs(
            vinputs, no_scale, no_bias, no_mean, no_variance)

        for i in range(5):
            inputs[0] = rng.randn(*inputs[0].shape) + device_id
            vinputs[0].d[...] = inputs[0]
            ref_y = ref_batch_normalization(
                *(inputs + [comm, axes, decay_rate, eps, batch_stat, output_stat]))
            with nn.context_scope(ctx), nn.auto_forward():
                y = F.sync_batch_normalization(
                    *(vinputs + [comm, "world", axes, decay_rate, eps, batch_stat, output_stat]))
            assert_allclose(vinputs[0].d, inputs[0])

    # Check if running mean and var works.
    if no_mean and no_variance:
        return

    vinputs = []
    for i in inputs:
        vinputs.append(nn.Variable(i.shape, True))
        vinputs[-1].d = i

    vinputs = mask_vinputs(vinputs, no_scale, no_bias, no_mean, no_variance)

    for i in range(5):
        inputs[0] = rng.randn(*inputs[0].shape) + device_id
        vinputs[0].d[...] = inputs[0]
        ref_y = ref_batch_normalization(
            *(inputs + [comm, axes, decay_rate, eps, batch_stat, output_stat]))
        with nn.context_scope(ctx), nn.auto_forward():
            y = F.sync_batch_normalization(
                *(vinputs + [comm, "world", axes, decay_rate, eps, batch_stat, output_stat]))
        if not no_mean:
            assert_allclose(vinputs[3].d, inputs[3])

        if not no_variance:
            assert_allclose(vinputs[4].d, inputs[4], atol=1e-3)
