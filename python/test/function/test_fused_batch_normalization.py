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

from six.moves import range

import pytest
import numpy as np
import nnabla as nn
import nnabla.functions as F
from nbla_test_utils import list_context
from nnabla.testing import assert_allclose

ctxs = list_context('FusedBatchNormalization')
cpu_context = nn.Context(["cpu:float"])


def ref_fused_batch_normalization(x, beta, gamma, rmean, rvar, z, axes, decay_rate,
                                  eps, batch_stat, nonlinearity, output_stat):
    with nn.context_scope(cpu_context):
        xvar = nn.Variable.from_numpy_array(x)
        betavar = nn.Variable.from_numpy_array(beta)
        gammavar = nn.Variable.from_numpy_array(gamma)
        rmeanvar = nn.Variable.from_numpy_array(rmean)
        rvarvar = nn.Variable.from_numpy_array(rvar)
        if z is not None:
            zvar = nn.Variable.from_numpy_array(z)
        with nn.auto_forward():
            bn = F.batch_normalization(xvar, betavar, gammavar, rmeanvar, rvarvar,
                                       axes, decay_rate, eps, batch_stat, output_stat)
            if z is None:
                if output_stat:
                    y = bn[0]
                else:
                    y = bn
            else:
                if output_stat:
                    y = F.add2(bn[0], zvar)
                else:
                    y = F.add2(bn, zvar)
            y = F.relu(y)
        rmean[:] = rmeanvar.d
        rvar[:] = rvarvar.d
        if output_stat:
            return y.d, bn[1].d, bn[2].d
        else:
            return y.d


def ref_grad_fused_batch_normalization(x, beta, gamma, rmean, rvar, z, dy, axes, decay_rate,
                                       eps, batch_stat, nonlinearity, output_stat, **kw):
    with nn.context_scope(cpu_context):
        xvar = nn.Variable.from_numpy_array(x, need_grad=True)
        xvar.g = 0
        betavar = nn.Variable.from_numpy_array(beta, need_grad=True)
        betavar.g = 0
        gammavar = nn.Variable.from_numpy_array(gamma, need_grad=True)
        gammavar.g = 0
        rmeanvar = nn.Variable.from_numpy_array(rmean)
        rmeanvar.g = 0
        rvarvar = nn.Variable.from_numpy_array(rvar)
        rvarvar.g = 0
        zvar = None
        if z is not None:
            zvar = nn.Variable.from_numpy_array(z, need_grad=True)
            zvar.g = 0
        with nn.auto_forward():
            bn = F.batch_normalization(xvar, betavar, gammavar, rmeanvar, rvarvar,
                                       axes, decay_rate, eps, batch_stat, output_stat)
            if z is None:
                if output_stat:
                    y1 = bn[0]
                else:
                    y1 = bn
            else:
                if output_stat:
                    y1 = F.add2(bn[0], zvar)
                else:
                    y1 = F.add2(bn, zvar)
            y = F.relu(y1)
        y.g = dy
        y.backward(dy)
        concat = [xvar.g.flatten(), betavar.g.flatten(), gammavar.g.flatten()]
        if z is not None:
            concat.append(zvar.g.flatten())
        return np.concatenate(concat)


def create_inputs(rng, axis, add):
    x = rng.randn(2, 3, 5, 4).astype(np.float32) * 2
    shape_stat = [1 for _ in range(x.ndim)]
    if add:
        # Note: The last dimension must be a multiple of 4
        # if we want to test cudnn BN persistent mode.
        z = rng.randn(2, 3, 5, 4).astype(np.float32) * 2
    else:
        z = None

    shape_stat[axis] = x.shape[axis]
    beta = rng.randn(*shape_stat).astype(np.float32)
    gamma = rng.randn(*shape_stat).astype(np.float32)
    rmean = np.zeros(shape_stat, dtype=np.float32)
    rvar = np.zeros(shape_stat, dtype=np.float32)
    return x, beta, gamma, rmean, rvar, z


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


@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("axis", [0, 3])
@pytest.mark.parametrize("decay_rate", [0.9])
@pytest.mark.parametrize("eps", [1e-5])
@pytest.mark.parametrize("nonlinearity", ['relu'])
@pytest.mark.parametrize("output_stat", [False])  # [True, False])
@pytest.mark.parametrize("add", [True, False])
@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("no_scale, no_bias", [[False, False], [True, True]])
@pytest.mark.parametrize("no_mean", [True, False])
@pytest.mark.parametrize("no_variance", [True, False])
def test_fused_batch_normalization_forward_backward(seed, axis, decay_rate, eps,
                                                    nonlinearity,
                                                    output_stat, add,
                                                    ctx, func_name,
                                                    no_scale, no_bias, no_mean, no_variance):
    import platform
    if platform.system() == 'Windows' and len(ctx.backend) > 1:
        pytest.skip(
            "Currently not worked with CUDA/cuDNN on Windows platform.")  # TODO

    from nbla_test_utils import function_tester
    rng = np.random.RandomState(seed)
    inputs = list(create_inputs(rng, axis, add))
    axes = [axis]
    batch_stat = True
    inputs = mask_inputs(inputs, no_scale, no_bias, no_mean, no_variance)
    function_tester(rng, F.fused_batch_normalization, ref_fused_batch_normalization,
                    inputs,
                    ref_grad=ref_grad_fused_batch_normalization,
                    func_args=[axes, decay_rate, eps,
                               batch_stat, nonlinearity, output_stat],
                    backward=[True, True, True, False, False, add],
                    ctx=ctx, func_name=func_name, dstep=1e-2, atol_b=1e-2)

    # Check if running mean and var works.
    if no_mean and no_variance:
        return

    vinputs = []
    for i in inputs:
        if i is None:
            vinputs.append(None)
            continue
        vinputs.append(nn.Variable.from_numpy_array(i, need_grad=True))
    for i in range(5):
        inputs[0] = rng.randn(*inputs[0].shape)
        vinputs[0].d[...] = inputs[0]
        ref_y = ref_fused_batch_normalization(
            *(inputs + [axes, decay_rate, eps, batch_stat, nonlinearity, output_stat]))
        with nn.context_scope(ctx), nn.auto_forward():
            y = F.fused_batch_normalization(
                *(vinputs + [axes, decay_rate, eps, batch_stat, nonlinearity, output_stat]))
        assert_allclose(vinputs[3].d, inputs[3])
        assert_allclose(vinputs[4].d, inputs[4], atol=1e-3)

    # Check if global stat mode works
    batch_stat = False
    if output_stat:
        return
    ref_y = ref_fused_batch_normalization(
        *(inputs + [axes, decay_rate, eps, batch_stat, nonlinearity, output_stat]))
    with nn.context_scope(ctx), nn.auto_forward():
        y = F.fused_batch_normalization(
            *(vinputs + [axes, decay_rate, eps, batch_stat, nonlinearity, output_stat]))
    assert_allclose(ref_y, y.d, atol=1e-6)


@pytest.mark.parametrize("seed", [313])
# @pytest.mark.parametrize("axis", [0, 3])
@pytest.mark.parametrize("axis", [1, 3])
@pytest.mark.parametrize("decay_rate", [0.9])
@pytest.mark.parametrize("eps", [1e-5])
@pytest.mark.parametrize("nonlinearity", ['relu'])
# Wait until FusedBN with test mode is supported
# @pytest.mark.parametrize("output_stat, batch_stat", [[False, False], [False, True]])
@pytest.mark.parametrize("output_stat, batch_stat", [[False, True]])
@pytest.mark.parametrize("add", [True, False])
@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("no_scale, no_bias", [[False, False], [True, True]])
@pytest.mark.parametrize("no_mean", [True, False])
@pytest.mark.parametrize("no_variance", [True, False])
def test_fused_batch_normalization_double_backward(seed, axis, decay_rate, eps,
                                                   nonlinearity,
                                                   output_stat, batch_stat,
                                                   add,
                                                   ctx, func_name,
                                                   no_scale, no_bias, no_mean, no_variance):
    import platform
    if platform.system() == 'Windows' and len(ctx.backend) > 1:
        pytest.skip(
            "Currently not worked with CUDA/cuDNN on Windows platform.")  # TODO

    from nbla_test_utils import backward_function_tester, grad_function_forward_function_output
    from nnabla.backward_function.fused_batch_normalization import FusedBatchNormalizationBackward

    rng = np.random.RandomState(seed)
    inputs = list(create_inputs(rng, axis, add))
    axes = [axis]
    func_args = [axes, decay_rate, eps, batch_stat, nonlinearity, output_stat]
    inputs = mask_inputs(inputs, no_scale, no_bias, no_mean, no_variance)

    # 2nd-order
    backward = [True, True, True, False, False, add] if batch_stat else \
        [False, False, False, False, False, False]
    backward_function_tester(rng, F.fused_batch_normalization,
                             inputs,
                             func_args=func_args,
                             backward=backward,
                             ctx=ctx)
    # 3rd-order
    func_args = func_args[:-1]
    fused_batch_normalization_backward, y = \
        grad_function_forward_function_output(FusedBatchNormalizationBackward,
                                              F.fused_batch_normalization,
                                              ctx, inputs, *func_args)
    fused_batch_normalization_backward.is_add = add
    ginputs = [rng.randn(*y.shape)] + inputs + [rng.randn(*y.shape)] if add else \
        [rng.randn(*y.shape)] + inputs[:-1] + [rng.randn(*y.shape)]
    backward_function_tester(rng, fused_batch_normalization_backward,
                             inputs=ginputs,
                             func_args=[],
                             backward=[True, True, False, True,
                                       False, False, False, add],
                             ctx=ctx, atol_accum=5e-2, dstep=1e-3, non_accum_check=True)
