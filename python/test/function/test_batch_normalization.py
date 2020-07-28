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

ctxs = list_context('BatchNormalization')


def ref_batch_normalization(x, beta, gamma, rmean, rvar, axes, decay_rate,
                            eps, batch_stat, output_stat):

    beta, gamma, rmean, rvar

    assert len(axes) == 1
    reduc_axes = list(range(x.ndim))
    del reduc_axes[axes[0]]
    reduc_axes = tuple(reduc_axes)
    m = x.size / x.shape[axes[0]]
    if batch_stat:
        mean = x.mean(reduc_axes, keepdims=True)
        var = x.var(reduc_axes, keepdims=True)
        rmean[...] = decay_rate * rmean + (1 - decay_rate) * mean
        rvar[...] = decay_rate * rvar + (1 - decay_rate) * var * m / (m - 1)
    else:
        mean = rmean
        var = rvar
    x_hat = (x - mean) / np.sqrt(var + eps)
    y = x_hat * gamma + beta
    if output_stat:
        return y, mean, var
    return y


def create_inputs(rng, axis):
    x = rng.randn(2, 3, 4).astype(np.float32) * 2
    shape_stat = [1 for _ in range(x.ndim)]
    shape_stat[axis] = x.shape[axis]
    beta = rng.randn(*shape_stat).astype(np.float32)
    gamma = rng.randn(*shape_stat).astype(np.float32)
    rmean = rng.randn(*shape_stat).astype(np.float32)
    rvar = 1 + rng.rand(*shape_stat).astype(np.float32)
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
@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("decay_rate", [0.9])
@pytest.mark.parametrize("eps", [1e-5])
@pytest.mark.parametrize("output_stat, batch_stat", [[False, False], [False, True], [True, True]])
@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("no_scale, no_bias", [[False, False], [True, True]])
@pytest.mark.parametrize("no_mean", [True, False])
@pytest.mark.parametrize("no_variance", [True, False])
def test_batch_normalization_forward_backward(seed, axis, decay_rate, eps,
                                              output_stat, batch_stat, ctx, func_name,
                                              no_scale, no_bias, no_mean, no_variance):
    from nbla_test_utils import function_tester
    rng = np.random.RandomState(seed)
    inputs = list(create_inputs(rng, axis))
    axes = [axis]
    if not batch_stat and (no_mean or no_variance):
        # check prohibited condition for mean=None and variance=None
        vinputs = []
        for i in inputs:
            vinputs.append(nn.Variable(i.shape, True))

        vinputs = mask_vinputs(
            vinputs, no_scale, no_bias, no_mean, no_variance)
        with pytest.raises(ValueError):
            F.batch_normalization(*vinputs, axes=axes, decay_rate=decay_rate,
                                  eps=eps, batch_stat=batch_stat, output_stat=output_stat)
        return
    else:
        inputs = mask_inputs(inputs, no_scale, no_bias, no_mean, no_variance)
        function_tester(rng, F.batch_normalization, ref_batch_normalization,
                        inputs,
                        func_args=[axes, decay_rate, eps,
                                   batch_stat, output_stat],
                        backward=[True, True, True, False, False],
                        ctx=ctx, func_name=func_name, dstep=1e-2, atol_b=1e-2)

    # Check if running mean and var works.
    if no_mean and no_variance:
        return

    vinputs = []
    for i in inputs:
        vinputs.append(nn.Variable(i.shape, True))
        vinputs[-1].d = i

    vinputs = mask_vinputs(vinputs, no_scale, no_bias, no_mean, no_variance)

    for i in range(5):
        inputs[0] = rng.randn(*inputs[0].shape)
        vinputs[0].d[...] = inputs[0]
        ref_y = ref_batch_normalization(
            *(inputs + [axes, decay_rate, eps, batch_stat, output_stat]))
        with nn.context_scope(ctx), nn.auto_forward():
            y = F.batch_normalization(
                *(vinputs + [axes, decay_rate, eps, batch_stat, output_stat]))
        if not no_mean:
            assert_allclose(vinputs[3].d, inputs[3], atol=1e-7)

        if not no_variance:
            assert_allclose(vinputs[4].d, inputs[4])

    # Check if global stat mode works
    batch_stat = False

    if no_mean or no_variance:
        return

    if output_stat:
        return
    ref_y = ref_batch_normalization(
        *(inputs + [axes, decay_rate, eps, batch_stat, output_stat]))
    with nn.context_scope(ctx), nn.auto_forward():
        y = F.batch_normalization(
            *(vinputs + [axes, decay_rate, eps, batch_stat, output_stat]))
    assert_allclose(ref_y, y.d, atol=1e-6)


def ref_batch_normalization_for_multiple_axes(x, beta, gamma, rmean, rvar, axes, decay_rate,
                                              eps, batch_stat, output_stat):
    reduc_axes = [i for i in range(x.ndim) if not i in axes]
    div_factor = 1
    for i in range(len(axes)):
        div_factor = div_factor*x.shape[axes[i]]
    reduc_axes = tuple(reduc_axes)
    m = x.size / div_factor
    if batch_stat:
        mean = x.mean(reduc_axes, keepdims=True)
        var = x.var(reduc_axes, keepdims=True)
        rmean[...] = decay_rate * rmean + (1 - decay_rate) * mean
        rvar[...] = decay_rate * rvar + (1 - decay_rate) * var * m / (m - 1)
    else:
        mean = rmean
        var = rvar
    x_hat = (x - mean) / np.sqrt(var + eps)
    y = x_hat * gamma + beta
    if output_stat:
        return y, mean, var
    return y


def create_inputs_for_multiple_axes(rng, axes):
    x = rng.randn(2, 3, 4).astype(np.float32) * 2
    shape_stat = [1 for _ in range(x.ndim)]
    for i in range(len(axes)):
        shape_stat[axes[i]] = x.shape[axes[i]]
    beta = rng.randn(*shape_stat).astype(np.float32)
    gamma = rng.randn(*shape_stat).astype(np.float32)
    rmean = np.zeros(shape_stat, dtype=np.float32)
    rvar = np.zeros(shape_stat, dtype=np.float32)
    return x, beta, gamma, rmean, rvar


@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("axes", [[0, 1], [1, 2], [0, 2]])
@pytest.mark.parametrize("decay_rate", [0.9])
@pytest.mark.parametrize("eps", [1e-5])
@pytest.mark.parametrize("output_stat", [True, False])
@pytest.mark.parametrize("ctx, func_name", ctxs)
def test_batch_normalization_for_multiple_axes_forward_backward(seed, axes, decay_rate, eps,
                                                                output_stat, ctx, func_name):
    rng = np.random.RandomState(seed)
    inputs = list(create_inputs_for_multiple_axes(rng, axes))
    vinputs = []
    for i in inputs:
        vinputs.append(nn.Variable(i.shape, True))
        vinputs[-1].d = i

    # Check if global stat mode works
    batch_stat = False
    if output_stat:
        return
    ref_y = ref_batch_normalization_for_multiple_axes(
        *(inputs + [axes, decay_rate, eps, batch_stat, output_stat]))
    with nn.context_scope(ctx), nn.auto_forward():
        y = F.batch_normalization(
            *(vinputs + [axes, decay_rate, eps, batch_stat, output_stat]))
    assert_allclose(ref_y, y.d, atol=1e-6)


@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("decay_rate", [0.9])
@pytest.mark.parametrize("eps", [1e-5])
@pytest.mark.parametrize("output_stat, batch_stat", [[False, False], [False, True]])
@pytest.mark.parametrize("no_scale, no_bias", [[False, False], [True, True]])
@pytest.mark.parametrize("ctx, func_name", ctxs)
def test_batch_normalization_double_backward(seed, axis, decay_rate, eps,
                                             output_stat, batch_stat, no_scale, no_bias, ctx, func_name):
    from nbla_test_utils import backward_function_tester
    from nnabla.backward_function.batch_normalization import BatchNormalizationBackward

    rng = np.random.RandomState(seed)
    inputs = list(create_inputs(rng, axis))
    inputs = mask_vinputs(inputs, no_scale, no_bias, False, False)

    axes = [axis]
    func_args = [axes, decay_rate, eps, batch_stat, output_stat]

    # 2nd-order
    backward_function_tester(rng, F.batch_normalization,
                             inputs,
                             func_args=func_args,
                             backward=[True, not no_bias,
                                       not no_scale, False, False],
                             ctx=ctx, atol_accum=1e-2, dstep=1e-3)
    # 3rd-order
    func_args = func_args[:-1] + [no_scale, no_bias]
    df = BatchNormalizationBackward(ctx, *func_args)
    ginputs = list(filter(lambda x: x is not None, [
                   rng.randn(*inputs[0].shape)] + inputs))
    if not no_scale and not no_bias:
        backward = [True, True, False, True, False, False]
    else:
        backward = [True, True, False, False]
    backward_function_tester(rng, df,
                             inputs=ginputs,
                             func_args=[],
                             backward=backward,
                             ctx=ctx, atol_accum=1e-2, dstep=1e-3, non_accum_check=True)
