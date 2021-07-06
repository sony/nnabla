# Copyright 2019,2020,2021 Sony Corporation.
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

from nnabla.normalization_functions import _force_list, _get_axes_excluding

ctxs = list_context('InstanceNormalization')


def ref_instance_normalization(x, beta, gamma, channel_axis, batch_axis, eps, output_stat):

    ignore_axes = _force_list(batch_axis) + [channel_axis, ]

    axes = tuple(_get_axes_excluding(len(x.shape), ignore_axes))

    x_mean = x.mean(axis=axes, keepdims=True)
    x_var = x.var(axis=axes, keepdims=True)

    norm = (x - x_mean) / (x_var + eps) ** 0.5

    if gamma is not None:
        norm *= gamma

    if beta is not None:
        norm += beta

    if output_stat:
        return norm, x_mean, x_var

    return norm


def create_inputs(rng, x_shape, batch_axis, channel_axis, no_scale, no_bias):
    x = np.array(rng.randn(*x_shape).astype(np.float32))

    stat_shape = tuple([x_shape[i] if i in _force_list(batch_axis) + [channel_axis, ] else 1
                        for i in range(len(x_shape))])

    beta = None if no_bias else rng.randn(*stat_shape).astype(np.float32)
    gamma = None if no_scale else rng.randn(*stat_shape).astype(np.float32)

    return x, beta, gamma


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("x_shape , batch_axis, channel_axis",
                         [((4, 32, 8, 8), 0, 1),  # convolution (NCHW)
                          ((4, 16, 16, 8), 0, 3),  # convolution (NHWC)
                          ((16, 4), 0, 1),  # affine
                          # time-series (T, B, C) or (B, T, C)
                          ((10, 4, 16), [0, 1], 2)
                          ])
@pytest.mark.parametrize("eps", [1e-05])
@pytest.mark.parametrize("output_stat", [False, True])
@pytest.mark.parametrize("no_scale", [False, True])
@pytest.mark.parametrize("no_bias", [False, True])
def test_instance_normalization_forward(ctx, func_name, seed, x_shape, batch_axis, channel_axis, eps, output_stat, no_scale, no_bias):
    from nbla_test_utils import function_tester

    rng = np.random.RandomState(seed)
    x, beta, gamma = create_inputs(
        rng, x_shape, batch_axis, channel_axis, no_scale, no_bias)

    function_tester(rng, F.instance_normalization, ref_instance_normalization, [x, beta, gamma], [channel_axis, batch_axis, eps, output_stat], ctx=ctx,
                    func_name=func_name, dstep=1e-2, atol_b=1e-2, backward=[False, False, False], disable_half_test=True)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("x_shape , batch_axis, channel_axis",
                         [((2, 4, 3, 3), 0, 1),  # convolution (NCHW)
                          ((2, 3, 3, 4), 0, 3),  # convolution (NHWC)
                          ((16, 4), 0, 1),  # affine
                          # time-series (T, B, C) or (B, T, C)
                          ((5, 2, 6), [0, 1], 2)
                          ])
@pytest.mark.parametrize("eps", [1e-05])
@pytest.mark.parametrize("output_stat", [False, True])
@pytest.mark.parametrize("no_scale", [False, True])
@pytest.mark.parametrize("no_bias", [False, True])
def test_instance_normalization_forward_backward(ctx, func_name, seed, x_shape, batch_axis, channel_axis, eps, output_stat, no_scale, no_bias):
    from nbla_test_utils import function_tester

    rng = np.random.RandomState(seed)
    x, beta, gamma = create_inputs(
        rng, x_shape, batch_axis, channel_axis, no_scale, no_bias)

    function_tester(rng, F.instance_normalization, ref_instance_normalization, [x, beta, gamma], [channel_axis, batch_axis, eps, output_stat], ctx=ctx,
                    func_name=func_name, dstep=1e-2, atol_b=1e-2, backward=[True, not no_bias, not no_scale], disable_half_test=True)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("x_shape , batch_axis, channel_axis",
                         [((2, 4, 3, 3), 0, 1),  # convolution (NCHW)
                          ((2, 3, 3, 4), 0, 3),  # convolution (NHWC)
                          ((16, 4), 0, 1),  # affine
                          # time-series (T, B, C) or (B, T, C)
                          ((5, 2, 6), [0, 1], 2)
                          ])
@pytest.mark.parametrize("eps", [1e-05])
@pytest.mark.parametrize("output_stat", [False])
@pytest.mark.parametrize("no_scale", [False, True])
@pytest.mark.parametrize("no_bias", [False, True])
def test_instance_normalization_double_backward(ctx, func_name, seed, x_shape, batch_axis, channel_axis, eps, output_stat, no_scale, no_bias):
    from nbla_test_utils import backward_function_tester
    rng = np.random.RandomState(seed)
    x, beta, gamma = create_inputs(
        rng, x_shape, batch_axis, channel_axis, no_scale, no_bias)
    backward = [True, not no_bias, not no_scale]
    backward_function_tester(rng, F.instance_normalization,
                             inputs=[x, beta, gamma],
                             func_args=[channel_axis,
                                        batch_axis, eps, output_stat],
                             backward=backward,
                             atol_f=2e-4,
                             ctx=ctx)
