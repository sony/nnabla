# Copyright 2019,2020,2021 Sony Corporation.
# Copyright 2021 Sony Group Corporation.
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

ctxs = list_context('LayerNormalization')


def ref_layer_normalization(x, beta, gamma, batch_axis, eps, output_stat):
    batch_axis = _force_list(batch_axis)

    axes = tuple(_get_axes_excluding(len(x.shape), batch_axis))

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


def create_inputs(rng, x_shape, batch_axis, no_scale, no_bias):
    x = rng.randn(*x_shape).astype(np.float32)

    stat_shape = list(x_shape)
    for baxis in _force_list(batch_axis):
        stat_shape[baxis] = 1

    beta = None if no_bias else rng.randn(*stat_shape).astype(np.float32)
    gamma = None if no_scale else rng.randn(*stat_shape).astype(np.float32)

    return x, beta, gamma


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("x_shape , batch_axis", [((4, 3, 8, 8), 0),
                                                  ((4, 16, 16, 8), 0),
                                                  ((16, 1), 0),
                                                  ((3, 32, 4), 0),
                                                  # time-series (T, B, C) or (B, T, C)
                                                  ((10, 4, 16), [0, 1])
                                                  ])
@pytest.mark.parametrize("eps", [1e-05])
@pytest.mark.parametrize("output_stat", [False, True])
@pytest.mark.parametrize("no_scale", [False, True])
@pytest.mark.parametrize("no_bias", [False, True])
def test_layer_normalization_forward(ctx, func_name, seed, x_shape, batch_axis, eps, output_stat, no_scale, no_bias):
    from nbla_test_utils import function_tester

    rng = np.random.RandomState(seed)
    x, beta, gamma = create_inputs(rng, x_shape, batch_axis, no_scale, no_bias)

    function_tester(rng, F.layer_normalization, ref_layer_normalization, [x, beta, gamma], [batch_axis, eps, output_stat], ctx=ctx,
                    func_name=func_name, dstep=1e-2, atol_b=1e-2, backward=[False, False, False], disable_half_test=True)


x_shape_and_batch_axis = [
    # Outer batch axis cases
    ((2, 3, 4, 4), 0),
    ((16, 1), 0),
    ((1, 1), 0),
    # Inner batch axis cases
    ((2, 3, 5, 7), 1),
    ((2, 3, 5, 7), 3),
    # Multiple batch axis cases
    ((2, 3, 5, 7), [0, 2]),
    ((2, 3, 5, 7), [3, 1]),
    # time-series (T, B, C) or (B, T, C)
    ((3, 2, 5), [0, 1]),
]


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("x_shape , batch_axis", x_shape_and_batch_axis)
@pytest.mark.parametrize("eps", [1e-05])
@pytest.mark.parametrize("output_stat", [False, True])
@pytest.mark.parametrize("no_scale", [False, True])
@pytest.mark.parametrize("no_bias", [False, True])
def test_layer_normalization_forward_backward(ctx, func_name, seed, x_shape, batch_axis, eps, output_stat, no_scale, no_bias):
    from nbla_test_utils import function_tester

    rng = np.random.RandomState(seed)
    x, beta, gamma = create_inputs(rng, x_shape, batch_axis, no_scale, no_bias)

    function_tester(rng, F.layer_normalization, ref_layer_normalization, [x, beta, gamma], [batch_axis, eps, output_stat], ctx=ctx,
                    func_name=func_name, dstep=1e-2, atol_b=1e-2, backward=[True, not no_bias, not no_scale], disable_half_test=True)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("x_shape , batch_axis", x_shape_and_batch_axis)
@pytest.mark.parametrize("eps", [1e-05])
@pytest.mark.parametrize("output_stat", [False])
@pytest.mark.parametrize("no_scale", [False, True])
@pytest.mark.parametrize("no_bias", [False, True])
def test_layer_normalization_double_backward(ctx, func_name, seed, x_shape, batch_axis, eps, output_stat, no_scale, no_bias):
    from nbla_test_utils import backward_function_tester
    rng = np.random.RandomState(seed)
    x, beta, gamma = create_inputs(rng, x_shape, batch_axis, no_scale, no_bias)
    backward = [True, not no_bias, not no_scale]
    backward_function_tester(rng, F.layer_normalization,
                             inputs=[x, beta, gamma],
                             func_args=[batch_axis, eps, output_stat],
                             backward=backward,
                             atol_f=2e-4,
                             ctx=ctx)
