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
import nnabla.functions as F
from nbla_test_utils import list_context

ctxs = list_context('QuantizeLinear')


def std_round(x):
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def ref_quantize_linear(x, scale, zero_point, round_mode, narrow_range, dtype, **kw):
    y = x / scale
    if round_mode == "HALF_AWAY_FROM_ZERO":
        y = std_round(y)
    elif round_mode == "HALF_TO_EVEN":
        y = np.round(y)
    y = y + zero_point

    min_range = np.iinfo(
        dtype).min if not narrow_range else np.iinfo(dtype).min + 1
    max_range = np.iinfo(dtype).max
    y = np.clip(y, min_range, max_range)
    return y


def ref_grad_quantize_linear(x, scale, zero_point, dy, round_mode, narrow_range, dtype, **kw):
    dx = dy / scale
    return dx.flatten()


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("x_shape, s_shape", [((2, 4, 8, 8), (1, 1, 1, 1)),
                                              ((2, 4, 8, 8), (1, 4, 1, 1)),
                                              ((2, 8, 8, 4), (1, 1, 1, 4)),
                                              ])
@pytest.mark.parametrize("round_mode", ["HALF_AWAY_FROM_ZERO", "HALF_TO_EVEN"])
@pytest.mark.parametrize("narrow_range", [False, True])
@pytest.mark.parametrize("dtype", [np.int8, np.uint8])
def test_quantize_linear_forward_backward(dtype, narrow_range, round_mode,
                                          x_shape, s_shape,
                                          seed, ctx, func_name):
    from nbla_test_utils import cap_ignore_region, function_tester
    rng = np.random.RandomState(seed)

    x = rng.randn(*x_shape)
    scale = rng.rand(*s_shape) + 0.001
    zero_point = rng.randint(np.iinfo(dtype).min // 3,
                             np.iinfo(dtype).max // 3, s_shape)
    inputs = [x, scale, zero_point]
    func_args = [round_mode, narrow_range, dtype]
    function_tester(rng, F.quantize_linear, ref_quantize_linear, inputs,
                    func_args=func_args,
                    ctx=ctx, func_name=func_name, ref_grad=ref_grad_quantize_linear,
                    disable_half_test=True,
                    backward=[True, False, False])


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("x_shape, s_shape", [((2, 4, 8, 8), (1, 1, 1, 1)),
                                              ((2, 4, 8, 8), (1, 4, 1, 1)),
                                              ((2, 8, 8, 4), (1, 1, 1, 4)),
                                              ])
@pytest.mark.parametrize("round_mode", ["HALF_AWAY_FROM_ZERO", "HALF_TO_EVEN"])
@pytest.mark.parametrize("narrow_range", [False, True])
@pytest.mark.parametrize("dtype", [np.int8, np.uint8])
def test_quantize_linear_double_backward(dtype, narrow_range, round_mode,
                                         x_shape, s_shape,
                                         seed, ctx, func_name):
    from nbla_test_utils import cap_ignore_region, backward_function_tester
    rng = np.random.RandomState(seed)

    x = rng.randn(*x_shape)
    scale = rng.rand(*s_shape) + 0.001
    zero_point = rng.randint(np.iinfo(dtype).min // 3,
                             np.iinfo(dtype).max // 3, s_shape)
    inputs = [x, scale, zero_point]
    func_args = [round_mode, narrow_range, dtype]
    backward_function_tester(rng, F.quantize_linear, inputs,
                             func_args=func_args,
                             ctx=ctx,
                             backward=[True, False, False], atol_accum=3e-2)
