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

ctxs = list_context('DequantizeLinear')


def std_round(x):
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def ref_dequantize_linear(x, scale, zero_point):
    return (x - zero_point) * scale


def ref_grad_dequantize_linear(x, scale, zero_point, dy, **kw):
    dx = dy * scale
    return dx.flatten()


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("x_shape, s_shape", [((2, 4, 8, 8), (1, 1, 1, 1)),
                                              ((2, 4, 8, 8), (1, 4, 1, 1)),
                                              ((2, 8, 8, 4), (1, 1, 1, 4)),
                                              ])
def test_dequantize_linear_forward_backward(x_shape, s_shape,
                                            seed, ctx, func_name):
    from nbla_test_utils import cap_ignore_region, function_tester
    rng = np.random.RandomState(seed)
    x = rng.randn(*x_shape)
    scale = rng.rand(*s_shape) + 0.001
    zero_point = rng.randint(np.iinfo(np.int8).min,
                             np.iinfo(np.int8).max, s_shape)
    inputs = [x, scale, zero_point]
    function_tester(rng, F.dequantize_linear, ref_dequantize_linear, inputs,
                    ctx=ctx, func_name=func_name, ref_grad=ref_grad_dequantize_linear,
                    disable_half_test=True,
                    backward=[True, False, False])


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("x_shape, s_shape", [((2, 4, 8, 8), (1, 1, 1, 1)),
                                              ((2, 4, 8, 8), (1, 4, 1, 1)),
                                              ((2, 8, 8, 4), (1, 1, 1, 4)),
                                              ])
def test_dequantize_linear_double_backward(x_shape, s_shape,
                                           seed, ctx, func_name):
    from nbla_test_utils import cap_ignore_region, backward_function_tester
    rng = np.random.RandomState(seed)
    x = rng.randn(*x_shape)
    scale = rng.rand(*s_shape) + 0.001
    zero_point = rng.randint(np.iinfo(np.int8).min,
                             np.iinfo(np.int8).max, s_shape)
    inputs = [x, scale, zero_point]
    backward_function_tester(rng, F.dequantize_linear, inputs,
                             ctx=ctx,
                             backward=[True, False, False])
