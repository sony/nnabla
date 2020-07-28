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

from nbla_test_utils import (
    function_tester,
    list_ctx_and_func_name)


def ref_func_binary_tanh(x):
    # Binary tanh, value @x==0 is set to -1
    return 2 * np.round(np.clip((x + 1) / 2, 0, 1)) - 1


def ref_grad_binary_tanh(x, dy, **kw):
    return (dy * (1 - np.floor(np.minimum(np.abs(x), 1)))).flatten()


def ref_func_binary_sigmoid(x):
    # Binary sigmoid, value @x==0 is set to 0
    return np.round(np.clip((x + 1) / 2, 0, 1))


def ref_grad_binary_sigmoid(x, dy, **kw):
    return (dy * (1 - np.floor(np.minimum(np.abs(x), 1))) / 2).flatten()


@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("act_name, ctx, func_name", list_ctx_and_func_name(['binary_tanh', 'binary_sigmoid']))
def test_activation_forward_backward(act_name, seed, ctx, func_name):
    act = getattr(F, act_name)
    ref_func = eval('ref_func_' + act_name)
    ref_grad = eval('ref_grad_' + act_name)
    rng = np.random.RandomState(seed)
    inputs = [rng.randn(2, 3, 4).astype(np.float32) * 2]

    function_tester(rng, act, ref_func, inputs,
                    atol_b=1e-2, dstep=1e-3, ref_grad=ref_grad,
                    ctx=ctx, func_name=func_name)


@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("act_name, ctx, func_name", list_ctx_and_func_name(['binary_tanh', 'binary_sigmoid']))
def test_activation_double_backward(act_name, seed, ctx, func_name):
    from nbla_test_utils import backward_function_tester
    act = getattr(F, act_name)
    rng = np.random.RandomState(seed)
    inputs = [rng.randn(2, 3, 4).astype(np.float32) * 2]

    backward_function_tester(rng, act, inputs, ctx=ctx)
