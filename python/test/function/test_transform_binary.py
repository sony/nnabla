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


from nbla_test_utils import (
    function_tester,
    list_ctx_and_func_name)

# -----------------------------------------------------------------------------
# Reference functions
# -----------------------------------------------------------------------------


def ref_add2(x, y):
    return x + y


def ref_sub2(x, y):
    return x - y


def ref_mul2(x, y):
    return x * y


def ref_div2(x, y):
    return x / y


def ref_pow2(x, y):
    return x ** y


def ref_maximum2(x, y):
    return np.maximum(x, y)


def ref_minimum2(x, y):
    return np.minimum(x, y)

# -----------------------------------------------------------------------------
# Initializer
# -----------------------------------------------------------------------------


def get_inputs(fname, shapes, rng):
    if fname == 'div2':
        denom = rng.randn(*shapes[1]).astype(np.float32)
        denom[np.abs(denom) < 0.5] = 0.5
        return [rng.randn(*shapes[0]).astype(np.float32), denom]
    if fname == 'pow2':
        return [rng.rand(*shapes[0]).astype(np.float32) + 0.5,
                rng.randn(*shapes[1]).astype(np.float32)]
    return [rng.randn(*shapes[i]).astype(np.float32) * 2 for i in range(2)]

# -----------------------------------------------------------------------------
# Test body
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("fname, ctx, func_name",
                         list_ctx_and_func_name(['sub2',
                                                 'mul2',
                                                 'div2',
                                                 'pow2']))
@pytest.mark.parametrize("seed", [314])
def test_transform_binary_inplace(seed, fname, ctx, func_name):
    from nbla_test_utils import inplace_function_test_helper
    x0 = nn.Variable([2, 3, 4], need_grad=True)
    x1 = nn.Variable([2, 3, 4], need_grad=True)
    func = getattr(F, fname)
    inplace_function_test_helper(
        [x0, x1], func, ctx=ctx, rng=np.random.RandomState(seed))


atol_list = {
    'add2': (1e-6, 4e-3),
    'sub2': (1e-6, 3e-3),
    'mul2': (1e-6, 2e-2),
    'div2': (1e-4, 1e-1),
    'pow2': (1e-4, 1e-1),
    'maximum2': (1e-6, 3e-3),
    'minimum2': (1e-6, 4e-3),
}


@pytest.mark.parametrize("fname, ctx, func_name",
                         list_ctx_and_func_name(['add2',
                                                 'sub2',
                                                 'mul2',
                                                 'div2',
                                                 'pow2',
                                                 'maximum2',
                                                 'minimum2']))
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("broadcast_dims", [
    (None, None),
    (None, (0,)),
    ((1,), None),
    (None, (2,)),
    ((0, 2), None),
    ((0,), (2,))])
def test_transform_binary_forward_backward(fname, ctx, func_name, broadcast_dims, seed):
    from nbla_test_utils import function_tester
    atol_f, atol_b = atol_list[fname]
    func = getattr(F, fname)
    ref_func = eval('ref_' + fname)
    rng = np.random.RandomState(seed)
    shape = [2, 3, 4]
    shapes = []
    for i in range(2):
        if broadcast_dims[i] is None:
            shapes.append(shape)
            continue
        s = np.array(shape).copy()
        s[np.array(broadcast_dims[i])] = 1
        shapes.append(s.tolist())
    inputs = get_inputs(fname, shapes, rng)
    function_tester(rng, func, ref_func, inputs,
                    ctx=ctx, func_name=func_name,
                    atol_f=atol_f, atol_b=atol_b)
