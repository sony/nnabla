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


# ----------------------------------------------------------------------------
# Logical scalar
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [314])
@pytest.mark.parametrize("fname, ctx, func_name",
                         list_ctx_and_func_name(['logical_and_scalar', 'logical_or_scalar', 'logical_xor_scalar']))
@pytest.mark.parametrize("val", [False, True])
def test_logical_scalar_forward_backward(val, seed, fname, ctx, func_name):
    func = getattr(F, fname)
    ref_func = getattr(np, fname.replace('_scalar', ''))
    rng = np.random.RandomState(seed)
    inputs = [rng.randint(0, 2, size=(2, 3, 4)).astype(np.float32)]
    function_tester(rng, func, ref_func, inputs, [val],
                    ctx=ctx, backward=[False], func_name=func_name)


opstrs = {
    'greater': '>',
    'greater_equal': '>=',
    'less': '<',
    'less_equal': '<=',
    'equal': '==',
    'not_equal': '!='}


@pytest.mark.parametrize("seed", [314])
@pytest.mark.parametrize("fname, ctx, func_name",
                         list_ctx_and_func_name(['greater_scalar',
                                                 'greater_equal_scalar',
                                                 'less_scalar',
                                                 'less_equal_scalar',
                                                 'equal_scalar',
                                                 'not_equal_scalar']))
@pytest.mark.parametrize("val", [-0.5, 0., 1.])
def test_logical_scalar_compare_forward_backward(val, seed, fname, ctx, func_name):
    opstr = opstrs[fname.replace('_scalar', '')]
    func = getattr(F, fname)
    ref_func = eval('lambda x, y: x {} y'.format(opstr))
    rng = np.random.RandomState(seed)
    inputs = [rng.randint(0, 2, size=(2, 3, 4)).astype(np.float32)
              for _ in range(1)]
    inputs[0][..., :2] = val
    function_tester(rng, func, ref_func, inputs, [val],
                    ctx=ctx, backward=[False, False], func_name=func_name)

# ----------------------------------------------------------------------------
# Logical binary
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [314])
@pytest.mark.parametrize("fname, ctx, func_name",
                         list_ctx_and_func_name(['logical_and', 'logical_or', 'logical_xor']))
def test_logical_binary_forward_backward(seed, fname, ctx, func_name):
    func = getattr(F, fname)
    ref_func = getattr(np, fname)
    rng = np.random.RandomState(seed)
    inputs = [rng.randint(0, 2, size=(2, 3, 4)).astype(np.float32)
              for _ in range(2)]
    function_tester(rng, func, ref_func, inputs,
                    ctx=ctx, backward=[False, False], func_name=func_name)


@pytest.mark.parametrize("seed", [314])
@pytest.mark.parametrize("fname, ctx, func_name",
                         list_ctx_and_func_name(['greater',
                                                 'greater_equal',
                                                 'less',
                                                 'less_equal',
                                                 'equal',
                                                 'not_equal']))
def test_logical_binary_compare_forward_backward(seed, fname, ctx, func_name):
    func = getattr(F, fname)
    opstr = opstrs[fname]
    ref_func = eval('lambda x, y: x {} y'.format(opstr))
    rng = np.random.RandomState(seed)
    inputs = [rng.randint(0, 2, size=(2, 3, 4)).astype(np.float32)
              for _ in range(2)]
    inputs[0][..., :2] = inputs[1][..., :2]
    function_tester(rng, func, ref_func, inputs,
                    ctx=ctx, backward=[False, False], func_name=func_name)


# ----------------------------------------------------------------------------
# Logical not
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [314])
@pytest.mark.parametrize("fname, ctx, func_name", list_ctx_and_func_name(['logical_not']))
def test_logical_not_forward_backward(seed, fname, ctx, func_name):
    func = getattr(F, fname)
    ref_func = getattr(np, fname)
    rng = np.random.RandomState(seed)
    inputs = [rng.randint(0, 2, size=(2, 3, 4)).astype(np.float32)]
    function_tester(rng, func, ref_func, inputs,
                    ctx=ctx, backward=[False], func_name=func_name)
