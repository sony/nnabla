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
import nnabla as nn
from nbla_test_utils import list_context
from nnabla.testing import assert_allclose

ctxs = list_context('PowScalar')


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("val", [0.5, 1, 2])
def test_pow_scalar_forward_backward(seed, val, ctx, func_name):
    from nbla_test_utils import function_tester
    rng = np.random.RandomState(seed)
    inputs = [rng.rand(2, 3, 4).astype(np.float32) + 0.5]
    function_tester(rng, F.pow_scalar, lambda x, y: x ** y, inputs,
                    func_args=[val], atol_b=5e-2,
                    ctx=ctx, func_name=func_name)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("val", [0.5, 1, 2])
def test_pow_scalar_inplace(val, seed, ctx, func_name):
    inputs = [nn.Variable([2, 3, 4], need_grad=True)]
    func_args = [val]
    func_kwargs = {}
    func = F.pow_scalar
    rng = np.random.RandomState(seed)

    # copied from `inplace_function_test_helper` function and modified
    with nn.context_scope(ctx):
        a_s = [inp * 1.0 for inp in inputs]
        y = func(*(a_s + list(func_args)), inplace=False, **func_kwargs)
        l = F.sum(y)
        a_s_i = [inp * 1.0 for inp in inputs]
        y_i = func(*(a_s_i + list(func_args)), inplace=True, **func_kwargs)
        l_i = F.sum(y_i)
    data = [((rng.randint(5, size=inp.shape).astype(np.float32) + 1.0) * 0.2,
             rng.randn(*inp.shape)) for inp in inputs]
    for i in range(len(data)):
        inputs[i].d = data[i][0]
        inputs[i].g = data[i][1]
    l.forward()
    l.backward()
    grads = [inp.g.copy() for inp in inputs]
    for i in range(len(data)):
        inputs[i].d = data[i][0]
        inputs[i].g = data[i][1]
    l_i.forward()
    l_i.backward()
    grads_i = [inp.g.copy() for inp in inputs]
    for g, g_i in zip(grads, grads_i):
        assert_allclose(g, g_i)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("val", [0.5, -0.5, 1, 2])
def test_pow_scalar_double_backward(seed, val, ctx, func_name):
    from nbla_test_utils import backward_function_tester
    rng = np.random.RandomState(seed)
    inputs = [(rng.randint(5, size=(2, 3)).astype(np.float32) + 1.0) * 0.2]
    backward_function_tester(rng, F.pow_scalar, None,
                             inputs=inputs,
                             func_args=[val], func_kwargs={},
                             atol_b=1e-2,
                             atol_accum=2e-2,
                             dstep=1e-3,
                             ctx=ctx, func_name=None,
                             disable_half_test=False)
