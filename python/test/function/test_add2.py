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

"""
This is still left unlike other *2 operation (sub2, mul2, ...) because
it has cudnn implementation.
"""
import pytest
import numpy as np
import nnabla as nn
import nnabla.functions as F
from nbla_test_utils import list_context

ctxs = list_context('Add2')


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [314])
@pytest.mark.parametrize("broadcast_dims", [
    (None, None),
    (None, (0,)),
    ((1,), None),
    (None, (2,)),
    ((0, 2), None),
    ((0,), (2,))])
def test_add2_forward_backward(broadcast_dims, seed, ctx, func_name):
    from nbla_test_utils import function_tester
    rng = np.random.RandomState(seed)
    inputs = [rng.randn(2, 3, 4).astype(np.float32) * 2 for _ in range(2)]
    for i in range(2):
        if broadcast_dims[i] is not None:
            for d in broadcast_dims[i]:
                inputs[i] = inputs[i].mean(d, keepdims=True)
    function_tester(rng, F.add2, lambda x, y: x + y, inputs,
                    ctx=ctx, func_name=func_name, atol_b=2e-3)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [314])
def test_add2_inplace(seed, ctx, func_name):
    from nbla_test_utils import inplace_function_test_helper
    x0 = nn.Variable([2, 3, 4], need_grad=True)
    x1 = nn.Variable([2, 3, 4], need_grad=True)
    inplace_function_test_helper(
        [x0, x1], F.add2, ctx=ctx, func_name=func_name, rng=np.random.RandomState(seed))


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
def test_add2_double_backward(seed, ctx, func_name):
    # inplace test is not needed since grad does not depend on the inputs
    from nbla_test_utils import backward_function_tester
    rng = np.random.RandomState(seed)
    inputs = [rng.randn(2, 3).astype(np.float32),
              rng.randn(2, 3).astype(np.float32)]
    backward_function_tester(rng, F.add2,
                             inputs=inputs,
                             func_args=[], func_kwargs={},
                             atol_accum=1e-3,
                             dstep=1e-3,
                             ctx=ctx)
