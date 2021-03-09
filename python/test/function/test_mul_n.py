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
from nbla_test_utils import list_context, function_tester

ctxs = list_context('MulN')


def ref_function(*inputs, **params):
    y = 1
    for i in range(len(inputs)):
        y *= inputs[i]
    return y


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [314])
@pytest.mark.parametrize('num_inputs', [2, 3, 5])
def test_mul_n_forward_backward(num_inputs, seed, ctx, func_name):
    rng = np.random.RandomState(seed)
    shape0 = [2, 3, 4]
    inputs = []
    for i in range(num_inputs):
        inputs.append(rng.randn(*shape0).astype(np.float32))
    function_tester(rng, F.mul_n, ref_function, inputs,
                    ctx=ctx, func_name=func_name, atol_b=2e-3)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize('num_inputs', [2, 3, 5])
def test_mul_n_double_backward(num_inputs, seed, ctx, func_name):
    from nbla_test_utils import backward_function_tester
    rng = np.random.RandomState(seed)
    shape0 = [2, 3, 4]
    inputs = []
    for i in range(num_inputs):
        inputs.append(rng.randn(*shape0).astype(np.float32))
    backward_function_tester(rng, F.mul_n,
                             inputs=inputs,
                             func_args=[], func_kwargs={},
                             atol_accum=5e-2,
                             dstep=1e-3,
                             ctx=ctx)
