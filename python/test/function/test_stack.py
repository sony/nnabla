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

ctxs = list_context('Stack')


def ref_stack(*inputs, **params):
    axis = params['axis']
    return np.stack(inputs, axis=axis)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("axis", [0, 1, 2, -1, -2, -3])
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("num_inputs", [2, 3])
def test_stack_forward_backward(seed, axis, num_inputs, ctx, func_name):
    from nbla_test_utils import cap_ignore_region, function_tester
    rng = np.random.RandomState(seed)
    shape = [2, 3, 4]
    inputs = [rng.randn(*shape).astype(np.float32) for x in range(num_inputs)]
    function_tester(rng, F.stack, ref_stack, inputs,
                    func_kwargs=dict(axis=axis), ctx=ctx, func_name=func_name,
                    atol_b=2e-3)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("axis", [0, 1, 2, -1, -2, -3])
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("num_inputs", [2, 3])
def test_stack_double_backward(seed, axis, num_inputs, ctx, func_name):
    from nbla_test_utils import backward_function_tester
    rng = np.random.RandomState(seed)
    shape = [2, 3, 4]
    inputs = [rng.randn(*shape).astype(np.float32) for x in range(num_inputs)]
    backward_function_tester(rng, F.stack,
                             inputs=inputs,
                             func_args=[], func_kwargs=dict(axis=axis),
                             atol_accum=1e-3,
                             dstep=1e-3,
                             ctx=ctx)
