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
from nbla_test_utils import list_context

ctxs = list_context('Split')


def ref_split(x, axis):
    return tuple([np.squeeze(sq, axis) for sq in np.split(x, x.shape[axis], axis=axis)])


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("axis", [0, 1, 2, -1, -2, -3])
@pytest.mark.parametrize("seed", [313])
def test_split_forward_backward(seed, axis, ctx, func_name):
    from nbla_test_utils import function_tester
    rng = np.random.RandomState(seed)
    shape = [2, 3, 4]
    x = rng.randn(*shape).astype(np.float32)
    inputs = [x]
    function_tester(rng, F.split, ref_split, inputs,
                    func_args=[axis], ctx=ctx, func_name=func_name,
                    atol_b=1e-2)


def test_zero_value():
    a = nn.Variable((1, 6, 0))
    with pytest.raises(RuntimeError):
        F.split(a, axis=1)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("axis", [0, 1, 2, -1, -2, -3])
@pytest.mark.parametrize("seed", [313])
def test_split_double_backward(seed, axis, ctx, func_name):
    from nbla_test_utils import backward_function_tester
    rng = np.random.RandomState(seed)
    shape = [2, 3, 4]
    x = rng.randn(*shape).astype(np.float32)
    inputs = [x]
    backward_function_tester(rng, F.split,
                             inputs=inputs,
                             func_args=[axis], func_kwargs={},
                             atol_f=1e-3,
                             atol_accum=5e-3,
                             dstep=1e-2,
                             ctx=ctx)
