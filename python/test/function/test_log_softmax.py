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


def ref_log_softmax(x, axis):
    x = x - x.max(axis, keepdims=True)
    x = x - np.log(np.exp(x).sum(axis, keepdims=True))
    return x


@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("axis", [0, 1, 2, -1, -2, -3])
@pytest.mark.parametrize("ctx, func_name", list_context('LogSoftmax'))
def test_log_softmax_forward_backward(seed, axis, ctx, func_name):
    from nbla_test_utils import function_tester
    rng = np.random.RandomState(seed)
    inputs = [rng.randn(2, 3, 4).astype(np.float32)]
    function_tester(rng, F.log_softmax, ref_log_softmax, inputs, func_args=[axis],
                    ctx=ctx, func_name=func_name, atol_b=1e-2)


@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("axis", [0, 1, 2, -1, -2, -3])
@pytest.mark.parametrize("ctx, func_name", list_context('LogSoftmax'))
def test_log_softmax_double_backward(seed, axis, ctx, func_name):
    from nbla_test_utils import backward_function_tester
    rng = np.random.RandomState(seed)
    inputs = [rng.randn(2, 3, 4).astype(np.float32)]
    backward_function_tester(rng, F.log_softmax, inputs,
                             func_args=[axis], ctx=ctx)
