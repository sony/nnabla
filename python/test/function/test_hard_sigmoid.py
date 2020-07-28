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

ctxs = list_context('HardSigmoid')


def ref_hard_sigmoid(x):
    return np.maximum(0, np.minimum(1, x*0.2 + 0.5))


def ref_hard_sigmoid_backward(x, dy, **kw):
    return np.array([dy*0.2 if 2.5 >= i >= -2.5 else 0 for i in np.nditer(x)])


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
def test_hard_sigmoid_forward_backward(seed, ctx, func_name):
    from nbla_test_utils import cap_ignore_region, function_tester
    rng = np.random.RandomState(seed)
    inputs = [
        np.clip(np.abs(rng.randn(2, 3, 4).astype(np.float32)) * 1e4, 1e-2, 1e4)]
    function_tester(rng, F.hard_sigmoid, ref_hard_sigmoid, inputs,
                    ctx=ctx, func_name=func_name, ref_grad=ref_hard_sigmoid_backward)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
def test_hard_sigmoid_double_backward(seed, ctx, func_name):
    from nbla_test_utils import backward_function_tester
    rng = np.random.RandomState(seed)
    inputs = [rng.randn(2, 3, 4).astype(np.float32) * 2]
    backward_function_tester(rng, F.hard_sigmoid,
                             inputs=inputs,
                             func_args=[], func_kwargs={},
                             atol_accum=1e-3,
                             dstep=1e-3,
                             ctx=ctx)
