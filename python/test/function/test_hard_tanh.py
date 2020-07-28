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

ctxs = list_context('HardTanh')


def ref_hard_tanh(x):
    return np.maximum(-1, np.minimum(1, x))


def ref_hard_tanh_backward(x, dy, **kw):
    return np.array([dy if -1 <= i <= 1 else 0 for i in np.nditer(x)])


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
def test_hard_tanh_forward_backward(seed, ctx, func_name):
    from nbla_test_utils import cap_ignore_region, function_tester
    rng = np.random.RandomState(seed)
    inputs = [
        np.clip(np.abs(rng.randn(2, 3, 4).astype(np.float32)) * 1e4, 1e-2, 1e4)]
    function_tester(rng, F.hard_tanh, ref_hard_tanh, inputs,
                    ctx=ctx, func_name=func_name, ref_grad=ref_hard_tanh_backward)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
def test_hard_tanh_double_backward(seed, ctx, func_name):
    from nbla_test_utils import backward_function_tester, cap_ignore_region
    rng = np.random.RandomState(seed)
    inputs = [cap_ignore_region(
        rng.randn(2, 3).astype(np.float32), (-0.9, 0.9))]
    backward_function_tester(rng, F.hard_tanh,
                             inputs=inputs,
                             func_args=[], func_kwargs={},
                             atol_accum=1e-3,
                             dstep=1e-3,
                             ctx=ctx)
