# Copyright 2020,2021 Sony Corporation.
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

ctxs = list_context('Mish')


def ref_mish(x):
    return x*np.tanh(np.log(np.exp(x, dtype=x.dtype)+1))


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
def test_mish_forward_backward(seed, ctx, func_name):
    from nbla_test_utils import function_tester
    rng = np.random.RandomState(seed)
    inputs = [rng.randn(2, 3, 4).astype(np.float32) * 1]
    function_tester(rng, F.mish, ref_mish, inputs,
                    ctx=ctx, func_name=func_name,
                    atol_b=1e-3, atol_accum=1e-3)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
def test_mish_forward_backward_with_reset(seed, ctx, func_name):
    from nbla_test_utils import function_tester
    rng = np.random.RandomState(seed)
    inputs = [rng.randn(2, 3, 4).astype(np.float32) * 1]
    reset_inputs = [rng.randn(3, 2, 4).astype(np.float32) * 1]
    function_tester(rng, F.mish, ref_mish, inputs,
                    ctx=ctx, func_name=func_name,
                    atol_b=1e-3, atol_accum=1e-3,
                    reset_inputs=reset_inputs)
