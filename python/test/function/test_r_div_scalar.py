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

ctxs = list_context('RDivScalar')


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [314])
@pytest.mark.parametrize("val", [0.5, 1, 2])
def test_r_div_scalar_forward_backward(seed, val, ctx, func_name):
    from nbla_test_utils import function_tester, cap_ignore_region
    rng = np.random.RandomState(seed)
    inputs = [
        cap_ignore_region(
            rng.randn(2, 3, 4).astype(np.float32) * 3, (-0.5, 0.5))]
    function_tester(rng, F.r_div_scalar, lambda x, y: y / x, inputs,
                    func_args=[val], dstep=1e-4, atol_b=1e-1,
                    ctx=ctx, func_name=func_name)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("val", [0.5, 1, 2])
def test_r_div_scalar_double_backward(seed, val, ctx, func_name):
    from nbla_test_utils import backward_function_tester, cap_ignore_region
    rng = np.random.RandomState(seed)
    inputs = [rng.randn(2, 3).astype(np.float32) * 10]
    backward_function_tester(rng, F.r_div_scalar,
                             inputs=inputs,
                             func_args=[val], func_kwargs={},
                             atol_accum=4e-2,
                             dstep=1e-3,
                             ctx=ctx)
