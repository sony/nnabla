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

ctxs = list_context('RSubScalar')


def ref_r_sub_scalar(x, val):
    return val - x


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [315])
@pytest.mark.parametrize("val", [0.5, 1, 2])
def test_r_sub_scalar_forward_backward(seed, val, ctx, func_name):
    from nbla_test_utils import cap_ignore_region, function_tester
    rng = np.random.RandomState(seed)
    inputs = [rng.randn(2, 3, 4).astype(np.float32)]
    function_tester(rng, F.r_sub_scalar, ref_r_sub_scalar,
                    inputs, func_args=[val], ctx=ctx, func_name=func_name)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("val", [0.5, 1, 2])
def test_r_sub_scalar_double_backward(seed, val, ctx, func_name):
    from nbla_test_utils import backward_function_tester, cap_ignore_region
    rng = np.random.RandomState(seed)
    inputs = [
        cap_ignore_region(
            rng.randn(2, 3).astype(np.float32) * 3, (-0.5, 0.5))]
    backward_function_tester(rng, F.r_sub_scalar,
                             inputs=inputs,
                             func_args=[val], func_kwargs={},
                             atol_accum=1e-3,
                             dstep=1e-3,
                             ctx=ctx)
