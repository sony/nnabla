# Copyright 2021 Sony Corporation.
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
from refs import cumsum as ref_cumsum


@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("axis", [0, 1, 2, -1])
@pytest.mark.parametrize("exclusive", [True, False])
@pytest.mark.parametrize("reverse", [True, False])
@pytest.mark.parametrize("ctx, func_name", list_context('CumSum'))
def test_cumsum_forward_backward(seed, axis, exclusive, reverse, ctx, func_name):
    from nbla_test_utils import function_tester
    rng = np.random.RandomState(seed)
    inputs = [(rng.randn(5, 3, 4)).astype(np.float32)]
    function_tester(rng, F.cumsum, ref_cumsum, inputs, func_args=[axis, exclusive, reverse],
                    ctx=ctx, func_name=func_name, atol_b=3e-3, disable_half_test=True)


@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("axis", [0, 1, 2, -1])
@pytest.mark.parametrize("exclusive", [True, False])
@pytest.mark.parametrize("reverse", [True, False])
@pytest.mark.parametrize("ctx, func_name", list_context('CumSum'))
def test_cumsum_double_backward(seed, axis, exclusive, reverse, ctx, func_name):
    from nbla_test_utils import backward_function_tester
    rng = np.random.RandomState(seed)
    inputs = [(rng.randn(5, 3, 4)).astype(np.float32)]
    backward_function_tester(rng, F.cumsum, inputs=inputs, func_args=[axis, exclusive, reverse],
                             ctx=ctx, atol_b=3e-3)
