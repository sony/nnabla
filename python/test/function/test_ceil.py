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

ctxs = list_context('Ceil')


def ref_ceil(x):
    return np.ceil(x)


def ref_grad_ceil(x, dy):
    return dy.flatten()


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
def test_ceil_forward_backward(seed,
                               ctx, func_name):
    from nbla_test_utils import cap_ignore_region, \
        function_tester
    rng = np.random.RandomState(seed)
    inputs = [rng.randn(2, 3, 4).astype(np.float32) * 2]

    function_tester(rng, F.ceil,
                    ref_ceil,
                    inputs,
                    atol_b=1e-3, backward=[True],
                    ctx=ctx, func_name=func_name,
                    ref_grad=ref_grad_ceil)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
def test_ceil_double_backward(seed,
                              ctx, func_name):
    from nbla_test_utils import cap_ignore_region, backward_function_tester
    rng = np.random.RandomState(seed)
    inputs = [rng.randn(2, 3, 4).astype(np.float32) * 2]

    backward_function_tester(rng, F.ceil,
                             None,
                             inputs,
                             atol_b=5e-1, atol_accum=5e-1, backward=[True],
                             ctx=ctx, func_name=func_name)
