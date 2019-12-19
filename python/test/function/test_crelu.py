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

ctxs = list_context('CReLU')


def ref_crelu(x, axis):
    return np.concatenate([np.maximum(x, 0), np.maximum(-x, 0)], axis=axis)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("seed", [313])
def test_crelu_forward_backward(seed, axis, ctx, func_name):
    from nbla_test_utils import cap_ignore_region, function_tester
    rng = np.random.RandomState(seed)
    inputs = [
        cap_ignore_region(
            rng.randn(2, 3, 4).astype(np.float32) * 2,
            (-1e-3, 1e-3))]
    function_tester(rng, F.crelu, ref_crelu, inputs, func_args=[axis],
                    ctx=ctx, func_name=func_name, atol_b=1e-2)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("seed", [313])
def test_crelu_double_backward(seed, axis, ctx, func_name):
    from nbla_test_utils import cap_ignore_region, backward_function_tester
    rng = np.random.RandomState(seed)
    inputs = [
        cap_ignore_region(
            rng.randn(2, 3, 4).astype(np.float32) * 2,
            (-1e-3, 1e-3))]
    backward_function_tester(rng, F.crelu, None, inputs, func_args=[axis],
                             ctx=ctx, func_name=func_name, atol_b=1e-3, atol_accum=1e-3)
