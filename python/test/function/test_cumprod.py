# Copyright 2021 Sony Group Corporation.
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
from test_cumsum import ref_cumsum
from refs import cumsum as ref_cumsum
from refs import cumprod as ref_cumprod


@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("axis", [0, 1, -1])
@pytest.mark.parametrize("exclusive", [True, False])
@pytest.mark.parametrize("reverse", [True, False])
# A flag whether input has zero or not.
@pytest.mark.parametrize("with_mask", [True, False])
@pytest.mark.parametrize("ctx, func_name", list_context('CumProd'))
def test_cumprod_forward_backward(seed, axis, exclusive, reverse, with_mask, ctx, func_name):
    from nbla_test_utils import function_tester
    rng = np.random.RandomState(seed)
    inputs = [(rng.randn(5, 7, 8)).astype(np.float32)]
    if with_mask:
        masks = [np.random.choice(2, i.size).reshape(i.shape) for i in inputs]
        inputs = [i*m for i, m in zip(inputs, masks)]

    # Half test is disabled since CumProd for fp16 is not implemented currently.
    function_tester(rng, F.cumprod, ref_cumprod, inputs, func_args=[axis, exclusive, reverse],
                    ctx=ctx, func_name=func_name, atol_b=5e-3, disable_half_test=True)


# Calculate cumprod backward in O(N).
# See `nnabla/src/nbla/function/generic/cumprod.cpp` for the detail algorithm.
def ref_grad_cumprod(x, dy, axis, exclusive, reverse, **kw):
    zero_mask = np.where(x == 0, 1, 0)

    # Create masks for following calculation
    # [1, ... , 1, 0, 0, ... , 0]
    before_first_zero_mask = np.where(ref_cumsum(
        zero_mask, axis, False, reverse) == 0, 1, 0)
    # [0, ... , 0, 0, 1, ... , 1]
    after_first_zero_mask = np.where(ref_cumsum(
        zero_mask, axis, True, reverse) == 0, 0, 1)
    # [0, ... , 0, 1, 0, ... , 0]
    first_zero_mask = np.where(
        before_first_zero_mask + after_first_zero_mask == 0, 1, 0)

    masked_x = np.where(first_zero_mask == 1, 1, x)

    # Check if masks are correctly generated.
    assert(np.all(first_zero_mask + before_first_zero_mask +
           after_first_zero_mask == 1))
    assert(np.all(masked_x - x == first_zero_mask))
    assert(np.all(x * first_zero_mask == 0))

    cumprod = ref_cumprod(x, axis, exclusive, reverse)
    masked_cumprod = ref_cumprod(masked_x, axis, exclusive, reverse)

    cumprod_dy = cumprod * dy
    masked_cumprod_dy = masked_cumprod * dy

    reversed_cumprod_dy = ref_cumsum(cumprod_dy, axis, exclusive, not reverse)
    reversed_masked_cumprod_dy = ref_cumsum(
        masked_cumprod_dy, axis, exclusive, not reverse)

    # Calculate dx
    full_masked_x = np.where(x == 0, 1, x)  # Avoid generating `nan`
    dx_before_zero_pos = reversed_cumprod_dy / \
        full_masked_x * before_first_zero_mask
    dx_zero_pos = reversed_masked_cumprod_dy * first_zero_mask
    dx = dx_before_zero_pos + dx_zero_pos
    return dx.flatten()


@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("axis", [0, 1, -1])
@pytest.mark.parametrize("shape", [(12, 34, 56), (12345, 5, 7), (3, 12345, 7), (3, 5, 12345)])
@pytest.mark.parametrize("exclusive", [True, False])
@pytest.mark.parametrize("reverse", [True, False])
# A flag whether input has zero or not.
@pytest.mark.parametrize("with_mask", [True, False])
@pytest.mark.parametrize("ctx, func_name", list_context('CumProd'))
def test_cumprod_forward_backward_large(seed, shape, axis, exclusive, reverse, with_mask, ctx, func_name):
    """ Test for large input cases.
    Instead of numerical gradient calculation, `ref_grad_cumuprod` is used here because calculation of numerical gradient takes long time with large input.
    """
    from nbla_test_utils import function_tester
    rng = np.random.RandomState(seed)
    inputs = [(rng.randn(*shape)).astype(np.float32)]
    if with_mask:
        masks = [np.random.choice(2, i.size).reshape(i.shape) for i in inputs]
        inputs = [i*m for i, m in zip(inputs, masks)]

    # Half test is disabled since CumProd for fp16 is not implemented currently.
    function_tester(rng, F.cumprod, ref_cumprod, inputs, ref_grad=ref_grad_cumprod, func_args=[axis, exclusive, reverse],
                    ctx=ctx, func_name=func_name, disable_half_test=True)
