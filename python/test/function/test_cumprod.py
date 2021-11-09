# Copyright 2021 Sony Corporation.
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
from refs import cumprod as ref_cumprod
from refs import cumprod_backward as ref_grad_cumprod


def create_cumprod_input(rng, shape, axis, with_mask, with_random_zero_pos, zero_pos):
    x = (rng.randn(*shape)).astype(np.float32)
    if with_mask:
        if with_random_zero_pos:
            # Make zero elements with the probability of `1 / x_shape[axis]`.
            # It is the probability of existence of one zero element in each scan axis.
            mask = rng.rand(*shape) > (1.0 / shape[axis])
            x = x * mask
        else:
            x.swapaxes(0, axis)[zero_pos] = 0
    return x


@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("shape", [(5, 7, 8)])
@pytest.mark.parametrize("axis", [0, 1, -1])
@pytest.mark.parametrize("exclusive", [True, False])
@pytest.mark.parametrize("reverse", [True, False])
# A flag whether input has zero or not.
@pytest.mark.parametrize("with_mask, with_random_zero_pos, zero_pos",
                         [(False, False, 0),
                          (True, False, 0),  # First element
                          (True, False, 2),  # Middle element
                          (True, False, -1),  # Last element
                          (True, True, 0),
                          ])
@pytest.mark.parametrize("ctx, func_name", list_context('CumProd'))
def test_cumprod_forward_backward(seed, shape, axis, exclusive, reverse, with_mask, with_random_zero_pos, zero_pos, ctx, func_name):
    from nbla_test_utils import function_tester
    rng = np.random.RandomState(seed)
    input = create_cumprod_input(
        rng, shape, axis, with_mask, with_random_zero_pos, zero_pos)

    # Half test is disabled since CumProd for fp16 is not implemented currently.
    function_tester(rng, F.cumprod, ref_cumprod, [input], func_args=[axis, exclusive, reverse],
                    ctx=ctx, func_name=func_name, atol_b=6e-3, disable_half_test=True)


@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("shape", [(12, 34, 56), (12345, 5, 7), (3, 12345, 7), (3, 5, 12345)])
@pytest.mark.parametrize("axis", [0, 1, -1])
@pytest.mark.parametrize("exclusive", [True, False])
@pytest.mark.parametrize("reverse", [True, False])
# with_mask: A flag whether input has zero or not.
# with_random_zero_pos: Whether the zero positions are determined randomly.
# zero_pos: Zero position on scan axis when `with_random_zero_pos == False`.
@pytest.mark.parametrize("with_mask, with_random_zero_pos, zero_pos",
                         [(False, False, 0),
                          (True, False, 0),  # First element
                          (True, False, 2),  # Middle element
                          (True, False, -1),  # Last element
                          (True, True, 0),
                          ])
@pytest.mark.parametrize("ctx, func_name", list_context('CumProd'))
def test_cumprod_forward_backward_large(seed, shape, axis, exclusive, reverse, with_mask, with_random_zero_pos, zero_pos, ctx, func_name):
    """ Test for large input cases.
    Instead of numerical gradient calculation, `ref_grad_cumuprod` is used here because calculation of numerical gradient takes long time with large input.
    """
    from nbla_test_utils import function_tester
    rng = np.random.RandomState(seed)
    input = create_cumprod_input(
        rng, shape, axis, with_mask, with_random_zero_pos, zero_pos)

    # Half test is disabled since CumProd for fp16 is not implemented currently.
    function_tester(rng, F.cumprod, ref_cumprod, [input], ref_grad=ref_grad_cumprod, func_args=[axis, exclusive, reverse],
                    ctx=ctx, func_name=func_name, disable_half_test=True)
