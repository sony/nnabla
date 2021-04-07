# Copyright (c) 2020 Sony Corporation. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0
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


def ref_cumprod(x, axis, exclusive, reverse):

    if reverse:
        out = np.flip(np.cumprod(np.flip(x, axis=axis), axis=axis), axis=axis)
    else:
        out = np.cumprod(x, axis=axis)

    if exclusive:

        if axis < 0:
            axis += out.ndim

        shift_ = 1 if not reverse else -1
        out = np.roll(out, shift_, axis=axis)
        index = 0 if not reverse else -1
        if axis == 0:
            out[index, :, :] = 1.0
        elif axis == 1:
            out[:, index, :] = 1.0
        elif axis == 2:
            out[:, :, index] = 1.0
        else:
            raise NotImplementedError

    return out


@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("axis", [0, 1, 2, -1])
@pytest.mark.parametrize("exclusive", [True, False])
@pytest.mark.parametrize("reverse", [True, False])
@pytest.mark.parametrize("ctx, func_name", list_context('CumProd'))
def test_cumprod_forward_backward(seed, axis, exclusive, reverse, ctx, func_name):
    from nbla_test_utils import function_tester
    rng = np.random.RandomState(seed)
    inputs = [(rng.randn(5, 7, 8)).astype(np.float32)]
    function_tester(rng, F.cumprod, ref_cumprod, inputs, func_args=[axis, exclusive, reverse],
                    ctx=ctx, func_name=func_name, atol_b=4e-3, disable_half_test=True)
