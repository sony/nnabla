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
from scipy.ndimage.interpolation import shift as scipy_shift

import nnabla.functions as F
from nbla_test_utils import list_context

ctxs = list_context('Shift')


def ref_shift(x, shifts, border_mode):
    if shifts is None:
        shifts = (0,) * len(x.shape)
    return scipy_shift(x, shifts, mode=border_mode, order=1)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("inshape", [(2, 3, 4)])
@pytest.mark.parametrize("shifts", [None, (0, 0, 2,), (0, -2, 1), (1, 0, -1)])
@pytest.mark.parametrize("border_mode", ["nearest", "reflect"])
def test_shift_forward_backward(seed, inshape, shifts, border_mode, ctx, func_name):
    from nbla_test_utils import function_tester
    rng = np.random.RandomState(seed)
    # Input
    inputs = [rng.randn(*inshape).astype(np.float32)]
    function_tester(rng, F.shift, ref_shift, inputs, func_args=[
                    shifts, border_mode], ctx=ctx, func_name=func_name, atol_f=1e-6, atol_b=1e-2)
