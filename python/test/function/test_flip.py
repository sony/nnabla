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

ctxs = list_context('Flip')


def ref_flip(x, axes):
    if axes is None:
        axes = [len(x.shape) - 1]
    # x = x.copy()
    # for axis in axes:
    #     if axis == 0:
    #         x = x[::-1]
    #     elif axis == 1:
    #         x = x[::, ::-1]
    #     elif axis == 2:
    #         x = x[::, ::, ::-1]
    #     else:
    #         raise
    return np.flip(x, axes)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("inshape", [(2, 3, 4)])
@pytest.mark.parametrize("axes", [None, (0,), (0, 1), (2, 0), (-1, -2), (0, -1), (-1,)])
def test_flip_forward_backward(seed, inshape, axes, ctx, func_name):
    from nbla_test_utils import function_tester
    rng = np.random.RandomState(seed)
    # Input
    inputs = [rng.randn(*inshape).astype(np.float32)]
    function_tester(rng, F.flip, ref_flip, inputs, func_args=[
                    axes], ctx=ctx, func_name=func_name, atol_f=1e-6, atol_b=1e-2)
