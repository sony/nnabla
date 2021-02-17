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


from nbla_test_utils import (function_tester, list_context)


def ref_norm(x, p, axis, keepdims):
    if p is None:
        p = 2.0
    x = np.abs(x)
    x = np.power(x, p)
    x = np.sum(x, axis, keepdims=keepdims)
    x = np.power(x, 1./p)
    return x


@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("p", [None, 1.0, 1.3])  # None -> 2.0
@pytest.mark.parametrize("axis", [None, 0, 1, 2, 3, -1, -2, (0, 2), (1, 2, 3), (-1, -2)])
@pytest.mark.parametrize("keepdims", [False, True])
@pytest.mark.parametrize("inshape", [(2, 1, 4, 5), (3, 1, 4, 5)])
@pytest.mark.parametrize("ctx, func_name", list_context('Norm'))
def test_norm_forward_backward(seed, p, axis, keepdims, inshape, ctx, func_name):
    func = F.norm
    ref_func = ref_norm
    rng = np.random.RandomState(seed)
    inputs = [rng.randn(*inshape).astype(np.float32)]
    function_tester(rng, func, ref_func, inputs,
                    func_args=[p, axis],
                    func_kwargs=dict(keepdims=keepdims),
                    ctx=ctx, func_name=func_name,
                    atol_b=6e-3)
