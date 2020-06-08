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

ctxs = list_context('ClipGradByValue')


def ref_clip_grad_by_value(x, min_, max_):
    y = np.copy(x)
    return y


def ref_grad_clip_grad_by_value(x, min_, max_, dy, **kw):
    dx = dy
    idx_min = np.where(dy < min_)
    idx_max = np.where(dy > max_)

    dx[idx_min] = min_[idx_min]
    dx[idx_max] = max_[idx_max]
    return dx.flatten()


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
def test_clip_grad_by_value_forward_backward(seed, ctx, func_name):
    from nbla_test_utils import cap_ignore_region, function_tester
    rng = np.random.RandomState(seed)
    inputs = [
        cap_ignore_region(
            rng.randn(2, 3, 4).astype(np.float32) * 2,
            (-1e-3, 1e-3)),
        cap_ignore_region(
            rng.randn(2, 3, 4).astype(np.float32) * 2,
            (-1e-3, 1e-3)),
        cap_ignore_region(
            rng.randn(2, 3, 4).astype(np.float32) * 2,
            (-1e-3, 1e-3))
    ]
    func_args = []
    function_tester(rng, F.clip_grad_by_value, ref_clip_grad_by_value, inputs,
                    func_args=func_args, backward=[True, False, False],
                    ctx=ctx, func_name=func_name,
                    ref_grad=ref_grad_clip_grad_by_value)
