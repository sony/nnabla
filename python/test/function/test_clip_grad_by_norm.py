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

ctxs = list_context('ClipGradByNorm')


def ref_clip_grad_by_norm(x, clip_norm, axes):
    y = np.copy(x)
    return y


def ref_grad_clip_by_norm(x, dy, clip_norm, axes, **kw):
    dx = np.copy(dy)
    dx = clip_norm * dy / \
        np.broadcast_to(
            np.sqrt(np.sum(dy**2, axis=axes, keepdims=True)), dy.shape)
    return dx.flatten()


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("clip_norm", [-5, 0.5])
@pytest.mark.parametrize("axes", [(), (0, 1), (1, ), (0, 2, 3), (2, 3)])
def test_clip_by_norm_forward_backward(seed, ctx, func_name, clip_norm, axes):
    from nbla_test_utils import cap_ignore_region, function_tester
    rng = np.random.RandomState(seed)
    inputs = [rng.randn(2, 3, 4, 4).astype(np.float32) * 2]
    func_args = [clip_norm, axes]
    function_tester(rng, F.clip_grad_by_norm, ref_clip_grad_by_norm, inputs,
                    func_args=func_args, backward=[True],
                    ctx=ctx, func_name=func_name,
                    ref_grad=ref_grad_clip_by_norm)
