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
import nnabla as nn
import nnabla.functions as F
from test_flip import ref_flip
from nbla_test_utils import list_context
from nnabla.testing import assert_allclose

ctxs = list_context('RandomFlip')


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("axes", [None, (1,), (2,)])
def test_random_flip_forward_backward(seed, axes, ctx, func_name):
    from nbla_test_utils import cap_ignore_region, function_tester
    rng = np.random.RandomState(seed)
    inputs = [rng.randn(2, 3, 4).astype(np.float32)]
    i = nn.Variable(inputs[0].shape, need_grad=True)
    i.d = inputs[0]
    # NNabla forward
    with nn.context_scope(ctx), nn.auto_forward():
        o = F.random_flip(i, axes, 0, seed)
    flip_close = np.allclose(o.d, ref_flip(inputs[0], axes))
    assert flip_close or (not flip_close and np.allclose(o.d, i.d))
    assert o.parent.name == func_name

    # NNabla backward
    orig_grad = rng.randn(*i.shape).astype(i.data.dtype)
    i.g[...] = orig_grad
    o_grad = rng.randn(*i.shape).astype(i.data.dtype)
    o.g = o_grad
    o.parent.backward([i], [o])

    # Verify
    if flip_close:
        ref_grad = ref_flip(o_grad, axes)
    else:
        ref_grad = o_grad
    assert_allclose(i.g, orig_grad + ref_grad)

    # Check if accum option works
    i.g[...] = 1
    o.g = o_grad
    o.parent.backward([i], [o], [False])
    assert_allclose(i.g, ref_grad)

    # Check accum=False with NaN gradient
    i.g = np.float32('nan')
    o.parent.backward([i], [o], [False])
    assert not np.any(np.isnan(i.g))

    # Check if need_grad works
    i.g[...] = 0
    i.need_grad = False
    o.backward(o_grad)
    assert np.all(i.g == 0)
