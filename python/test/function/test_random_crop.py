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
import itertools
import numpy as np
from scipy.stats import pearsonr
import nnabla as nn
import nnabla.functions as F
from nbla_test_utils import list_context
from nnabla.testing import assert_allclose

ctxs = list_context('RandomCrop')


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("inshape", [(3, 4, 5)])
@pytest.mark.parametrize("shape", [None, (3, 2, 2,), (1, 3, 3)])
def test_random_crop_forward_backward(seed, inshape, shape, ctx, func_name):
    from nbla_test_utils import function_tester
    rng = np.random.RandomState(seed)
    inputs = [rng.randn(*inshape).astype(np.float32)]
    i = nn.Variable(inputs[0].shape, need_grad=True)
    i.d = inputs[0]
    # NNabla forward
    with nn.context_scope(ctx), nn.auto_forward():
        o = F.random_crop(i, shape, 0, seed)
    if shape is not None:
        max_correl = 0
        possible_crop_range = [
            input - output for output, input in zip(shape, inshape)]
        for crop_pos in itertools.product(*map(tuple, map(lambda x: range(*x), [(0, r + 1) for r in possible_crop_range]))):
            r = inputs[0][crop_pos[0]:crop_pos[0] + shape[0], crop_pos[1]:crop_pos[1] + shape[1], crop_pos[2]:crop_pos[2] + shape[2]]
            assert(o.d.shape == r.shape)
            correl_and_p = pearsonr(o.d.flatten(), r.flatten())
            if correl_and_p[0] > max_correl:
                max_correl = correl_and_p[0]
    else:
        max_correl = pearsonr(o.d.flatten(), inputs[0].flatten())[0]

    np.testing.assert_almost_equal(max_correl, 1.0)

    assert o.parent.name == func_name

    # Skipping Backward check
    g = np.random.randn(*i.shape)
    i.g = g
    o_grad = np.random.randn(*o.shape)
    o.g = o_grad
    o.parent.backward([i], [o])
    ref_grad = i.g.copy() - g

    # Check accum=False with NaN gradient
    i.g = np.float32('nan')
    o.parent.backward([i], [o], [False])
    assert not np.any(np.isnan(i.g))

    # Check if accum option works
    i.g[...] = 1
    o.g = o_grad
    o.parent.backward([i], [o], [False])
    assert_allclose(i.g, ref_grad, atol=1e-6)

    # Check if need_grad works
    i.g[...] = 0
    i.need_grad = False
    o_diff = rng.randn(*o.shape).astype(i.d.dtype)
    o.backward(o_diff)
    assert np.all(i.g == 0)
