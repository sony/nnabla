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
from scipy.ndimage.interpolation import shift as scipy_shift
from scipy.stats import pearsonr
import nnabla as nn
import nnabla.functions as F
from nbla_test_utils import list_context
from nnabla.testing import assert_allclose

ctxs = list_context('RandomShift')


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("inshape", [(7, 8, 9)])
@pytest.mark.parametrize("shifts", [None, (0, 0, 2,), (0, 2, 0), (0, 2, 2)])
@pytest.mark.parametrize("border_mode", ["nearest", "reflect", "constant"])
@pytest.mark.parametrize("constant_value", [0, -100])
def test_random_shift_forward_backward(seed, inshape, shifts, border_mode, constant_value, ctx, func_name):
    from nbla_test_utils import function_tester
    rng = np.random.RandomState(seed)
    inputs = [rng.randn(*inshape).astype(np.float32)]
    i = nn.Variable(inputs[0].shape, need_grad=True)
    i.d = inputs[0]
    # NNabla forward
    with nn.context_scope(ctx), nn.auto_forward():
        o = F.random_shift(i, shifts, border_mode, constant_value, 0, seed)
    result_shifts = (0, 0, 0)
    max_correl = 0
    for shift_amount in itertools.product(*map(tuple, map(lambda x: range(*x), [(-2, 3) for _ in range(len(inshape))]))):
        r = scipy_shift(inputs[0], shift_amount,
                        mode=border_mode, cval=constant_value)
        correl_and_p = pearsonr(o.d.flatten(), r.flatten())
        if correl_and_p[0] > max_correl:
            result_shifts = shift_amount
            max_correl = correl_and_p[0]
    ref = scipy_shift(inputs[0], result_shifts,
                      mode=border_mode, cval=constant_value)
    if shifts is None:
        shifts = (0,) * len(inputs[0].shape)
    for result, shift_range in zip(result_shifts, shifts):
        assert abs(result) <= shift_range

    assert_allclose(o.d, ref)
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
    o_grad = rng.randn(*i.shape).astype(i.data.dtype)
    o.backward(o_grad)
    assert np.all(i.g == 0)
