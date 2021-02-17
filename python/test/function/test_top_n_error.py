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
from nbla_test_utils import list_context
from nnabla.testing import assert_allclose

ctxs = list_context('TopNError')


def ref_top_n_error(x, l, axis, n):
    orig_x = x.copy()
    x = np.rollaxis(x, axis, x.ndim).reshape(-1, x.shape[axis])
    ll = np.rollaxis(l, axis, x.ndim).flatten()
    y = []
    for x_, ll_ in zip(x, ll):
        threshold = x_[ll_]
        count = 0
        for x__ in x_:
            if x__ >= threshold:
                count += 1
        y.append(1 if count > n else 0)
    return np.array(y).reshape(l.shape)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("axis", [0, 1, 2, -1, -2, -3])
@pytest.mark.parametrize("n", [3, 5])
def test_top_n_error_forward(seed, axis, n, ctx, func_name):
    ishape = [5, 6, 7]
    rng = np.random.RandomState(seed)

    l_shape = list(ishape)
    l_shape[axis] = 1
    n_class = ishape[axis]

    inputs = [
        rng.rand(5, 6, 7).astype(np.float32) * 0.9 + 0.05,
        rng.randint(0, n_class, size=l_shape).astype(np.int)]

    ref = ref_top_n_error(inputs[0], inputs[1], axis, n)

    x = nn.Variable(ishape)
    l = nn.Variable(l_shape)
    y = F.top_n_error(x, l, axis, n)
    x.d = inputs[0]
    l.d = inputs[1]
    y.forward()
    res = y.d

    atol_f = 1e-6
    assert_allclose(ref, res, atol=atol_f)
