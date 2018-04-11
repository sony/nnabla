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

ctxs = list_context('TopKGrad')


def ref_top_k_grad(k, abs, base_axis, grad):
    outer_dim = np.prod(grad.shape[:base_axis], dtype=int)
    inner_dim = np.prod(grad.shape[base_axis:], dtype=int)
    gg = grad.reshape(outer_dim, inner_dim).copy()
    ix = np.argsort(np.abs(gg) if abs else gg)[:, -k:]
    dx = np.zeros((outer_dim, inner_dim))
    for idx, row in enumerate(ix):
        dx[idx, row] = gg[idx, row]
    dx = dx.squeeze(axis=0) if base_axis == 0 else dx
    return dx.reshape(grad.shape)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("k", [1, 5])
@pytest.mark.parametrize("abs", [False, True])
@pytest.mark.parametrize("base_axis", [1, 0, 2])
def test_forward_backward(seed, k, abs, base_axis, ctx, func_name):
    rng = np.random.RandomState(seed)

    x = nn.Variable.from_numpy_array(rng.randn(4, 5, 6), need_grad=True)
    y = F.top_k_grad(x, k, abs, base_axis)
    g = rng.randn(*x.shape)
    ref = ref_top_k_grad(k, abs, base_axis, g)

    y.forward()
    assert np.allclose(x.d, y.d)

    x.grad.zero()
    y.backward(g)
    assert np.allclose(ref, x.g)

    x.grad.fill(0.5)
    y.backward(g)
    assert np.allclose(ref + 0.5, x.g)
