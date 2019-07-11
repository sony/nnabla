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

from six.moves import range
import pytest
import numpy as np
import nnabla as nn
import nnabla.functions as F
from nbla_test_utils import list_context

ctxs = list_context('Dropout')


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("p", [p / 10. for p in range(1, 9)])
def test_dropout_forward_backward(p, seed, ctx, func_name):
    from nbla_test_utils import cap_ignore_region, function_tester
    rng = np.random.RandomState(seed)
    inputs = [
        cap_ignore_region(
            rng.randn(2, 3, 4).astype(np.float32) * 2,
            (-1e-3, 1e-3))]  # Ensure there is no zero.
    i = nn.Variable(inputs[0].shape, need_grad=True)
    i.d = inputs[0]
    # NNabla forward
    with nn.context_scope(ctx), nn.auto_forward():
        o = F.dropout(i, p)
    scale = 1. / (1. - p)
    mask = o.d != 0
    assert np.allclose(o.d, i.d * mask * scale)
    assert o.parent.name == func_name

    # NNabla backward
    orig_grad = rng.randn(*i.shape).astype(i.data.dtype)
    i.g[...] = orig_grad
    o_grad = rng.randn(*i.shape).astype(i.data.dtype)
    o.backward(o_grad)
    ref_grad = o_grad * mask * scale

    # Verify
    assert np.allclose(i.g, orig_grad + ref_grad)

    # Check if accum option works
    i.g[...] = 1
    o.g = o_grad
    o.parent.backward([i], [o], [False])
    assert np.allclose(i.g, ref_grad)

    # Check accum=False with NaN gradient
    i.g = np.float32('nan')
    o.parent.backward([i], [o], [False])
    assert not np.any(np.isnan(i.g))

    # Check if need_grad works
    i.g[...] = 0
    i.need_grad = False
    o.backward(o_grad)
    assert np.all(i.g == 0)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("p", [p / 10. for p in range(1, 9)])
def test_dropout_double_backward(p, seed, ctx, func_name):
    from nbla_test_utils import cap_ignore_region, backward_function_tester

    rng = np.random.RandomState(seed)
    inpd = cap_ignore_region(
        rng.randn(2, 3, 4).astype(np.float32) * 2,
        (-1e-3, 1e-3))  # Ensure there is no zero.
    inp = nn.Variable.from_numpy_array(inpd).apply(need_grad=True)
    # ONLY test the double backward
    with nn.context_scope(ctx):
        dout = F.dropout(inp, p, seed)
        out = F.sigmoid(dout)

    # Check gradient w.r.t. dy only since no backward w.r.t. x
    grads = nn.grad([out], [inp])
    grad = grads[0]
    grad.forward()
    grad.backward(1.0, clear_buffer=False)
    g_dy = grad.parent.inputs[1].g
    scale = 1. / (1. - p)
    mask = dout.d != 0
    assert np.allclose(g_dy, mask * scale)
