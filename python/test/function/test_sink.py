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
from nnabla.testing import assert_allclose


@pytest.mark.parametrize("seed", [313])
def test_sink(seed):
    rng = np.random.RandomState(seed)
    v = nn.Variable((2, 3, 4), need_grad=True)
    h0 = F.tanh(v)
    h1 = F.sigmoid(v)
    v.d = rng.randn(*v.shape).astype(np.float32)

    # Create references
    v.grad.zero()
    h0.forward()
    h1.forward()
    h0.backward()
    h1.backward()  # v.grad is accumulated.
    h0d = h0.d.copy()
    h1d = h1.d.copy()
    vg = v.g.copy()

    # Reset values
    h0.data.zero()
    h1.data.zero()
    v.grad.zero()

    # Check if sink works
    dummy = F.sink(h0, h1, one_input_grad=True)
    dummy.forward()
    dummy.backward()
    assert np.all(h0d == h0.d)
    assert np.all(h1d == h1.d)
    assert np.all(vg == v.g)

    # Check if clear_buffer still keeps h0 an h1 even though they are not
    # leaf variables.
    # It's done by defining prohibit_clear_input_buffers function in sink.hpp.
    dummy = F.sink(h0, h1, one_input_grad=True)
    dummy.forward(clear_buffer=True)
    assert np.all(h0d == h0.d)
    assert np.all(h1d == h1.d)
    # Also checking backward when clear buffers.
    v.grad.zero()
    dummy = F.sink(h0, h1, one_input_grad=True)
    dummy.forward(clear_no_need_grad=True)
    dummy.backward(clear_buffer=True)
    assert np.all(h0d == h0.d)
    assert np.all(h1d == h1.d)
    assert np.all(vg == v.g)

    # Check if one_input_grad=False works
    dummy = F.sink(h0, h1, one_input_grad=False)
    g0 = rng.randn(*h0.shape).astype(np.float32)
    g1 = rng.randn(*h1.shape).astype(np.float32)
    h0.g = g0
    h1.g = g1
    dummy.forward()
    # Compute reference
    v.grad.zero()
    h0.backward(grad=g0)
    h1.backward(grad=g1)
    gv = v.g.copy()
    # Compute with sink
    v.grad.zero()
    dummy.backward()
    assert_allclose(v.g, gv)
