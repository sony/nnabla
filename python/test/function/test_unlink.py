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

ctxs = list_context('Unlink')


def ref_unlink(x):
    return x


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("shape", [(3, 6), (3, 7, 13)])
def test_unlink_forward_backward(seed, shape, ctx, func_name):

    rng = np.random.RandomState(seed)
    x0 = rng.randn(*(shape)).astype(np.float32)
    ref = ref_unlink(x0)

    x = nn.Variable(shape)
    y = F.unlink(x)
    x.d = x0
    x.grad.zero()
    y.forward()
    y.backward()
    res = y.d

    atol_f = 1e-6
    assert_allclose(ref, res, atol=atol_f)

    atol_b = 1e-6
    ref = np.zeros(shape)
    res = x.g
    assert_allclose(ref, res, atol=atol_b)
