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

ctxs = list_context('BinaryError')


def ref_binary_error(x, l):
    y = []
    for x_, l_ in zip(x, l):
        y.append((x_ >= 0.5) != (l_ >= 0.5))
    return np.array(y).reshape(x.shape)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
def test_binary_error_forward(seed, ctx, func_name):
    ishape = [5, 6, 7]
    rng = np.random.RandomState(seed)

    inputs = [
        rng.rand(5, 6, 7).astype(np.float32),
        rng.rand(5, 6, 7).astype(np.float32)]

    ref = ref_binary_error(inputs[0], inputs[1])

    x = nn.Variable(ishape)
    l = nn.Variable(ishape)
    y = F.binary_error(x, l)
    x.d = inputs[0]
    l.d = inputs[1]
    y.forward()
    res = y.d

    atol_f = 1e-6
    assert_allclose(ref, res, atol=atol_f)
