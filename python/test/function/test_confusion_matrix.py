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

ctxs = list_context('ConfusionMatrix')


def ref_confusion_matrix(x, l, axis):
    orig_x = x.copy()
    x = np.rollaxis(x, axis, x.ndim).reshape(-1, x.shape[axis])
    ll = np.rollaxis(l, axis, x.ndim).flatten()
    y = np.zeros((orig_x.shape[axis], orig_x.shape[axis]), np.int)
    for x_, ll_ in zip(x, ll):
        index = -1
        for i, x__ in enumerate(x_):
            if x__ >= x_[index]:
                index = i
        y[ll_][index] += 1
    return y


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("axis", [0, 1, 2, -1, -2, -3])
def test_confusion_matrix_forward(seed, ctx, axis, func_name):
    ishape = [5, 6, 7]
    rng = np.random.RandomState(seed)

    l_shape = list(ishape)
    l_shape[axis] = 1
    n_class = ishape[axis]

    inputs = [
        rng.rand(5, 6, 7).astype(np.float32),
        rng.randint(0, n_class, size=l_shape).astype(np.int)]

    ref = ref_confusion_matrix(inputs[0], inputs[1], axis)

    x = nn.Variable(ishape)
    l = nn.Variable(l_shape)
    y = F.confusion_matrix(x, l, axis)
    x.d = inputs[0]
    l.d = inputs[1]
    y.forward()
    res = y.d

    atol_f = 1e-6
    assert_allclose(ref, res, atol=atol_f)
