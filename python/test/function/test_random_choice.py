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

ctxs = list_context('RandomChoice')


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [-1, 313, 999])
def test_random_choice_with_replacement(ctx, func_name, seed):
    trials = 1000000
    x = nn.Variable([100], need_grad=True)
    x.d = np.random.random(x.size).astype(np.float32)
    w = nn.Variable([x.size], need_grad=True)
    w.d = np.random.randint(1, 100, w.size)
    with nn.context_scope(ctx), nn.auto_forward(True):
        y = F.random_choice(x, w, shape=[trials], replace=True, seed=seed)
    hist_nn, _ = np.histogram(y.d)
    hist_np, _ = np.histogram(np.random.choice(
        x.d, trials, True, w.d / np.float32(w.d.sum())))
    assert_allclose(hist_nn / trials, hist_np / trials, atol=1e-2)
    x.grad.zero()
    w.grad.zero()
    y.backward()
    assert_allclose(x.g / trials, w.d / np.float32(w.d.sum()), atol=1e-2)
    assert_allclose(w.g / trials, w.d / np.float32(w.d.sum()), atol=1e-2)

    x = nn.Variable.from_numpy_array(np.array([[1, 2, 3], [-1, -2, -3]]))
    w = nn.Variable.from_numpy_array(np.array([[1, 1, 1], [10, 10, 10]]))
    with nn.context_scope(ctx), nn.auto_forward():
        y = F.random_choice(x, w, shape=(10,), replace=True, seed=seed)
    assert y.shape == (2, 10) and np.all(y.d[0] > 0) and np.all(y.d[1] < 0)

    return
    x = nn.Variable((3, 3), need_grad=True)
    w = nn.Variable((3, 3), need_grad=True)
    w.d = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    with nn.context_scope(ctx), nn.auto_forward(True):
        y = F.random_choice(x, w, shape=[10], replace=True, seed=seed)
    x.grad.zero()
    w.grad.zero()
    y.backward(1)
    assert np.all(x.g == np.array([[10, 0, 0], [0, 10, 0], [0, 0, 10]]))
    assert np.all(w.g == np.array([[10, 0, 0], [0, 10, 0], [0, 0, 10]]))


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [-1, 313, 999])
def test_random_choice_without_replacement(ctx, func_name, seed):
    x = nn.Variable.from_numpy_array(np.array([0, 1, 2]).astype(np.int32))
    w = nn.Variable.from_numpy_array(np.array([5, 5, 90]).astype(np.int32))
    x.need_grad = True
    w.need_grad = True
    repeats = 1000
    with nn.context_scope(ctx):
        y = F.random_choice(x, w, shape=[w.size], replace=False, seed=seed)
    r = np.zeros((repeats, w.size)).astype(np.int32)
    for i in range(repeats):
        y.forward()
        r[i] = y.d.copy()
    assert np.all(np.bincount(r.flatten()) == x.size * [repeats])
