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

ctxs_rand = list_context('Rand')
ctxs_randint = list_context('Randint')
ctxs_randn = list_context('Randn')


@pytest.mark.parametrize("ctx, func_name", ctxs_rand)
@pytest.mark.parametrize("low, high", [(0, 1), (-2.5, 100), (0.1, 0.11)])
@pytest.mark.parametrize("shape", [[], [5], [100, 100]])
@pytest.mark.parametrize("seed", [-1, 313])
def test_rand_forward(seed, ctx, func_name, low, high, shape):
    with nn.context_scope(ctx):
        o = F.rand(low, high, shape, seed=seed)
    assert o.shape == tuple(shape)
    assert o.parent.name == func_name
    o.forward()
    assert np.all(o.d < high)
    assert np.all(o.d >= low)


@pytest.mark.parametrize("ctx, func_name", ctxs_randint)
@pytest.mark.parametrize("low, high", [(100, 50000), (-5, 100), (101, 102)])
@pytest.mark.parametrize("shape", [[], [5], [100, 100]])
@pytest.mark.parametrize("seed", [-1, 313])
def test_randint_forward(seed, ctx, func_name, low, high, shape):
    with nn.context_scope(ctx):
        o = F.randint(low, high, shape, seed=seed)
    assert o.shape == tuple(shape)
    assert o.parent.name == func_name
    o.forward()
    # NOTE: The following should be < high,
    # but use <= high because std::uniform_random contains a bug.
    assert np.all(o.d <= high)
    assert np.all(o.d >= low)


@pytest.mark.parametrize("ctx, func_name", ctxs_randn)
@pytest.mark.parametrize("mu, sigma", [(0, 1), (-10, 10), (10000.5, 0.5)])
@pytest.mark.parametrize("shape", [[], [5], [100, 100]])
@pytest.mark.parametrize("seed", [-1, 313])
def test_randn_forward_backward(seed, ctx, func_name, mu, sigma, shape):
    with nn.context_scope(ctx):
        o = F.randn(mu, sigma, shape, seed=seed)
    assert o.shape == tuple(shape)
    assert o.parent.name == func_name
    o.forward()
    if o.size >= 10000:
        est_mu = o.d.mean()
        est_sigma = o.d.std()
        np.isclose(est_mu, mu, atol=sigma)
        np.isclose(est_sigma, sigma, atol=sigma)
    else:
        data = []
        for i in range(10000):
            o.forward()
            data += [o.d.copy()]
        est_mu = np.mean(np.array(data))
        est_sigma = np.std(np.array(data))
        np.isclose(est_mu, mu, atol=sigma)
        np.isclose(est_sigma, sigma, atol=sigma)
