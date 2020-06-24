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
import platform

ctxs_rand = list_context('Rand')
ctxs_randint = list_context('Randint')
ctxs_randn = list_context('Randn')
ctxs_rand_binomial = list_context('RandBinomial')
ctxs_rand_beta = list_context('RandBeta')
ctxs_rand_gamma = list_context('RandGamma')


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
    # NOTE: The following should be < high,
    # but use <= high because std::uniform_random contains a bug.
    assert np.all(o.d <= high)
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


@pytest.mark.parametrize("ctx, func_name", ctxs_rand_beta)
@pytest.mark.parametrize("alpha, beta", [(0.5, 0.5), (5, 1), (1, 3), (2, 5), (2, 2)])
@pytest.mark.parametrize("shape", [[50], [100, 100], [32, 4, 16, 16]])
@pytest.mark.parametrize("seed", [-1, 313])
def test_rand_beta_forward(seed, ctx, func_name, alpha, beta, shape):
    with nn.context_scope(ctx):
        o = F.rand_beta(alpha, beta, shape, seed=seed)
    assert o.shape == tuple(shape)
    assert o.parent.name == func_name
    o.forward()
    if o.size >= 10000:
        est_mu = o.d.mean()
        est_sigma = o.d.std()
    else:
        data = []
        for i in range(10000):
            o.forward()
            data += [o.d.copy()]
        est_mu = np.mean(np.array(data))
        est_sigma = np.std(np.array(data))

    mu = alpha / (alpha + beta)  # theoretical mean
    var = alpha*beta / ((alpha + beta)*(alpha + beta)*(alpha + beta + 1))
    sigma = np.sqrt(var)  # theoretical std

    assert np.isclose(est_mu, mu, atol=5e-2)
    assert np.isclose(est_sigma, sigma, atol=5e-2)


@pytest.mark.parametrize("ctx, func_name", ctxs_rand_binomial)
@pytest.mark.parametrize("n, p", [(1, 0.5), (1, 0.9), (5, 0.5), (5, 0.15), (10, 0.45)])
@pytest.mark.parametrize("shape", [[50], [100, 100], [32, 4, 16, 16]])
@pytest.mark.parametrize("seed", [-1, 313])
def test_rand_binomial_forward(seed, ctx, func_name, n, p, shape):
    with nn.context_scope(ctx):
        o = F.rand_binomial(n, p, shape, seed=seed)
    assert o.shape == tuple(shape)
    assert o.parent.name == func_name
    o.forward()
    if o.size >= 10000:
        est_mu = o.d.mean()
        est_sigma = o.d.std()
    else:
        data = []
        for i in range(10000):
            o.forward()
            data += [o.d.copy()]
        est_mu = np.mean(np.array(data))
        est_sigma = np.std(np.array(data))

    mu = n * p  # theoretical mean
    sigma = np.sqrt(n * p * (1 - p))  # theoretical std

    assert np.isclose(est_mu, mu, atol=5e-2)
    assert np.isclose(est_sigma, sigma, atol=5e-2)


@pytest.mark.parametrize("ctx, func_name", ctxs_rand_gamma)
@pytest.mark.parametrize("k, theta", [(1, 2), (9, 0.5), (3, 2), (7.5, 1), (0.5, 1)])
@pytest.mark.parametrize("shape", [[50], [100, 100], [1000, 1000]])
@pytest.mark.parametrize("seed", [-1, 313])
@pytest.mark.skipif(platform.system() == "Darwin", reason='skipped on mac')
@pytest.mark.skipif(platform.system() == "Windows", reason='skipped on win')
def test_rand_gamma_forward(seed, ctx, func_name, k, theta, shape):
    with nn.context_scope(ctx):
        o = F.rand_gamma(k, theta, shape, seed=seed)
    assert o.shape == tuple(shape)
    assert o.parent.name == func_name
    o.forward()
    if o.size > 10000:
        est_mu = o.d.mean()
        est_sigma = o.d.std()
    else:
        data = []
        for i in range(1000000//o.size):
            o.forward()
            data += [o.d.copy()]
        est_mu = np.mean(np.array(data))
        est_sigma = np.std(np.array(data))

    mu = k * theta  # theoretical mean
    var = k * theta * theta
    sigma = np.sqrt(var)  # theoretical std

    assert np.isclose(est_mu, mu, atol=5e-2)
    assert np.isclose(est_sigma, sigma, atol=5e-2)
