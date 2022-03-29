# Copyright 2022 Sony Group Corporation.
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
import nnabla.functions as F
from nbla_test_utils import list_context, function_tester

ctxs = list_context('BatchCholesky')


def batch_cholesky(x, upper):
    L = np.linalg.cholesky(x)
    return np.transpose(L, axes=(0, 2, 1)) if upper else L


def batch_cholesky_numerical_grad(x, vgrads, upper, *args, **kwargs):
    # Our implementation of cholesky decomposition returns symmetric grad instead of lower triangular grad
    from scipy.optimize import approx_fprime

    def func(vector):
        matrix = np.reshape(vector, x.shape)
        L = batch_cholesky(matrix, upper) * vgrads
        return np.sum(L)
    x_c = np.concatenate([i.flatten() for i in x if i is not None])
    epsilon = 1e-4
    n_grad = approx_fprime(x_c, func, epsilon)
    n_grad = np.reshape(n_grad, x.shape)
    n_grad = (n_grad + np.transpose(n_grad, axes=(0, 2, 1))) * 0.5
    return n_grad.flatten()


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("upper", [False, True])
def test_forward_backward(seed, ctx, func_name, upper):
    rng = np.random.RandomState(seed)
    x = rng.randn(2, 4, 4).astype(np.float32)
    for i in range(x.shape[0]):
        x[i] = np.dot(x[i], np.transpose(x[i]))
    inputs = [x]
    function_tester(rng, F.batch_cholesky, batch_cholesky, inputs,
                    func_args=[upper],
                    atol_b=2e-2,
                    func_name=func_name, ctx=ctx, backward=[True],
                    ref_grad=batch_cholesky_numerical_grad,
                    disable_half_test=True)
