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
import nnabla.solvers as S
import numpy as np
from solver_test_utils import solver_tester, RefSolver
from nbla_test_utils import list_context

ctxs = list_context('AMSGRAD')


class RefAMSGRAD(RefSolver):

    def __init__(self, alpha, beta1, beta2, eps, bias_correction):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.bias_correction = bias_correction
        self.m = {}
        self.v = {}
        self.v_hat = {}
        self.t = {}

    def _set_state_impl(self, key, param):
        self.m[key] = np.zeros_like(param)
        self.v[key] = np.zeros_like(param)
        self.v_hat[key] = np.zeros_like(param)
        self.t[key] = 0

    def _update_impl(self, key, p, g):
        self.t[key] = min(self.t[key] + 1, np.iinfo(np.int32).max)
        _update_amsgrad(p, g, self.m[key], self.v[key], self.v_hat[key], self.t[key],
                        self.alpha, self.beta1, self.beta2, self.eps, self.bias_correction)


def _update_amsgrad(p, g, m, v, v_hat, t, alpha, beta1, beta2, eps, bias_correction):
    if bias_correction:
        alpha_t = alpha * \
            np.sqrt(1. - beta2 ** t) / (1. - beta1 ** t)
    else:
        alpha_t = alpha
    m[...] = beta1 * m + (1 - beta1) * g
    v[...] = beta2 * v + (1 - beta2) * g * g
    v_hat[...] = np.maximum(v_hat, v)
    p[...] = p - alpha_t * m / (np.sqrt(v_hat) + eps)


@pytest.mark.parametrize("ctx, solver_name", ctxs)
@pytest.mark.parametrize("decay", [1e-4])
@pytest.mark.parametrize("alpha", [1e-2, 1e-4])
@pytest.mark.parametrize("beta1, beta2", [(0.9, 0.999), (0.999, 0.9)])
@pytest.mark.parametrize("eps", [1e-8])
@pytest.mark.parametrize("bias_correction", [True, False])
@pytest.mark.parametrize("seed", [313])
def test_amsgrad(seed, alpha, beta1, beta2, eps, bias_correction, decay, ctx, solver_name):
    rng = np.random.RandomState(seed)
    solver_tester(
        rng, S.AMSGRAD, RefAMSGRAD, [alpha, beta1,
                                     beta2, eps, bias_correction], atol=1e-6,
        ctx=ctx, solver_name=solver_name)
