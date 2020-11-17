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

ctxs = list_context('AdaBelief')


class RefAdaBelief(RefSolver):
    def __init__(self, alpha, beta1, beta2, eps, wd, amsgrad, weight_decouple, fixed_decay, rectify):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.wd = wd
        self.amsgrad = amsgrad
        self.weight_decouple = weight_decouple
        self.fixed_decay = fixed_decay
        self.rectify = rectify

        self.m = {}
        self.s = {}
        self.t = {}
        self.s_max = {}

    def _set_state_impl(self, key, param):
        self.m[key] = np.zeros_like(param)
        self.s[key] = np.zeros_like(param)
        self.t[key] = 0
        # s_max is not always necessary but is set for keeping the code simple
        self.s_max[key] = np.zeros_like(param)

    def _update_impl(self, key, p, grad):
        self.t[key] = min(self.t[key] + 1, np.iinfo(np.int32).max)
        _update_adabelief(p, grad, self.m[key], self.s[key], self.s_max[key], self.t[key],
                          self.alpha, self.beta1, self.beta2, self.eps, self.wd,
                          self.amsgrad, self.weight_decouple, self.fixed_decay, self.rectify)


def _update_adabelief(p, grad, m, s, s_max, t,
                      alpha, beta1, beta2, eps, wd,
                      amsgrad, weight_decouple, fixed_decay, rectify):
    beta1_t = beta1 ** t
    beta2_t = beta2 ** t
    bias_correction1 = (1. - beta1_t)
    bias_correction2 = np.sqrt(1. - beta2_t)
    m[...] = beta1 * m + (1 - beta1) * grad
    s[...] = beta2 * s + (1 - beta2) * (grad - m) * (grad - m)
    if amsgrad:
        s_max[...] = np.maximum(s_max, s)
        s_max += eps
        denominator = np.sqrt(s_max) / bias_correction2
    else:
        s += eps
        denominator = np.sqrt(s) / bias_correction2
    if weight_decouple:
        if fixed_decay:
            p[...] = p - p * wd
        else:
            p[...] = p - p * wd * alpha
    if rectify:
        rho_inf = 2.0 / (1.0 - beta2) - 1.0
        rho_t = rho_inf - 2.0 * t * beta2_t / (1.0 - beta2_t)
        if rho_t > 4.0:
            r_t_numerator = (rho_t - 4.0) * (rho_t - 2.0) * rho_inf
            r_t_denominator = (rho_inf - 4.0) * (rho_inf - 2.0) * rho_t
            r_t = np.sqrt(r_t_numerator / r_t_denominator)
            alpha_t = alpha * r_t / bias_correction1
            p[...] = p - alpha_t * m / (denominator + eps)
        else:
            p[...] = p - alpha * m
    else:
        alpha_t = alpha / bias_correction1
        p[...] = p - alpha_t * m / (denominator + eps)


@pytest.mark.parametrize("ctx, solver_name", ctxs)
@pytest.mark.parametrize("alpha", [1e-2, 1e-4])
@pytest.mark.parametrize("beta1, beta2", [(0.9, 0.999), (0.999, 0.9)])
@pytest.mark.parametrize("eps", [1e-8, 1e-1])
@pytest.mark.parametrize("amsgrad", [True, False])
@pytest.mark.parametrize("weight_decouple, fixed_decay, wd",
                         [(True, False, 1e-3), (True, True, 1e-4), (False, False, 1e-5)])
@pytest.mark.parametrize("rectify", [True, False])
@pytest.mark.parametrize("seed", [313])
def test_adabelief(seed, alpha, beta1, beta2, eps, wd,
                   amsgrad, weight_decouple, fixed_decay, rectify, ctx, solver_name):
    rng = np.random.RandomState(seed)
    solver_tester(
        rng, S.AdaBelief, RefAdaBelief, [
            alpha, beta1, beta2, eps, wd, amsgrad, weight_decouple, fixed_decay, rectify],
        atol=1e-6, ctx=ctx, solver_name=solver_name)
