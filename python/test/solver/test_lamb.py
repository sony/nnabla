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
import nnabla as nn
import nnabla.solvers as S
import numpy as np
from solver_test_utils import solver_tester, RefSolver, MixinWeightDecayFused
from nbla_test_utils import list_context

ctxs = list_context('Lamb')


def _f(x):
    return np.asarray(x, dtype=np.float32)


class RefLamb(MixinWeightDecayFused, RefSolver):
    def __init__(self, eta, beta1, beta2, gamma_l, gamma_u, eps, bias_correction):
        super().__init__()
        self.eta = _f(eta)
        self.beta1 = _f(beta1)
        self.beta2 = _f(beta2)
        self.gamma_l = _f(gamma_l)
        self.gamma_u = _f(gamma_u)
        self.eps = _f(eps)
        self.bias_correction = bias_correction
        self.v = {}
        self.t = {}

    def _set_state_impl(self, key, param):
        self.v[key] = {}
        self.v[key]["m"] = np.zeros_like(param)
        self.v[key]["v"] = np.zeros_like(param)
        self.t[key] = 0

    def _update_impl(self, key, w, g):
        self.t[key] = min(self.t[key] + 1, np.iinfo(np.int32).max)
        t = self.t[key]
        weight_decay = self.weight_decay_rate

        m = self.v[key]["m"]
        v = self.v[key]["v"]

        m[...] = self.beta1 * m + (1 - self.beta1) * g
        v[...] = self.beta2 * v + (1 - self.beta2) * g * g

        corr1 = 1
        corr2 = 1
        if self.bias_correction:
            corr1 = 1 - self.beta1 ** t
            corr2 = 1 - self.beta2 ** t

        r = (m / corr1) / (np.sqrt(v / corr2) + self.eps)
        r = r + weight_decay * w

        v_norm = np.linalg.norm(w)
        v_norm = np.clip(v_norm, a_min=self.gamma_l, a_max=self.gamma_u)
        g_norm = np.linalg.norm(r)
        if g_norm > self.eps:
            local_lr = v_norm / g_norm
        else:
            local_lr = _f(1.0)
        w[...] -= self.eta * local_lr * r


@pytest.mark.parametrize("ctx, solver_name", ctxs)
@pytest.mark.parametrize("decay", [1e-4])
@pytest.mark.parametrize("eta", [1e-2, 1e-4])
@pytest.mark.parametrize("beta1", [0.9, 0.5])
@pytest.mark.parametrize("beta2", [0.999, 0.5])
@pytest.mark.parametrize("gamma_l", [1e-6, 0.1])
@pytest.mark.parametrize("gamma_u", [10, 100])
@pytest.mark.parametrize("eps", [1e-8])
@pytest.mark.parametrize("bias_correction", [False, True])
@pytest.mark.parametrize("seed", [313])
def test_lamb(seed, eta, beta1, beta2, gamma_l, gamma_u, eps, bias_correction, decay, ctx, solver_name):
    rng = np.random.RandomState(seed)
    solver_tester(
        rng, S.Lamb, RefLamb,
        [eta, beta1, beta2, gamma_l, gamma_u, eps, bias_correction], atol=1e-6, decay=decay,
        ctx=ctx, solver_name=solver_name)
