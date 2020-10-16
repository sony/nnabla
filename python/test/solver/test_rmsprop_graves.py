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

ctxs = list_context('RMSpropGraves')


class RefRMSpropGraves(RefSolver):

    def __init__(self, lr, decay, momentum, eps):
        self.lr = lr
        self.decay = decay
        self.momentum = momentum
        self.eps = eps
        self.n = {}
        self.g = {}
        self.d = {}

    def _set_state_impl(self, key, param):
        self.n[key] = np.zeros_like(param)
        self.g[key] = np.zeros_like(param)
        self.d[key] = np.zeros_like(param)

    def _update_impl(self, key, p, grad):
        _update_rmsprop_graves(
            p, grad, self.n[key], self.g[key], self.d[key], self.lr, self.decay, self.momentum, self.eps)


def _update_rmsprop_graves(p, grad, n, g, d, lr, decay, momentum, eps):
    n[...] = n * decay + grad * grad * (1 - decay)
    g[...] = g * decay + grad * (1 - decay)
    d[...] = d * momentum - lr * grad / np.sqrt(n - g * g + eps)
    p[...] = p + d


@pytest.mark.parametrize("ctx, solver_name", ctxs)
@pytest.mark.parametrize("lr", [1e-1, 1e-3])
@pytest.mark.parametrize("decay", [0.9, 0.8])
@pytest.mark.parametrize("momentum", [0.9, 0.8])
@pytest.mark.parametrize("eps", [1e-8])
@pytest.mark.parametrize("seed", [313])
def test_rmsprop_graves(seed, lr, eps, momentum, decay, ctx, solver_name):
    rng = np.random.RandomState(seed)
    solver_tester(
        rng, S.RMSpropGraves, RefRMSpropGraves, [lr, decay, momentum, eps], atol=1e-6, ctx=ctx, solver_name=solver_name)
