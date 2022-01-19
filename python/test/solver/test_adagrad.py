# Copyright 2017,2018,2019,2020,2021 Sony Corporation.
# Copyright 2021 Sony Group Corporation.
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

ctxs = list_context('Adagrad')


class RefAdagrad(RefSolver):

    def __init__(self, lr, eps):
        self.lr = lr
        self.eps = eps
        self.G = {}

    def _set_state_impl(self, key, param):
        self.G[key] = np.zeros_like(param)

    def _update_impl(self, key, p, g):
        _update_adagrad(p, g, self.G[key], self.lr, self.eps)


def _update_adagrad(p, g, G, lr, eps):
    G[...] = G + g * g
    p[...] = p - lr / (np.sqrt(G) + eps) * g


@pytest.mark.parametrize("ctx, solver_name", ctxs)
@pytest.mark.parametrize("decay", [1e-4])
@pytest.mark.parametrize("lr", [1e-1, 1e-3])
@pytest.mark.parametrize("eps", [1e-8])
@pytest.mark.parametrize("seed", [313])
def test_adagrad(seed, lr, eps, decay, ctx, solver_name):
    rng = np.random.RandomState(seed)
    solver_tester(
        rng, S.Adagrad, RefAdagrad, [lr, eps], atol=1e-6, ctx=ctx, solver_name=solver_name)
