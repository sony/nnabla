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

ctxs = list_context('SgdW')


class RefSgdW(RefSolver):

    def __init__(self, lr, momentum, wd):
        self.lr = lr
        self.momentum = momentum
        self.v = {}
        self.wd = wd
        self.init_lr = lr

    def _set_state_impl(self, key, param):
        self.v[key] = np.zeros_like(param)

    def _update_impl(self, key, p, g):
        _update_momentum(p, g, self.v[key], self.lr,
                         self.momentum, self.wd, self.init_lr)


def _update_momentum(p, g, v, lr, momentum, wd, init_lr):
    eta_t = lr / init_lr
    v[...] = v * momentum + lr * g - eta_t * wd * v
    p[...] = p - v


@pytest.mark.parametrize("ctx, solver_name", ctxs)
@pytest.mark.parametrize("decay", [1e-4])
@pytest.mark.parametrize("lr", [1e-2, 1e-4])
@pytest.mark.parametrize("momentum", [0.9, 0.5])
@pytest.mark.parametrize("seed", [313])
def test_sgdw(seed, lr, momentum, decay, ctx, solver_name):
    rng = np.random.RandomState(seed)
    solver_tester(
        rng, S.SgdW, RefSgdW, [lr, momentum, decay], atol=1e-6, ctx=ctx, solver_name=solver_name)
