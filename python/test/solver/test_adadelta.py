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

ctxs = list_context('Adadelta')


class RefAdadelta(RefSolver):

    def __init__(self, lr, decay, eps):
        self.lr = lr
        self.decay = decay
        self.eps = eps
        self.msdelta = {}
        self.msgrad = {}
        self.delta = {}

    def _set_state_impl(self, key, param):
        self.msdelta[key] = np.zeros_like(param)
        self.msgrad[key] = np.zeros_like(param)
        self.delta[key] = np.zeros_like(param)

    def _update_impl(self, key, p, g):
        _update_adadelta(p, g, self.msdelta[key], self.msgrad[
                         key], self.delta[key], self.lr, self.decay, self.eps)


def _update_adadelta(p, g, msdelta, msgrad, delta, lr, decay, eps):
    msgrad[...] = msgrad * decay + g * g * (1 - decay)
    msdelta[...] = msdelta * decay + delta * delta * (1 - decay)
    delta[...] = -np.sqrt((msdelta + eps) / (msgrad + eps)) * g
    p[...] = p + lr * delta


@pytest.mark.parametrize("ctx, solver_name", ctxs)
@pytest.mark.parametrize("lr", [1e-1, 1e-3])
@pytest.mark.parametrize("decay", [0.9, 0.8])
@pytest.mark.parametrize("eps", [1e-4])
@pytest.mark.parametrize("seed", [313])
def test_adadelta(seed, lr, eps, decay, ctx, solver_name):
    rng = np.random.RandomState(seed)
    solver_tester(
        rng, S.Adadelta, RefAdadelta, [lr, decay, eps], atol=1e-6, ctx=ctx, solver_name=solver_name)
