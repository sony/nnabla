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

ctxs = list_context('Sgd')


class RefSgd(RefSolver):

    def __init__(self, lr):
        self.lr = lr

    def _set_state_impl(self, key, param):
        pass

    def _update_impl(self, key, p, g):
        p[...] = p - self.lr * g


@pytest.mark.parametrize("ctx, solver_name", ctxs)
@pytest.mark.parametrize("decay", [1e-4])
@pytest.mark.parametrize("lr", [1e-2, 1e-4])
@pytest.mark.parametrize("seed", [313])
def test_sgd(seed, lr, decay, ctx, solver_name):
    rng = np.random.RandomState(seed)
    solver_tester(
        rng, S.Sgd, RefSgd, [lr], atol=1e-6, ctx=ctx, solver_name=solver_name)
