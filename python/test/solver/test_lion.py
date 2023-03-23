# Copyright 2023 Sony Group Corporation.
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
from solver_test_utils import solver_tester, RefSolver, MixinWeightDecayFused
from nbla_test_utils import list_context

ctxs = list_context('Lion')


def _f(x):
    return np.asarray(x, dtype=np.float32)


class RefLion(MixinWeightDecayFused, RefSolver):

    def __init__(self, lr, beta1, beta2):
        super().__init__()
        self.lr = _f(lr)
        self.beta1 = _f(beta1)
        self.beta2 = _f(beta2)
        self.m = {}
        self.t = {}

    def _set_state_impl(self, key, param):
        self.m[key] = np.zeros_like(param)
        self.t[key] = 0

    def _update_impl(self, key, p, g):
        self.t[key] = min(self.t[key] + 1, np.iinfo(np.int32).max)
        _update_lion(p, g, self.m[key],
                     self.lr, self.beta1, self.beta2, self.weight_decay_rate)


def _update_lion(p, g, m, lr, beta1, beta2, wd):
    def _lerp(a, b, t):
        return a + t * (b - a)

    u = _lerp(g, m, beta1)
    u = np.sign(u)
    m[...] = _lerp(g, m, beta2)
    p[...] = p - lr * (u + wd * p)


@pytest.mark.parametrize("ctx, solver_name", ctxs)
@pytest.mark.parametrize("decay", [1e-4, 0.])
@pytest.mark.parametrize("lr", [1e-2, 1e-4])
@pytest.mark.parametrize("beta1, beta2", [(0.9, 0.999), (0.999, 0.9)])
@pytest.mark.parametrize("seed", [313])
def test_lion(seed, lr, beta1, beta2, decay, ctx, solver_name):
    rng = np.random.RandomState(seed)

    def _hook(itr, sol, ref_sol):
        ref_m = ref_sol.m
        act_m = {k: v.pstate["m"].data.get_data(
            'r') for k, v in sol.get_states().items()}
        print(f'---- iter={itr} ----')
        for k, ref in ref_m.items():
            act = act_m[k]
            print(f'[{k}]')
            print(f'ref={ref}')
            print(f'act={act}')

    _hook = None

    solver_tester(
        rng, S.Lion, RefLion, [lr, beta1, beta2],
        atol=1e-6, decay=decay,
        ctx=ctx, solver_name=solver_name, hook_solver_update=_hook)
