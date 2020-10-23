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
    def __init__(self, alpha, beta1, beta2, eps, amsgrad, weight_decouple, fixed_decay, rectify):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.amsgrad = amsgrad
        self.weight_decouple = weight_decouple
        self.fixed_decay = fixed_decay
        self.rectify = rectify

    def _set_state_impl(self, key, param):
        pass

    def _update_impl(self, key, p, grad):
        # TODO: Fill missing internal parameters
        _update_adabelief(p, grad, self.alpha, self.beta1, self.beta2, self.eps,
                          self.amsgrad, self.weight_decouple, self.fixed_decay, self.rectify)


def _update_adabelief(p, grad, alpha, beta1, beta2, eps,
                      amsgrad, weight_decouple, fixed_decay, rectify):
    pass


@pytest.mark.parametrize("ctx, solver_name", ctxs)
@pytest.mark.parametrize("alpha", [1e-2, 1e-4])
@pytest.mark.parametrize("beta1, beta2", [(0.9, 0.999), (0.999, 0.9)])
@pytest.mark.parametrize("eps", [1e-8])
@pytest.mark.parametrize("amsgrad", [True, False])
@pytest.mark.parametrize("weight_decouple", [True, False])
@pytest.mark.parametrize("fixed_decay", [True, False])
@pytest.mark.parametrize("rectify", [True, False])
@pytest.mark.parametrize("seed", [313])
def test_adam(seed, alpha, beta1, beta2, eps, amsgrad, weight_decouple, fixed_decay, rectify, ctx, solver_name):
    rng = np.random.RandomState(seed)
    solver_tester(
        rng, S.AdaBelief, RefAdaBelief, [
            alpha, beta1, beta2, eps, amsgrad, weight_decouple, fixed_decay, rectify],
        atol=1e-6, ctx=ctx, solver_name=solver_name)
