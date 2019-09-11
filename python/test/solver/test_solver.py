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
import nnabla as nn
import nnabla.solvers as S
import numpy as np
from solver_test_utils import solver_tester, RefSolver
from nbla_test_utils import list_context
from nnabla.testing import assert_allclose


def test_solver_zeroing():
    xs = [nn.Variable([2, 3, 4], need_grad=True) for _ in range(3)]

    s = S.Sgd(1)
    s.set_parameters({str(i): x for i, x in enumerate(xs)})

    for x in xs:
        x.data.fill(1)
        x.grad.zero()

    s.weight_decay(1.0)
    s.update()
    for x in xs:
        # Grad is not referenced since neither weight decay nor update is performed.
        assert x.grad.zeroing
        assert_allclose(x.d, 1)

    for x in xs:
        x.grad.fill(1)

    s.weight_decay(0.1)
    s.update()

    for x in xs:
        assert_allclose(x.d, 1 - (1 + 0.1))
