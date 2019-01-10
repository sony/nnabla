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

"""
This is still left unlike other *2 operation (sub2, mul2, ...) because
it has cudnn implementation.
"""
import pytest
import numpy as np
import nnabla as nn
import nnabla.functions as F
from nbla_test_utils import list_context

ctxs = list_context('Assign')


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [314])
def test_assign_forward_backward(seed, ctx, func_name):
    rng = np.random.RandomState(seed)
    dst = nn.Variable((2, 3, 4), need_grad=True)
    src = nn.Variable((2, 3, 4), need_grad=True)

    assign = F.assign(dst, src)

    src.d = np.random.random((2, 3, 4))
    assign.forward()

    # destination variable should be equal to source variable
    assert np.allclose(dst.d, src.d)
    # output variable of assign function should be equal to soure variable
    assert np.allclose(assign.d, src.d)

    dummy = assign + np.random.random()

    dst.grad.zero()
    src.grad.zero()
    dummy.forward()
    dummy.backward()

    # assign should not propagate gradients
    assert np.all(dst.g == np.zeros((2, 3, 4)))
    assert np.all(src.g == np.zeros((2, 3, 4)))
