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
import numpy as np
import nnabla as nn
import nnabla.functions as F
from nbla_test_utils import list_context
from nnabla.testing import assert_allclose

ctxs = list_context('MeanSubtraction')


def ref_mean_subtraction(x, rmean, t, base_axis, batch_stat):
    if batch_stat:
        mean = x.mean(tuple(range(0, base_axis))) if base_axis >= 0 else x.mean(
            tuple(range(0, len(x.shape)+base_axis)))
        rmean[...] = rmean + (mean - rmean) / (t + 1)
        t += 1
    return x - rmean


@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("inshape", [(2, 3, 4)])
@pytest.mark.parametrize("base_axis", [0, 1, 2, -1, -2, -3])
@pytest.mark.parametrize("ctx, func_name", ctxs)
def test_mean_subtraction_forward_backward(seed, inshape, base_axis, ctx, func_name):
    from nbla_test_utils import function_tester
    rng = np.random.RandomState(seed)
    mean_shape = inshape[base_axis:
                         ] if base_axis >= 0 else inshape[base_axis+len(inshape):]
    inputs = [np.array(rng.randn(*inshape).astype(np.float32)),
              np.zeros(mean_shape),
              np.array([1000])]
    batch_stat = True

    function_tester(rng, F.mean_subtraction, ref_mean_subtraction,
                    inputs,
                    func_args=[base_axis, batch_stat],
                    ctx=ctx, func_name=func_name, dstep=1e-2,
                    backward=[True, False, False],
                    atol_b=1e-2)

    # Check if running mean works.
    vinputs = []
    for input in inputs:
        vinputs.append(nn.Variable(input.shape, True))
        vinputs[-1].d = input
    for i in range(5):
        inputs[0] = rng.randn(*inputs[0].shape)
        vinputs[0].d[...] = inputs[0]
        ref_y, rmean = ref_mean_subtraction(
            *(inputs + [base_axis, batch_stat]))
        with nn.auto_forward():
            y = F.mean_subtraction(*(vinputs + [base_axis, batch_stat]))
        # print('vinput[1].d', vinputs[1].d, vinputs[2].d)
        # print('inputs[1]', inputs[1], inputs[2])
        assert_allclose(vinputs[1].d, inputs[1])

    # Check if global stat mode works
    batch_stat = False
    ref_y = ref_mean_subtraction(*(inputs + [base_axis, batch_stat]))
    with nn.auto_forward():
        y = F.mean_subtraction(*(vinputs + [base_axis, batch_stat]))
    assert_allclose(ref_y, y.d)
