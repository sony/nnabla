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

ctxs = list_context('Concatenate')


def ref_concatenate(*inputs, **params):
    axis = params.pop('axis', len(inputs[0].shape) - 1)
    return np.concatenate(inputs, axis=axis)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("axis", [0, 1, 2, -1, -2, -3])
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize('different_size', [False, True])
@pytest.mark.parametrize('num_inputs', [2, 3])
def test_concatenate_forward_backward(seed, axis, different_size, num_inputs, ctx, func_name):
    from nbla_test_utils import cap_ignore_region, function_tester
    rng = np.random.RandomState(seed)
    shape0 = [2, 3, 4]
    inputs = []
    for i in range(num_inputs):
        inputs.append(rng.randn(*shape0).astype(np.float32))
        shape0[axis] += int(different_size)
    function_tester(rng, F.concatenate, ref_concatenate, inputs,
                    func_kwargs=dict(axis=axis), ctx=ctx, func_name=func_name,
                    atol_b=1e-2)


def test_no_value():
    a = nn.Variable(())
    b = nn.Variable(())
    with pytest.raises(RuntimeError):
        F.concatenate(*[a, b], axis=0)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("axis", [0, 1, 2, -1, -2, -3])
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize('different_size', [False, True])
@pytest.mark.parametrize('num_inputs', [2, 3])
def test_concatenate_double_backward(seed, axis, different_size, num_inputs, ctx, func_name):
    from nbla_test_utils import cap_ignore_region, backward_function_tester, grad_function_forward_function_output
    from nnabla.backward_function.concatenate import ConcatenateDataGrad
    rng = np.random.RandomState(seed)
    shape0 = [2, 3, 4]
    inputs = []
    for i in range(num_inputs):
        inputs.append(rng.randn(*shape0).astype(np.float32))
        shape0[axis] += int(different_size)
    func_kwargs = dict(axis=axis)

    # 2nd-order
    backward_function_tester(rng, F.concatenate,
                             inputs=inputs,
                             func_args=[], func_kwargs=func_kwargs,
                             atol_accum=1e-2,
                             dstep=1e-3,
                             ctx=ctx)
    # 3rd-order
    df, y = grad_function_forward_function_output(ConcatenateDataGrad,
                                                  F.concatenate,
                                                  ctx, inputs,
                                                  *[], **func_kwargs)
    df.xshapes = [x.shape for x in inputs]
    ginputs = [rng.randn(*y.shape)]
    backward_function_tester(rng, df,
                             ginputs,
                             ctx=ctx, non_accum_check=True)
