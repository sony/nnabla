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

ctxs = list_context('Transpose')


def ref_transpose(x, axes):
    return x.transpose(axes).copy()


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("inshape, axes", [
    ((10,), (0,)),
    ((10, 11), (0, 1)),
    ((10, 11), (1, 0)),
    ((11, 13, 7), (0, 2, 1)),
    ((11, 13, 7), (2, 1, 0)),
    ((3, 7, 4, 5), (3, 0, 1, 2)),
    ((4, 2, 5, 2, 3), (3, 0, 1, 2, 4)),
    ((4, 2, 3, 2, 3, 3), (5, 3, 0, 1, 2, 4)),
    ((4, 4, 4, 4, 4), (4, 3, 2, 1, 0)),
    ((7, 16, 2, 2, 224, 448), (0, 1, 4, 2, 5, 3)),
])
def test_transpose_forward_backward(seed, inshape, axes, ctx, func_name):
    from nbla_test_utils import function_tester
    rng = np.random.RandomState(seed)
    if np.product(inshape) > 1000000:
        with nn.context_scope(ctx):
            x = nn.Variable(inshape)
            y = F.transpose(x, axes)
            y.forward()
            assert y.d.shape == np.ndarray(inshape).transpose(axes).shape
    else:
        inputs = [rng.randn(*inshape).astype(np.float32)]
        function_tester(rng, F.transpose, ref_transpose, inputs,
                        func_args=[axes], ctx=ctx, func_name=func_name,
                        atol_f=1e-6, atol_b=1e-2)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("inshape, axes", [
    ((11, 13, 7), (2, 1, 0)),
    ((3, 7, 4, 5), (3, 0, 1, 2)),
    ((4, 2, 5, 2, 3), (3, 0, 1, 2, 4)),
    ((4, 2, 3, 2, 3, 3), (5, 3, 0, 1, 2, 4)),
    ((4, 4, 4, 4, 4), (4, 3, 2, 1, 0)),
])
def test_transpose_double_backward(seed, inshape, axes, ctx, func_name):
    from nbla_test_utils import backward_function_tester, grad_function_forward_function_output
    from nnabla.backward_function.transpose import TransposeDataGrad
    rng = np.random.RandomState(seed)
    # Input
    inputs = [rng.randn(*inshape).astype(np.float32)]
    func_args = [axes]
    # 2rd-order
    backward_function_tester(rng, F.transpose, inputs,
                             func_args=func_args, ctx=ctx)
    # 3rd-order
    df, y = grad_function_forward_function_output(TransposeDataGrad,
                                                  F.transpose,
                                                  ctx, inputs,
                                                  *func_args)
    df.xshape = inputs[0].shape
    ginputs = [rng.randn(*y.shape)]
    backward_function_tester(rng, df,
                             ginputs, func_args=[],
                             ctx=ctx, non_accum_check=True)
