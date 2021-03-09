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
import nnabla.functions as F
from nbla_test_utils import list_context

ctxs = list_context('Affine')


def ref_affine(x, w, b, base_axis):
    shape = list(x.shape[:base_axis])
    shape += [-1]
    out_shape = w.shape[1:]
    y = np.dot(x.reshape(*shape), w.reshape(w.shape[0], -1))
    if b is not None:
        y += b.reshape((1,) * (len(shape) - 1) + (-1,))
    return y.reshape(tuple(shape[:-1]) + tuple(out_shape))


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("base_axis, weight_shape",
                         [(1, (12, 2, 3)), (2, (4, 4))])
@pytest.mark.parametrize("bias", [True, False])
def test_affine_forward_backward(seed, base_axis, weight_shape, bias,
                                 ctx, func_name):

    from nbla_test_utils import function_tester
    rng = np.random.RandomState(seed)
    # Input
    inputs = [rng.randn(2, 3, 4).astype(np.float32)]
    # Weight
    inputs += [rng.randn(*weight_shape).astype(np.float32)]
    # Bias
    if bias:
        inputs += [rng.randn(*weight_shape[1:]).astype(np.float32)]
    else:
        inputs += [None]
    function_tester(rng, F.affine, ref_affine, inputs, func_args=[base_axis],
                    atol_b=1e-2, dstep=1e-3, ctx=ctx, func_name=func_name)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("base_axis, weight_shape",
                         [(1, (12, 3, 4)), (2, (4, 4))])
@pytest.mark.parametrize("bias", [True, False])
def test_affine_double_backward(seed, base_axis, weight_shape, bias,
                                ctx, func_name):

    from nbla_test_utils import backward_function_tester, grad_function_forward_function_output
    from nnabla.backward_function.affine import AffineDataGrad, AffineFilterGrad
    rng = np.random.RandomState(seed)
    # Input
    inputs = [rng.randn(2, 3, 4).astype(np.float32)]
    # Weight
    inputs += [rng.randn(*weight_shape).astype(np.float32)]
    # Bias
    if bias:
        inputs += [rng.randn(*weight_shape[1:]).astype(np.float32) * 1e2]
    else:
        inputs += [None]
    func_args = [base_axis]
    # Affine
    backward_function_tester(rng, F.affine, inputs, func_args=func_args,
                             dstep=1e-3, ctx=ctx)
    # DataGrad
    df, y = grad_function_forward_function_output(AffineDataGrad,
                                                  F.affine, ctx, inputs, *func_args)
    df.xshape = inputs[0].shape
    ginputs = [rng.randn(*y.shape), inputs[1]]
    backward_function_tester(rng, df, ginputs, func_args=[],
                             atol_accum=2e-2, dstep=1e-3, ctx=ctx, non_accum_check=True)

    # FilterGrad
    df, y = grad_function_forward_function_output(AffineFilterGrad,
                                                  F.affine, ctx, inputs, *func_args)
    df.wshape = inputs[1].shape
    ginputs = [rng.randn(*y.shape), inputs[0]]
    backward_function_tester(rng, df, ginputs, func_args=[],
                             dstep=1e-3, ctx=ctx, non_accum_check=True)
