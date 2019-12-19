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
                         [(1, (12, 2, 3)), (2, (4, 4))])
@pytest.mark.parametrize("bias", [True, False])
def test_affine_double_backward(seed, base_axis, weight_shape, bias,
                                ctx, func_name):

    from nbla_test_utils import backward_function_tester
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
    backward_function_tester(rng, F.affine, None, inputs, func_args=[base_axis],
                             atol_b=1e-2, atol_accum=1e-2, dstep=1e-3, ctx=ctx, func_name=func_name)
