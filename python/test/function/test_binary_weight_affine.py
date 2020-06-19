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

ctxs = list_context('BinaryWeightAffine')


def binarize(x, quantize_zero_to):
    # binarization to +/-H where H is the average L1-norm for each output
    y = np.sign(x)
    y[y == 0] = quantize_zero_to
    y = y * np.mean(np.abs(x), 0)
    return y


def ref_binary_weight_affine(x, w, wb, alpha, b, base_axis, quantize_zero_to):
    # wb and alpha is the placeholder for the binary weights and scaling factor
    shape = list(x.shape[:base_axis])
    shape += [-1]
    out_shape = w.shape[1:]

    # Binarize weights to -1 and 1, and set zeroes to -1
    binw = binarize(w.reshape(w.shape[0], -1), quantize_zero_to)

    y = np.dot(x.reshape(*shape), binw)
    if b is not None:
        y += b.reshape((1,) * (len(shape) - 1) + (-1,))
    return y.reshape(tuple(shape[:-1]) + tuple(out_shape))


def ref_grad_binary_weight_affine(x, w, wb, alpha, b, dy, base_axis, quantize_zero_to, **kw):
    # wb and alpha is the placeholder for the binary weights and scaling factor
    shape = list(x.shape[:base_axis])

    x_ = x.reshape(np.prod(shape), -1)
    wb_ = binarize(w, quantize_zero_to).reshape(w.shape[0], -1)
    dy_ = dy.reshape(np.prod(shape), -1)

    dx = np.dot(dy_, np.transpose(wb_))
    dw = np.dot(np.transpose(x_), dy_)

    if b is not None:
        db = np.sum(dy_, 0)
    else:
        db = np.empty(0)

    return np.concatenate([dx.flatten(),
                           dw.flatten(),
                           db.flatten()])


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("base_axis, weight_shape",
                         [(1, (12, 2, 3)), (2, (4, 4))])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("quantize_zero_to", [0.0, -1.0, 1.0])
def test_binary_weight_affine_forward_backward(seed, base_axis, weight_shape, bias, quantize_zero_to,
                                               ctx, func_name):
    from nbla_test_utils import function_tester
    rng = np.random.RandomState(seed)
    # Input
    inputs = [rng.randn(2, 3, 4).astype(np.float32)]
    # Weights
    inputs += [rng.randn(*weight_shape).astype(np.float32)]
    # Binary Weights (initialized to zero)
    inputs += [np.zeros(weight_shape).astype(np.float32)]
    # Alpha: Scaling factor
    inputs += [np.zeros(np.prod(weight_shape[1:])).astype(np.float32)]
    # Bias
    if bias:
        inputs += [rng.randn(*weight_shape[1:]).astype(np.float32)]
    else:
        inputs += [None]

    function_tester(rng, F.binary_weight_affine, ref_binary_weight_affine, inputs, func_args=[base_axis, quantize_zero_to],
                    atol_b=1e-2, backward=[True, True, False, False, True], ctx=ctx, func_name=func_name,
                    ref_grad=ref_grad_binary_weight_affine)
