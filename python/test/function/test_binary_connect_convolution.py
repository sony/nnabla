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

from __future__ import division

import pytest
import numpy as np
import nnabla as nn
import nnabla.functions as F
import refs
from nbla_test_utils import list_context

ctxs = list_context('BinaryConnectConvolution')


def binarize(x, quantize_zero_to):
    y = np.sign(x)
    y[y == 0] = quantize_zero_to
    return y


def ref_convolution(x, w, b, base_axis, pad, stride, dilation, group, quantize_zero_to):
    y = []
    for xx in x.reshape((-1,) + x.shape[base_axis:]):
        y += [refs.convolution_2d(xx, w, b, pad, stride,
                                  dilation, group)[np.newaxis]]
    y = np.vstack(y)
    return y.reshape(x.shape[:base_axis] + y.shape[1:])


def ref_binary_connect_convolution(x, w, wb, b, base_axis, pad, stride, dilation, group, quantize_zero_to):
    return ref_convolution(x, binarize(w, quantize_zero_to), b, base_axis, pad, stride, dilation, group, quantize_zero_to)


def ref_grad_binary_connect_convolution(x, w, wb, b, dy, base_axis, pad, stride, dilation, group, quantize_zero_to, **kw):
    # Set variables
    vx = nn.Variable(x.shape, need_grad=True)
    vx.d = x
    vx.grad.zero()
    vw = nn.Variable(w.shape, need_grad=True)
    vw.d = binarize(w, quantize_zero_to)
    vw.grad.zero()
    vb = None
    if b is not None:
        vb = nn.Variable(b.shape, need_grad=True)
        vb.d = b
        vb.grad.zero()

    # Execute binarized forward and back prop.
    with nn.auto_forward():
        vy = F.convolution(vx, vw, vb, base_axis, pad, stride, dilation, group)
    vy.backward(dy)

    # Return grads
    if b is None:
        return np.concatenate([vx.g.flat, vw.g.flat])
    return np.concatenate([vx.g.flat, vw.g.flat, vb.g.flat])


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("inshape, kernel, outmaps, pad, stride, dilation",
                         [((2, 2, 10, 10), (3, 2), 4, (3, 0), (1, 2), (2, 1)), ])
@pytest.mark.parametrize("group", [1, 2])
@pytest.mark.parametrize("with_bias", [True, False])
@pytest.mark.parametrize("quantize_zero_to", [0.0, -1.0, 1.0])
def test_convolution_2d_forward_backward(inshape, kernel, outmaps, pad, stride,
                                         dilation, group, with_bias, quantize_zero_to, seed, ctx,
                                         func_name):
    from nbla_test_utils import function_tester
    rng = np.random.RandomState(seed)
    i = rng.randn(*inshape).astype(np.float32)
    inmaps = inshape[-3]
    kshape = (outmaps,) + (inmaps // group,) + kernel
    k = rng.randn(*kshape).astype(np.float32)
    base_axis = len(inshape) - 3
    b = None
    if with_bias:
        b = rng.randn(outmaps).astype(np.float32)
    inputs = [i, k, np.zeros_like(k), b]
    function_tester(rng, F.binary_connect_convolution, ref_binary_connect_convolution, inputs,
                    func_args=[base_axis, pad, stride,
                               dilation, group, quantize_zero_to],
                    atol_f=1e-4, atol_b=3e-3, atol_accum=1e-5, dstep=1e-2, backward=[True, True, False, True],
                    ctx=ctx, func_name=func_name, ref_grad=ref_grad_binary_connect_convolution)
