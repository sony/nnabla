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

ctxs = list_context('INQConvolution')


def quantize(x, max_absval, num_bits):
    y = x

    # get maximum/minimum exponent
    n1 = np.floor(np.log2(max_absval)) + (np.log2(max_absval) -
                                          np.floor(np.log2(max_absval)) >= np.log2(1.5))
    n2 = n1 + 1 - 2 ** (num_bits - 2)

    pruning_threshold = 2 ** (n2 - 1)

    # prune all small values
    y[np.abs(x) < pruning_threshold] = 0.0

    # quantize remaining values to powers of two
    _i = y != 0
    _s = np.sign(y[_i])
    _b = np.log2(np.abs(y[_i]))
    _d = np.log2(1.5)  # quantization threshold
    # _d = 0.5 use geometric mean
    # _d = np.log2(1.5) use arithmetic mean
    _e = np.floor(_b) + (_b - np.floor(_b) >= _d)
    _e = np.maximum(n2, np.minimum(n1, _e))
    y[_i] = _s * 2 ** (_e)
    return y


def ref_inq_convolution(x, w, i, b, base_axis, pad, stride, dilation, group,
                        num_bits, inq_iterations, selection_algorithm, seed):

    if inq_iterations[-1] == 0:
        # last element in `inq_iterations`, quantize all weights
        i = np.ones_like(i)
    elif 0 in inq_iterations:
        # only `largest_abs` is deterministic and currently tested
        assert(selection_algorithm == 'largest_abs')
        idx_var = np.flatnonzero(i == 0)
        idx_newfix = idx_var[np.argsort(
            np.abs(w.ravel()[idx_var]))[-(len(idx_var) // 2):]]
        i.ravel()[idx_newfix] = 1

    wq = np.copy(w)
    if np.any(i == 1):
        wq[i == 1] = quantize(w[i == 1], np.max(np.abs(w)), num_bits)

    y = []
    for xx in x.reshape((-1,) + x.shape[base_axis:]):
        y += [refs.convolution_2d(xx, wq, b, pad, stride,
                                  dilation, group)[np.newaxis]]
    y = np.vstack(y)
    return y.reshape(x.shape[:base_axis] + y.shape[1:])


def ref_grad_inq_convolution(x, w, i, b, dy, base_axis, pad, stride, dilation, group,
                             num_bits, inq_iterations, selection_algorithm, seed, **kw):
    if inq_iterations[-1] == 0:
        # last element in `inq_iterations`, quantize all weights
        i = np.ones_like(i)
    elif 0 in inq_iterations:
        # only `largest_abs` is deterministic
        assert(selection_algorithm == 'largest_abs')
        idx_var = np.flatnonzero(i == 0)
        idx_newfix = idx_var[np.argsort(
            np.abs(w.ravel()[idx_var]))[-(len(idx_var) // 2):]]
        i.ravel()[idx_newfix] = 1

    wq = np.copy(w)
    if np.any(i == 1):
        wq[i == 1] = quantize(w[i == 1], np.max(np.abs(w)), num_bits)

    # Set variables
    vx = nn.Variable(x.shape, need_grad=True)
    vx.d = x
    vx.grad.zero()
    vw = nn.Variable(w.shape, need_grad=True)
    vw.d = wq
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
@pytest.mark.parametrize("inshape, kernel, outmaps, pad, stride, dilation, num_bits",
                         [((2, 2, 10, 10), (3, 2), 4, (3, 0), (1, 2), (2, 1), 3), ])
@pytest.mark.parametrize("group", [1, 2])
@pytest.mark.parametrize("with_bias", [True, False])
@pytest.mark.parametrize("inq_iterations", [(10, 20), (0,), (0, 10)])
def test_convolution_2d_forward_backward(inshape, kernel, outmaps, pad, stride,
                                         dilation, group, with_bias, seed,
                                         num_bits, inq_iterations, ctx, func_name):
    from nbla_test_utils import function_tester
    rng = np.random.RandomState(seed)
    # Input
    inputs = [rng.randn(*inshape).astype(np.float32)]
    # Weights
    inmaps = inshape[-3]
    kshape = (outmaps,) + (inmaps // group,) + kernel
    inputs += [rng.randn(*kshape).astype(np.float32)]
    # Indices
    inputs += [np.random.randint(2, size=kshape)]
    # Bias
    if with_bias:
        inputs += [rng.randn(outmaps).astype(np.float32)]
    else:
        inputs += [None]

    base_axis = len(inshape) - 3
    selection_algorithm = 'largest_abs'

    function_tester(rng, F.inq_convolution, ref_inq_convolution, inputs,
                    func_args=[base_axis, pad, stride, dilation, group, num_bits, inq_iterations,
                               selection_algorithm, seed],
                    atol_f=1e-5, atol_b=1e-2, atol_accum=1e-5, backward=[True, True, False, True], ctx=ctx, func_name=func_name,
                    ref_grad=ref_grad_inq_convolution)
