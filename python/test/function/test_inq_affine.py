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

ctxs = list_context('INQAffine')


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


def ref_inq_affine(x, w, i, b, base_axis, num_bits,
                   inq_iterations, selection_algorithm, seed):
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

    shape = list(x.shape[:base_axis])
    shape += [-1]
    out_shape = w.shape[1:]

    # quantize weights (0 ... learnable, 1 ... fixed)
    wq = np.copy(w)
    if np.any(i == 1):
        wq[i == 1] = quantize(w[i == 1], np.max(np.abs(w)), num_bits)
    wq = wq.reshape(w.shape[0], -1)

    y = np.dot(x.reshape(*shape), wq)
    if b is not None:
        y += b.reshape((1,) * (len(shape) - 1) + (-1,))
    return y.reshape(tuple(shape[:-1]) + tuple(out_shape))


def ref_grad_inq_affine(x, w, i, b, dy, base_axis, num_bits,
                        inq_iterations, selection_algorithm, seed, **kw):
    shape = list(x.shape[:base_axis])

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

    x_ = x.reshape(np.prod(shape), -1)

    wq_ = np.copy(w)
    if np.any(i == 1):
        wq_[i == 1] = quantize(w[i == 1], np.max(np.abs(w)), num_bits)
    wq_ = wq_.reshape(w.shape[0], -1)

    dy_ = dy.reshape(np.prod(shape), -1)

    dx = np.dot(dy_, np.transpose(wq_))
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
@pytest.mark.parametrize("base_axis, weight_shape, num_bits",
                         [(1, (12, 2, 3), 2), (2, (4, 4), 4)])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("inq_iterations", [(10, 20), (0,), (0, 10)])
def test_inq_affine_forward_backward(seed, base_axis, weight_shape, num_bits,
                                     bias, inq_iterations, ctx, func_name):
    from nbla_test_utils import function_tester
    rng = np.random.RandomState(seed)
    # Input
    inputs = [rng.randn(2, 3, 4).astype(np.float32)]
    # Weights
    inputs += [rng.randn(*weight_shape).astype(np.float32)]
    # Indices
    inputs += [np.random.randint(2, size=weight_shape)]
    # Bias
    if bias:
        inputs += [rng.randn(*weight_shape[1:]).astype(np.float32)]
    else:
        inputs += [None]

    selection_algorithm = 'largest_abs'

    function_tester(rng, F.inq_affine, ref_inq_affine, inputs,
                    func_args=[base_axis, num_bits,
                               inq_iterations, selection_algorithm, seed],
                    atol_b=1e-2, backward=[True, True, False, True], ctx=ctx, func_name=func_name,
                    ref_grad=ref_grad_inq_affine)
