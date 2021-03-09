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

ctxs = list_context('SoftmaxCrossEntropy')


def ref_softmax_cross_entropy(x, l, axis):
    orig_x = x.copy()
    x = x - x.max(axis, keepdims=True)
    x = np.exp(x) / np.exp(x).sum(axis, keepdims=True)
    x = np.rollaxis(x, axis, x.ndim).reshape(-1, x.shape[axis])
    ll = np.rollaxis(l, axis, x.ndim).flatten()
    y = - \
        np.log(
            np.maximum(x[np.arange(x.shape[0]), ll],
                       np.finfo(np.float32).tiny))
    return y.reshape(l.shape)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [314])
@pytest.mark.parametrize("axis", [0, 1, 2, -1, -2, -3])
def test_softmax_cross_entropy_forward_backward(seed, axis, ctx, func_name):
    from nbla_test_utils import function_tester
    ishape = [2, 3, 4]
    rng = np.random.RandomState(seed)

    l_shape = list(ishape)
    l_shape[axis] = 1
    n_class = ishape[axis]

    inputs = [
        rng.randn(2, 3, 4).astype(np.float32) * 2,
        rng.randint(0, n_class, size=l_shape).astype(np.int)]

    function_tester(rng, F.softmax_cross_entropy, ref_softmax_cross_entropy,
                    inputs, func_args=[axis], backward=[True, False],
                    atol_b=2e-3, ctx=ctx, func_name=func_name)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [314])
@pytest.mark.parametrize("axis", [0, 1, 2, -1, -2, -3])
def test_softmax_cross_entropy_double_backward(seed, axis, ctx, func_name):
    from nbla_test_utils import backward_function_tester
    ishape = [2, 3, 4]
    rng = np.random.RandomState(seed)

    l_shape = list(ishape)
    l_shape[axis] = 1
    n_class = ishape[axis]

    inputs = [
        rng.randn(2, 3, 4).astype(np.float32) * 2,
        rng.randint(0, n_class, size=l_shape).astype(np.int)]

    backward_function_tester(rng, F.softmax_cross_entropy,
                             inputs, func_args=[axis], backward=[True, False],
                             atol_accum=1e-3,
                             dstep=1e-3,
                             ctx=ctx)
