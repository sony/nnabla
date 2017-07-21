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

ctxs = list_context('Deconvolution')


def ref_deconvolution(x, w, b, base_axis, pad, stride, dilation, group):
    y = []
    for xx in x.reshape((-1,) + x.shape[base_axis:]):
        y += [refs.deconvolution_2d(xx, w, b, pad, stride,
                                    dilation, group)[np.newaxis]]
    y = np.vstack(y)
    return y.reshape(x.shape[:base_axis] + y.shape[1:])


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("inshape", [(2, 2, 4, 5), (2, 1, 4, 5, 4)])
@pytest.mark.parametrize(
    "kernel,outmaps,pad", [((3, 3), 2, (0, 1)), ((1, 3), 4, (1, 2))])
@pytest.mark.parametrize("stride", [(1, 1), (2, 2)])
@pytest.mark.parametrize("dilation", [(1, 1), (2, 2)])
@pytest.mark.parametrize("group", [1, 2])
@pytest.mark.parametrize("with_bias", [True, False])
def test_deconvolution_2d_forward_backward(inshape, kernel, outmaps, pad, stride,
                                           dilation, group, with_bias, seed, ctx,
                                           func_name):
    from nbla_test_utils import function_tester

    rng = np.random.RandomState(seed)
    i = rng.randn(*inshape).astype(np.float32)
    inmaps = inshape[-3]
    kshape = (inmaps,) + (outmaps // group,) + kernel
    k = rng.randn(*kshape).astype(np.float32)
    base_axis = len(inshape) - 3
    b = None
    if with_bias:
        b = rng.randn(outmaps).astype(np.float32)
    inputs = [i, k, b]
    function_tester(rng, F.deconvolution, ref_deconvolution, inputs,
                    func_args=[base_axis, pad, stride, dilation, group],
                    atol_f=1e-4, atol_b=2e-3, dstep=1e-2,
                    ctx=ctx, func_name=func_name)
