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
import refs
from nbla_test_utils import list_context

ctxs = list_context('MaxPooling')


def ref_max_pooling_2d(x, kernel, stride, ignore_border, pad):
    y = []
    for xx in x.reshape((-1,) + x.shape[-3:]):
        if xx.ndim == 2:
            xx = xx[np.newaxis]
        y += [refs.pooling_2d(xx, 'max', kernel, stride, pad,
                              ignore_border)[np.newaxis]]
    y = np.vstack(y)
    if x.ndim == 2:
        y = np.squeeze(y, 1)
    return y.reshape(x.shape[:-3] + y.shape[1:])


def ref_max_pooling_3d(x, kernel, stride, ignore_border, pad):
    y = []
    for xx in x.reshape((-1,) + x.shape[-4:]):
        if xx.ndim == 3:
            xx = xx[np.newaxis]
        y += [refs.pooling_3d(xx, 'max', kernel, stride, pad,
                              ignore_border)[np.newaxis]]
    y = np.vstack(y)
    if x.ndim == 3:
        y = np.squeeze(y, 1)
    print(y.reshape(x.shape[:-4] + y.shape[1:]).shape)
    return y.reshape(x.shape[:-4] + y.shape[1:])


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("ignore_border", [True, False])
@pytest.mark.parametrize("inshape, kernel, stride, pad", [
    ((2, 3), (2, 3), (2, 3), (0, 0)),
    ((2, 3), (2, 3), (1, 1), (0, 0)),
    ((2, 3), (2, 3), (2, 3), (1, 1)),
    ((2, 4, 6), (2, 3), (2, 3), (0, 0)),
    ((2, 4, 7), (2, 4), (2, 3), (1, 2)),
    ((2, 9, 9), (2, 2), (3, 3), (1, 1)),
    ((2, 2, 4, 6), (2, 2), (2, 1), (0, 0)),
    ((2, 2, 2, 4, 6), (2, 2), (1, 2), (0, 0)),
])
def test_max_pooling_2d(seed, inshape, kernel, stride, pad, ignore_border,
                        ctx, func_name):
    from nbla_test_utils import function_tester
    rng = np.random.RandomState(seed)
    inputs = [rng.randn(*inshape).astype(np.float32)]
    func_args = [kernel, stride, ignore_border, pad]
    function_tester(rng, F.max_pooling, ref_max_pooling_2d, inputs=inputs,
                    func_args=func_args, func_name=func_name, ctx=ctx,
                    atol_f=1e-6, atol_b=1e-2)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("ignore_border", [True, False])
@pytest.mark.parametrize("inshape, kernel, stride, pad", [
    ((2, 3, 4), (2, 3, 4), (2, 3, 4), (0, 0, 0)),
    ((2, 3, 4), (2, 3, 4), (1, 1, 1), (0, 0, 0)),
    ((2, 3, 4), (2, 3, 4), (2, 3, 4), (1, 1, 1)),
    ((2, 3, 4, 6), (2, 3, 4), (2, 3, 4), (0, 0, 0)),
    ((2, 3, 4, 7), (2, 4, 5), (2, 3, 2), (1, 2, 1)),
    ((2, 3, 9, 9), (2, 2, 2), (3, 3, 3), (1, 1, 1)),
    ((2, 2, 3, 4, 6), (2, 2, 2), (2, 1, 1), (0, 0, 0)),
    ((2, 2, 2, 3, 4, 6), (2, 2, 2), (1, 1, 2), (0, 0, 0)),
])
def test_max_pooling_3d(seed, inshape, kernel, stride, pad, ignore_border,
                        ctx, func_name):
    from nbla_test_utils import function_tester
    rng = np.random.RandomState(seed)
    inputs = [rng.randn(*inshape).astype(np.float32)]
    func_args = [kernel, stride, ignore_border, pad]
    function_tester(rng, F.max_pooling, ref_max_pooling_3d, inputs=inputs,
                    func_args=func_args, func_name=func_name, ctx=ctx,
                    atol_f=1e-6, atol_b=1e-2)
