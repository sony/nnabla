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

ctxs = list_context('SumPooling')


def ref_sum_pooling(x, kernel, stride, ignore_border, pad):
    # Only 2d
    y = []
    for xx in x.reshape((-1,) + x.shape[-3:]):
        if xx.ndim == 2:
            xx = xx[np.newaxis]
        y += [refs.pooling_2d(xx, 'sum', kernel, stride,
                              pad, ignore_border)[np.newaxis]]
    y = np.vstack(y)
    if x.ndim == 2:
        y = np.squeeze(y, 1)
    return y.reshape(x.shape[:-3] + y.shape[1:])


'''
27th/sep/2016
Do not use the conditions of
 1. kernel > stride & inshape % stride != 0 and ignore_border=false
 2. inshape == kernel & stride < inshape & ignore_border = false
 -- Theano-0.8.2 looks like incorrect.
'''


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("inshape", [(2, 3), (2, 4, 6), (2, 2, 4, 6), (2, 2, 2, 4, 6)])
# pool shape might be smaller than inshape in theano's pool_2d
@pytest.mark.parametrize("kernel", [(2, 3)])
@pytest.mark.parametrize("stride", [(2, 3)])
# pad must be 0 when ignore_border=false
@pytest.mark.parametrize("pad, ignore_border", [((0, 0), False), ((1, 2), True)])
def test_sum_pooling_forward_backward(seed, inshape, kernel, stride, pad, ignore_border, ctx, func_name):
    from nbla_test_utils import function_tester
    rng = np.random.RandomState(seed)
    inputs = [rng.randn(*inshape).astype(np.float32)]
    function_tester(rng,
                    F.sum_pooling, ref_sum_pooling,
                    inputs=inputs,
                    func_args=[kernel, stride, ignore_border, pad],
                    ctx=ctx, func_name=func_name,
                    atol_f=1e-6, atol_b=1e-2)
