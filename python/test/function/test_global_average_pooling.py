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

from nbla_test_utils import (
    function_tester,
    list_ctx_and_func_name)


def ref_global_average_pooling(x):
    xs = x.shape
    newshape = (xs[0], xs[1], xs[2]*xs[3])
    newx = np.reshape(x, newshape)
    return np.average(newx, 2)[:, :, np.newaxis, np.newaxis]


@pytest.mark.parametrize("seed", [314])
@pytest.mark.parametrize("fname, ctx, func_name", list_ctx_and_func_name(['global_average_pooling']))
def test_global_average_pooling_forward_backward(seed, fname, ctx, func_name):
    rng = np.random.RandomState(seed)
    ref_func = eval('ref_' + fname)
    func = getattr(F, fname)
    inputs = [rng.random_sample((2, 3, 4, 5))]
    function_tester(rng, func, ref_func, inputs, [],
                    ctx=ctx, func_name=func_name)


@pytest.mark.parametrize("seed", [314])
@pytest.mark.parametrize("fname, ctx, func_name", list_ctx_and_func_name(['global_average_pooling']))
def test_global_average_pooling_double_backward(seed, fname, ctx, func_name):
    from nbla_test_utils import backward_function_tester, grad_function_forward_function_output
    from nnabla.backward_function.global_average_pooling import GlobalAveragePoolingDataGrad
    rng = np.random.RandomState(seed)
    ref_func = eval('ref_' + fname)
    func = getattr(F, fname)
    inputs = [rng.random_sample((2, 3, 4, 5))]
    # 2nd-order
    backward_function_tester(rng, func, inputs, ctx=ctx)

    # 3rd-order
    df, y = grad_function_forward_function_output(GlobalAveragePoolingDataGrad,
                                                  F.global_average_pooling,
                                                  ctx, inputs)
    df.xshape = inputs[0].shape
    ginputs = [rng.randn(*y.shape)]
    backward_function_tester(rng, df,
                             inputs=ginputs,
                             ctx=ctx, atol_f=1e-6, atol_accum=1e-2, non_accum_check=True)
