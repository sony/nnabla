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
import refs
from nbla_test_utils import list_context

ctxs = list_context('Unpooling')


def ref_unpooling(x, kernel, channel_last):
    if channel_last:
        n_sdim = len(kernel)
        t = refs.ChannelLastToFirstTranspose(x.ndim, n_sdim)
        x = t(x)
    y = x
    for ind, p in enumerate(kernel[::-1]):
        y = y.repeat(p, axis=y.ndim - (ind + 1))
    if channel_last:
        y = t.inv(y)
    return y


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("inshape", [(1, 2, 3), (2, 4, 6), (2, 2, 4, 6), (2, 2, 2, 4, 6)])
@pytest.mark.parametrize("kernel", [(1, 1), (2, 3), (2, 1, 2)])
@pytest.mark.parametrize("channel_last", [False, True])
def test_unpooling_forward_backward(seed, inshape, channel_last, kernel, ctx, func_name):
    if channel_last and func_name == "Unpooling":
        pytest.skip("Unpooling with channel_last is only supported in CUDA.")
    if channel_last and len(inshape) == len(kernel):
        pytest.skip(
            "len(input shape) == len(kernel) is only valid for the channel first.")

    from nbla_test_utils import function_tester
    rng = np.random.RandomState(seed)
    inputs = [rng.randn(*inshape).astype(np.float32)]
    function_tester(rng,
                    F.unpooling, ref_unpooling,
                    inputs=inputs, backward=[True],
                    func_args=[kernel, channel_last],
                    ctx=ctx, func_name=func_name,
                    atol_f=1e-6, atol_b=1e-2)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("inshape", [(1, 2, 3), (2, 4, 6), (2, 2, 4, 6), (2, 2, 2, 4, 6)])
@pytest.mark.parametrize("kernel", [(1, 1), (2, 3), (2, 1, 2)])
@pytest.mark.parametrize("channel_last", [False, True])
def test_unpooling_double_backward(seed, inshape, kernel, channel_last, ctx, func_name):
    if channel_last and func_name == "Unpooling":
        pytest.skip("Unpooling with channel_last is only supported in CUDA.")
    if channel_last and len(inshape) == len(kernel):
        pytest.skip(
            "len(input shape) == len(kernel) is only valid for the channel first.")

    from nbla_test_utils import backward_function_tester, grad_function_forward_function_output
    from nnabla.backward_function.unpooling import UnpoolingDataGrad
    rng = np.random.RandomState(seed)
    inputs = [rng.randn(*inshape).astype(np.float32)]
    func_args = [kernel, channel_last]
    # 2nd-order
    backward_function_tester(rng,
                             F.unpooling,
                             inputs=inputs,
                             func_args=func_args,
                             ctx=ctx)
    # 3rd-order
    df, y = grad_function_forward_function_output(UnpoolingDataGrad,
                                                  F.unpooling,
                                                  ctx, inputs,
                                                  *func_args)
    df.xshape = inputs[0].shape
    ginputs = [rng.randn(*y.shape)]
    backward_function_tester(rng, df,
                             inputs=ginputs, ctx=ctx, non_accum_check=True)
