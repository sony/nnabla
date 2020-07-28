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

ctxs = list_context('SumPooling')


def ref_sum_pooling_2d(x, kernel, stride, ignore_border, pad):
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


def ref_sum_pooling_3d(x, kernel, stride, ignore_border, pad):
    y = []
    for xx in x.reshape((-1,) + x.shape[-4:]):
        if xx.ndim == 3:
            xx = xx[np.newaxis]
        y += [refs.pooling_3d(xx, 'sum', kernel, stride,
                              pad, ignore_border)[np.newaxis]]
    y = np.vstack(y)
    if x.ndim == 3:
        y = np.squeeze(y, 1)
    return y.reshape(x.shape[:-4] + y.shape[1:])


def ref_sum_pooling(x, kernel, stride, ignore_border, pad, channel_last):
    if channel_last:
        t = refs.ChannelLastToFirstTranspose(x.ndim, len(kernel))
        x = t(x)
        y = ref_sum_pooling(
            x, kernel, stride, ignore_border, pad, False)
        return t.inv(y)
    if len(kernel) == 3:
        y = ref_sum_pooling_3d(
            x, kernel, stride, ignore_border, pad)
        return y
    y = ref_sum_pooling_2d(
        x, kernel, stride, ignore_border, pad)
    return y


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("ignore_border", [True, False])
@pytest.mark.parametrize("channel_last", [False, True])
@pytest.mark.parametrize("inshape, kernel, stride, pad", [
    ((2, 2, 4, 6), (2, 2), (2, 1), (1, 0)),
    ((2, 2, 2, 4, 6), (2, 2), (1, 2), (0, 1)),
])
def test_sum_pooling_2d(seed, inshape, kernel, stride, pad, ignore_border, channel_last,
                        ctx, func_name):
    from nbla_test_utils import function_tester
    if channel_last and not func_name.endswith('Cudnn'):
        pytest.skip('Channel last is only supported in Cudnn so far')
    if channel_last:
        t = refs.ChannelLastToFirstTranspose(len(inshape), len(kernel))
        inshape = tuple(inshape[i] for i in t.inv_axes)
    if not ignore_border and func_name.endswith('Cudnn'):
        pytest.skip('ignore_border=False in Cudnn is not supported.')
    rng = np.random.RandomState(seed)
    inputs = [rng.randn(*inshape).astype(np.float32)]
    func_args = [kernel, stride, ignore_border, pad, channel_last]
    function_tester(rng, F.sum_pooling, ref_sum_pooling, inputs=inputs,
                    func_args=func_args, func_name=func_name, ctx=ctx,
                    atol_f=1e-6, atol_b=1e-2)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("ignore_border", [True, False])
@pytest.mark.parametrize("channel_last", [False, True])
@pytest.mark.parametrize("inshape, kernel, stride, pad", [
    ((2, 2, 3, 4, 6), (2, 2, 2), (2, 1, 1), (1, 0, 1)),
    ((2, 2, 2, 3, 4, 6), (2, 2, 2), (1, 1, 2), (0, 1, 0)),
])
def test_sum_pooling_3d(seed, inshape, kernel, stride, pad, ignore_border, channel_last,
                        ctx, func_name):
    from nbla_test_utils import function_tester
    if channel_last and not func_name.endswith('Cudnn'):
        pytest.skip('Channel last is only supported in Cudnn so far')
    if channel_last:
        t = refs.ChannelLastToFirstTranspose(len(inshape), len(kernel))
        inshape = tuple(inshape[i] for i in t.inv_axes)
    if not ignore_border and func_name.endswith('Cudnn'):
        pytest.skip('ignore_border=False in Cudnn is not supported.')
    rng = np.random.RandomState(seed)
    inputs = [rng.randn(*inshape).astype(np.float32)]
    func_args = [kernel, stride, ignore_border, pad, channel_last]
    function_tester(rng, F.sum_pooling, ref_sum_pooling, inputs=inputs,
                    func_args=func_args, func_name=func_name, ctx=ctx,
                    atol_f=1e-6, atol_b=5e-2, atol_accum=1e-2)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("ignore_border", [True, False])
@pytest.mark.parametrize("channel_last", [False, True])
@pytest.mark.parametrize("inshape, kernel, stride, pad", [
    ((2, 2, 4, 6), (2, 2), (2, 1), (1, 0)),
    ((2, 2, 2, 4, 6), (2, 2), (1, 2), (0, 1)),
])
def test_sum_pooling_2d_double_backward(seed, inshape, kernel, stride, pad, ignore_border, channel_last,
                                        ctx, func_name):
    from nbla_test_utils import backward_function_tester, grad_function_forward_function_output
    from nnabla.backward_function.sum_pooling import SumPoolingDataGrad
    if channel_last and not func_name.endswith('Cudnn'):
        pytest.skip('Channel last is only supported in Cudnn so far')
    if channel_last:
        t = refs.ChannelLastToFirstTranspose(len(inshape), len(kernel))
        inshape = tuple(inshape[i] for i in t.inv_axes)
    if not ignore_border and func_name.endswith('Cudnn'):
        pytest.skip('ignore_border=False in Cudnn is not supported.')
    rng = np.random.RandomState(seed)
    inputs = [rng.randn(*inshape).astype(np.float32)]
    func_args = [kernel, stride, ignore_border, pad, channel_last]
    # 2nd-order
    backward_function_tester(rng, F.sum_pooling, inputs=inputs,
                             func_args=func_args, ctx=ctx)

    # 3rd-order
    df, y = grad_function_forward_function_output(SumPoolingDataGrad,
                                                  F.sum_pooling,
                                                  ctx, inputs,
                                                  *func_args)
    df.xshape = inputs[0].shape
    ginputs = [rng.randn(*y.shape)]
    backward_function_tester(rng, df,
                             inputs=ginputs, func_args=[],
                             ctx=ctx)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("ignore_border", [True, False])
@pytest.mark.parametrize("channel_last", [False, True])
@pytest.mark.parametrize("inshape, kernel, stride, pad", [
    ((2, 2, 3, 4, 6), (2, 2, 2), (2, 1, 1), (1, 0, 1)),
    ((2, 2, 2, 3, 4, 6), (2, 2, 2), (1, 1, 2), (0, 1, 0)),
])
def test_sum_pooling_3d_double_backward(seed, inshape, kernel, stride, pad, ignore_border, channel_last,
                                        ctx, func_name):
    from nbla_test_utils import backward_function_tester, grad_function_forward_function_output
    from nnabla.backward_function.sum_pooling import SumPoolingDataGrad
    if channel_last and not func_name.endswith('Cudnn'):
        pytest.skip('Channel last is only supported in Cudnn so far')
    if channel_last:
        t = refs.ChannelLastToFirstTranspose(len(inshape), len(kernel))
        inshape = tuple(inshape[i] for i in t.inv_axes)
    if not ignore_border and func_name.endswith('Cudnn'):
        pytest.skip('ignore_border=False in Cudnn is not supported.')
    rng = np.random.RandomState(seed)
    inputs = [rng.randn(*inshape).astype(np.float32)]
    func_args = [kernel, stride, ignore_border, pad, channel_last]
    # 2nd-order
    backward_function_tester(rng, F.sum_pooling, inputs=inputs,
                             func_args=func_args, ctx=ctx)
    # 3rd-order
    df, y = grad_function_forward_function_output(SumPoolingDataGrad,
                                                  F.sum_pooling,
                                                  ctx, inputs,
                                                  *func_args)
    df.xshape = inputs[0].shape
    ginputs = [rng.randn(*y.shape)]
    backward_function_tester(rng, df,
                             inputs=ginputs, ctx=ctx, atol_accum=3e-2, non_accum_check=True)
