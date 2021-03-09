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
import nnabla.functions as F
import refs
from nbla_test_utils import list_context

ctxs = list_context('Deconvolution')


def ref_deconvolution_2d(x, w, b, base_axis, pad, stride, dilation, group,
                         channel_last=False, output_padding=(0, 0)):
    if channel_last:
        transpose_x = refs.ChannelLastToFirstTranspose(x.ndim, len(pad))
        transpose_w = refs.ChannelLastToFirstTranspose(w.ndim, len(pad))
        return transpose_x.inv(
                ref_deconvolution_2d(transpose_x(x), transpose_w(w), b,
                                     base_axis, pad, stride, dilation, group,
                                     False, output_padding))

    y = []
    for xx in x.reshape((-1,) + x.shape[base_axis:]):
        y += [refs.deconvolution_2d(xx, w, b, pad, stride, dilation, group,
                                    output_padding=output_padding)[np.newaxis]]
    y = np.vstack(y)
    return y.reshape(x.shape[:base_axis] + y.shape[1:])


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("inshape", [(2, 2, 4, 5), (2, 1, 4, 5, 4)])
@pytest.mark.parametrize("kernel, outmaps, pad", [
    ((3, 3), 2, (0, 1)),
    ((1, 3), 4, (1, 2)),
])
@pytest.mark.parametrize("stride, output_padding", [
    ((1, 1), (0, 0)),
    ((2, 2), (0, 0)),
    ((2, 2), (1, 1)),
])
@pytest.mark.parametrize("dilation", [(1, 1), (2, 2)])
@pytest.mark.parametrize("group", [1, 2])
@pytest.mark.parametrize("with_bias", [True, False])
@pytest.mark.parametrize("channel_last", [False, True])
def test_deconvolution_2d_forward_backward(inshape, kernel, outmaps, pad,
                                           stride, dilation, group, with_bias,
                                           channel_last, output_padding,
                                           seed, ctx, func_name):
    from nbla_test_utils import function_tester

    if channel_last and not func_name.endswith('Cudnn'):
        pytest.skip('channel_last=True is only supported in CUDNN backend.')
    base_axis = len(inshape) - len(kernel) - 1
    inmaps = inshape[base_axis]
    if channel_last:
        t = refs.ChannelLastToFirstTranspose(len(inshape), len(kernel))
        inshape = tuple(inshape[i] for i in t.inv_axes)
    rng = np.random.RandomState(seed)
    i = np.clip(rng.randn(*inshape).astype(np.float32), -0.5, 0.5)
    kshape = (inmaps,) + (outmaps // group,) + kernel
    if channel_last:
        t = refs.ChannelLastToFirstTranspose(len(kshape), len(kernel))
        kshape = tuple(kshape[i] for i in t.inv_axes)
    k = np.clip(rng.randn(*kshape).astype(np.float32), -0.5, 0.5)
    base_axis = len(inshape) - 3
    b = None
    if with_bias:
        b = np.clip(rng.randn(outmaps).astype(np.float32), -0.5, 0.5)
    inputs = [i, k, b]
    func_args = [base_axis, pad, stride, dilation, group, channel_last,
                 output_padding]
    function_tester(rng, F.deconvolution, ref_deconvolution_2d, inputs,
                    func_args=func_args, func_name=func_name, ctx=ctx,
                    atol_f=1e-4, atol_b=1e-2, atol_accum=1e-5, dstep=1e-2)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("inshape", [(2, 2, 4, 5), (2, 1, 4, 5, 4)])
@pytest.mark.parametrize("kernel, outmaps, pad", [
    ((3, 3), 2, (0, 1)),
    ((1, 3), 4, (1, 2))
])
@pytest.mark.parametrize("stride, output_padding", [
    ((1, 1), (0, 0)),
    ((2, 2), (0, 0)),
    ((2, 2), (1, 1)),
])
@pytest.mark.parametrize("dilation", [(1, 1), (2, 2)])
@pytest.mark.parametrize("group", [1, 2])
@pytest.mark.parametrize("channel_last", [False, True])
@pytest.mark.parametrize("with_bias", [True, False])
def test_deconvolution_2d_double_backward(inshape, kernel, outmaps, pad,
                                          stride, dilation, group, with_bias,
                                          channel_last, output_padding,
                                          seed, ctx, func_name):
    from nbla_test_utils import function_tester, backward_function_tester, grad_function_forward_function_output
    from nnabla.backward_function.deconvolution import DeconvolutionDataGrad, DeconvolutionFilterGrad

    if channel_last and not func_name.endswith('Cudnn'):
        pytest.skip('channel_last=True is only supported in CUDNN backend.')
    base_axis = len(inshape) - len(kernel) - 1
    inmaps = inshape[base_axis]
    if channel_last:
        t = refs.ChannelLastToFirstTranspose(len(inshape), len(kernel))
        inshape = tuple(inshape[i] for i in t.inv_axes)
    rng = np.random.RandomState(seed)
    i = np.clip(rng.randn(*inshape).astype(np.float32), -0.5, 0.5)
    kshape = (inmaps,) + (outmaps // group,) + kernel
    if channel_last:
        t = refs.ChannelLastToFirstTranspose(len(kshape), len(kernel))
        kshape = tuple(kshape[i] for i in t.inv_axes)
    k = np.clip(rng.randn(*kshape).astype(np.float32), -0.5, 0.5)
    base_axis = len(inshape) - 3
    b = None
    if with_bias:
        b = np.clip(rng.randn(outmaps).astype(np.float32), -0.5, 0.5)
    inputs = [i, k, b]
    func_args = [base_axis, pad, stride, dilation, group, channel_last,
                 output_padding]
    # Deconvolution
    backward_function_tester(rng, F.deconvolution,
                             inputs, func_args=func_args, ctx=ctx, atol_accum=1e-1)

    # DataGrad
    df, y = grad_function_forward_function_output(DeconvolutionDataGrad,
                                                  F.deconvolution,
                                                  ctx, inputs, *func_args)
    df.xshape = i.shape
    ginputs = [rng.randn(*y.shape), k]
    backward_function_tester(rng, df, ginputs, ctx=ctx, atol_accum=1e-1,
                             non_accum_check=True)

    # FilterGrad
    df, y = grad_function_forward_function_output(DeconvolutionFilterGrad,
                                                  F.deconvolution,
                                                  ctx, inputs, *func_args)
    df.wshape = k.shape
    ginputs = [rng.randn(*y.shape), i]
    backward_function_tester(rng, df, ginputs, func_args=[], ctx=ctx, atol_accum=1e-1,
                             non_accum_check=True)
