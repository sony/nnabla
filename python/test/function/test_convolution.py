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

ctxs = list_context('Convolution')


def ref_convolution(x, w, b, base_axis, pad, stride, dilation, group, channel_last):
    if channel_last:
        t = refs.ChannelLastToFirstTranspose(x.ndim, len(pad))
        x = t(x)
        tw = refs.ChannelLastToFirstTranspose(w.ndim, len(pad))
        w = tw(w)
        y = ref_convolution(x, w, b, base_axis, pad,
                            stride, dilation, group, False)
        return t.inv(y)
    y = []
    for xx in x.reshape((-1,) + x.shape[base_axis:]):
        y += [refs.convolution_nd(xx, w, b, pad, stride,
                                  dilation, group)[np.newaxis]]
    y = np.vstack(y)
    return y.reshape(x.shape[:base_axis] + y.shape[1:])


def core_test_convolution_forward_backward(inshape, kernel, outmaps, pad, stride,
                                           dilation, group, channel_last, with_bias, seed, ctx,
                                           func_name):
    from nbla_test_utils import function_tester
    if func_name == 'ConvolutionCuda':
        pytest.skip('CUDA Convolution N-D is only supported in CUDNN extension')
    if channel_last and not func_name.endswith('Cudnn'):
        pytest.skip(
            'channel_last=True is only supported in CUDNN backend so far.')
    if channel_last and func_name.endswith('Cudnn') and (np.any(np.asarray(dilation) > 1) or group > 1):
        import nnabla_ext.cuda as nc
        major, minor, revision = map(int, nc.__cudnn_version__.split('.'))
        version = major * 1000 + minor * 100
        if version < 7200:
            pytest.skip(
                'channel_last dilated convolution not work in CUDNN {}.'.format(version))

    base_axis = len(inshape) - len(kernel) - 1
    inmaps = inshape[base_axis]
    if channel_last:
        t = refs.ChannelLastToFirstTranspose(len(inshape), len(kernel))
        inshape = tuple(inshape[i] for i in t.inv_axes)
    rng = np.random.RandomState(seed)
    i = rng.randn(*inshape).astype(np.float32)
    kshape = (outmaps,) + (inmaps // group,) + kernel
    if channel_last:
        t = refs.ChannelLastToFirstTranspose(len(kshape), len(kernel))
        kshape = tuple(kshape[i] for i in t.inv_axes)
    k = rng.randn(*kshape).astype(np.float32)
    b = None
    if with_bias:
        b = rng.randn(outmaps).astype(np.float32)
    inputs = [i, k, b]
    atol_half = 1.0 if inmaps > 64 else 1e-1
    function_tester(rng, F.convolution, ref_convolution, inputs,
                    func_args=[base_axis, pad, stride,
                               dilation, group, channel_last],
                    atol_f=1e-4, atol_b=1e-2, atol_accum=1e-5, dstep=1e-2,
                    ctx=ctx, func_name=func_name, atol_half=atol_half)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("inshape, kernel, outmaps, pad, stride, dilation", [
    ((2, 2, 10), (1,), 4, (3,), (2,), (1,)),
    ((2, 2, 10), (3,), 2, (0,), (1,), (2,)),
])
@pytest.mark.parametrize("group", [1, 2])
@pytest.mark.parametrize("channel_last", [False, True])
@pytest.mark.parametrize("with_bias", [True, False])
def test_convolution_1d_forward_backward(inshape, kernel, outmaps, pad, stride,
                                         dilation, group, channel_last, with_bias, seed, ctx,
                                         func_name):
    core_test_convolution_forward_backward(inshape, kernel, outmaps, pad, stride,
                                           dilation, group, channel_last, with_bias, seed, ctx,
                                           func_name)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("inshape, kernel, outmaps, pad, stride, dilation", [
    ((2, 2, 10, 10), (3, 2), 4, (3, 0), (1, 2), (2, 1)),
    # ((32, 128, 64, 64), (3, 3), 128, (1, 1), (1, 1), (1, 1)),  # This takes looooooooong to test.
    ((2, 2, 10, 10), (3, 2), 4, (0, 0), (1, 1), (1, 1)),
])
@pytest.mark.parametrize("group", [1, 2])
@pytest.mark.parametrize("channel_last", [False, True])
@pytest.mark.parametrize("with_bias", [True, False])
def test_convolution_2d_forward_backward(inshape, kernel, outmaps, pad, stride,
                                         dilation, group, channel_last, with_bias, seed, ctx,
                                         func_name):
    core_test_convolution_forward_backward(inshape, kernel, outmaps, pad, stride,
                                           dilation, group, channel_last, with_bias, seed, ctx,
                                           func_name)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("inshape, kernel, outmaps, pad, stride, dilation", [
    ((2, 2, 7, 8, 5), (2, 3, 2), 4, (3, 0, 0), (1, 2, 1), (2, 1, 1)),
])
@pytest.mark.parametrize("group", [1, 2])
@pytest.mark.parametrize("channel_last", [False, True])
@pytest.mark.parametrize("with_bias", [True, False])
def test_convolution_3d_forward_backward(inshape, kernel, outmaps, pad, stride,
                                         dilation, group, channel_last, with_bias, seed, ctx,
                                         func_name):
    if channel_last:
        pytest.skip('3d')
    core_test_convolution_forward_backward(inshape, kernel, outmaps, pad, stride,
                                           dilation, group, channel_last, with_bias, seed, ctx,
                                           func_name)


def core_test_convolution_double_backward(inshape, kernel, outmaps, pad, stride,
                                          dilation, group, channel_last, with_bias, seed, ctx,
                                          func_name, non_accum_check=True,
                                          atol_f=1e-4, atol_b=1e-3, atol_accum=8e-2, dstep=1e-3):
    from nbla_test_utils import backward_function_tester, grad_function_forward_function_output
    from nnabla.backward_function.convolution import ConvolutionDataGrad, ConvolutionFilterGrad
    if func_name == 'ConvolutionCuda':
        pytest.skip('CUDA Convolution N-D is only supported in CUDNN extension')
    if channel_last and not func_name.endswith('Cudnn'):
        pytest.skip(
            'channel_last=True is only supported in CUDNN backend so far.')
    if channel_last and func_name.endswith('Cudnn') and (np.any(np.asarray(dilation) > 1) or group > 1):
        import nnabla_ext.cuda as nc
        major, minor, revision = map(int, nc.__cudnn_version__.split('.'))
        version = major * 1000 + minor * 100
        if version < 7200:
            pytest.skip(
                'channel_last dilated convolution not work in CUDNN {}.'.format(version))

    base_axis = len(inshape) - len(kernel) - 1
    inmaps = inshape[base_axis]
    if channel_last:
        t = refs.ChannelLastToFirstTranspose(len(inshape), len(kernel))
        inshape = tuple(inshape[i] for i in t.inv_axes)
    rng = np.random.RandomState(seed)
    i = np.clip(rng.randn(*inshape).astype(np.float32), -0.8, 0.8)
    kshape = (outmaps,) + (inmaps // group,) + kernel
    if channel_last:
        t = refs.ChannelLastToFirstTranspose(len(kshape), len(kernel))
        kshape = tuple(kshape[i] for i in t.inv_axes)
    k = np.clip(rng.randn(*kshape).astype(np.float32), -0.8, 0.8)
    b = None
    if with_bias:
        b = np.clip(rng.randn(outmaps).astype(np.float32), -0.8, 0.8)
    inputs = [i, k, b]
    atol_half = 1.0 if inmaps > 64 else 1e-1
    func_args = [base_axis, pad, stride, dilation, group, channel_last]
    # Convolution
    backward_function_tester(rng, F.convolution, inputs,
                             func_args=func_args,
                             atol_f=atol_f, atol_accum=atol_accum, dstep=dstep,
                             ctx=ctx)
    # DataGrad
    df, y = grad_function_forward_function_output(ConvolutionDataGrad,
                                                  F.convolution,
                                                  ctx, inputs, *func_args)
    df.xshape = i.shape
    ginputs = [rng.randn(*y.shape), k]
    backward_function_tester(rng, df, ginputs,
                             func_args=[],
                             atol_f=atol_f, atol_b=atol_b, atol_accum=atol_accum, dstep=dstep,
                             ctx=ctx, non_accum_check=non_accum_check)

    # FilterGrad
    df, y = grad_function_forward_function_output(ConvolutionFilterGrad,
                                                  F.convolution,
                                                  ctx, inputs, *func_args)
    df.wshape = k.shape
    ginputs = [rng.randn(*y.shape), i]
    backward_function_tester(rng, df, ginputs,
                             func_args=[],
                             atol_f=atol_f, atol_b=atol_b, atol_accum=atol_accum, dstep=dstep,
                             ctx=ctx, non_accum_check=non_accum_check)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("inshape, kernel, outmaps, pad, stride, dilation", [
    ((2, 2, 10), (1,), 4, (3,), (2,), (1,)),
    ((2, 2, 10), (3,), 2, (0,), (1,), (2,)),
])
@pytest.mark.parametrize("group", [1, 2])
@pytest.mark.parametrize("channel_last", [False, True])
@pytest.mark.parametrize("with_bias", [True, False])
def test_convolution_1d_double_backward(inshape, kernel, outmaps, pad, stride,
                                        dilation, group, channel_last, with_bias, seed, ctx,
                                        func_name):
    core_test_convolution_double_backward(inshape, kernel, outmaps, pad, stride,
                                          dilation, group, channel_last, with_bias, seed, ctx,
                                          func_name, non_accum_check=True)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("inshape, kernel, outmaps, pad, stride, dilation", [
    ((2, 2, 10, 10), (3, 2), 4, (3, 0), (1, 2), (2, 1)),
    # ((32, 128, 64, 64), (3, 3), 128, (1, 1), (1, 1), (1, 1)),  # This takes looooooooong to test.
    ((2, 2, 10, 10), (3, 2), 4, (0, 0), (1, 1), (1, 1)),
])
@pytest.mark.parametrize("group", [1, 2])
@pytest.mark.parametrize("channel_last", [False, True])
@pytest.mark.parametrize("with_bias", [True, False])
def test_convolution_2d_double_backward(inshape, kernel, outmaps, pad, stride,
                                        dilation, group, channel_last, with_bias, seed, ctx,
                                        func_name):
    core_test_convolution_double_backward(inshape, kernel, outmaps, pad, stride,
                                          dilation, group, channel_last, with_bias, seed, ctx,
                                          func_name, non_accum_check=True)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("inshape, kernel, outmaps, pad, stride, dilation", [
    ((2, 2, 7, 8, 5), (2, 3, 2), 4, (3, 0, 0), (1, 2, 1), (2, 1, 1)),
])
@pytest.mark.parametrize("group", [1, 2])
@pytest.mark.parametrize("channel_last", [False, True])
@pytest.mark.parametrize("with_bias", [True, False])
def test_convolution_3d_double_backward(inshape, kernel, outmaps, pad, stride,
                                        dilation, group, channel_last, with_bias, seed, ctx,
                                        func_name):
    if channel_last:
        pytest.skip('3d')
    import platform
    if platform.machine() == 'ppc64le':
        pytest.skip('Skip the ppc64le platform temporarily.')
    if platform.system() == "Linux" and platform.uname().machine not in ["x86_64", "ppc64le"]:
        pytest.skip('Convolution 3-D for x86_64 and ppc64 are only supported.')

    core_test_convolution_double_backward(inshape, kernel, outmaps, pad, stride,
                                          dilation, group, channel_last, with_bias, seed, ctx,
                                          func_name, atol_accum=2e-1, non_accum_check=True)
