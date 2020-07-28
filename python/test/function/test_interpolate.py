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

ctxs = list_context('Interpolate')


def compute_scale(isize, osize, align_corners):
    if osize > 1:
        return (isize - 1) / (osize - 1) if align_corners else isize / osize
    else:
        return 0.0


def compute_scale_for_nn(isize, osize, align_corners, half_pixel_for_nn):
    if half_pixel_for_nn:
        return isize / float(osize)
    else:
        return compute_scale(isize, osize, align_corners)


def get_source_index(scale, dst_index, half_pixel):
    return np.maximum(0, scale * (dst_index + 0.5) - 0.5) \
      if half_pixel else scale * dst_index


def get_source_index_for_nn(scale, dst_index, half_pixel, half_pixel_for_nn):
    if half_pixel_for_nn:
        return scale * (dst_index + 0.5)
    else:
        return get_source_index(scale, dst_index, half_pixel)


def ref_interpolate(x, scale, output_size, mode, align_corners=True, half_pixel=False,
                    half_pixel_for_nn=False, channel_last=False):
    assert scale or output_size, 'Need either scale or output_size.'
    assert not scale or len(scale) in (1, 2, 3), 'Only 1D/2D/3D'
    assert not output_size or len(output_size) in (1, 2, 3), 'Only 1D/2D/3D'

    if channel_last:
        n_sdim = len(scale) if scale else len(output_size)
        t = refs.ChannelLastToFirstTranspose(x.ndim, n_sdim)
        x = t(x)

    if not output_size:
        output_size = np.floor(np.array(scale) * x.shape[-len(scale):])
        output_size = tuple(map(int, output_size))

    if mode == "nearest":
        if len(output_size) == 1:
            out = ref_nearest_interpolate_1d(
                x, output_size, align_corners, half_pixel, half_pixel_for_nn)
            out = t.inv(out) if channel_last else out
        if len(output_size) == 2:
            out = ref_nearest_interpolate_2d(
                x, output_size, align_corners, half_pixel, half_pixel_for_nn)
            out = t.inv(out) if channel_last else out
        if len(output_size) == 3:
            out = ref_nearest_interpolate_3d(
                x, output_size, align_corners, half_pixel, half_pixel_for_nn)
            out = t.inv(out) if channel_last else out
    elif mode == "linear":
        if len(output_size) == 1:
            out = ref_linear_interpolate_1d(
                x, output_size, align_corners, half_pixel)

        if len(output_size) == 2:
            out = ref_linear_interpolate_2d(
                x, output_size, align_corners, half_pixel)

        if len(output_size) == 3:
            out = ref_linear_interpolate_3d(
                x, output_size, align_corners, half_pixel)
        out = t.inv(out) if channel_last else out
    return out


def ref_linear_interpolate_1d(x, output_size, align_corners, half_pixel):
    oshape = output_size          # output-width
    ishape = x.shape[-1:]         # input-width
    xx = x.reshape(-1, *ishape)
    ib = np.arange(xx.shape[0])  # batch index

    scale = (compute_scale(ishape[0], oshape[0], align_corners),)  # x

    # Real input indices as floats
    index = (get_source_index(scale[0], np.arange(oshape[0]), half_pixel),)

    # Nearest input indices per axis
    index_1 = (index[0].astype(np.int32),)
    index_2 = (np.minimum(index_1[0] + 1, ishape[0] - 1),)

    # Unit distance to left and right
    dist_1 = ((index[0] - index_1[0]).reshape(1, -1),)  # x
    dist_2 = (1.0 - dist_1[0],)

    val0 = dist_2[0] * xx[np.ix_(ib, index_1[0])]
    val1 = dist_1[0] * xx[np.ix_(ib, index_2[0])]

    yy = val0 + val1

    return yy.reshape(x.shape[:-len(oshape)] + oshape)


def ref_linear_interpolate_2d(x, output_size, align_corners, half_pixel):
    oshape = output_size          # output-height, output-width
    ishape = x.shape[-2:]         # input-height, input-width
    xx = x.reshape(-1, *ishape)
    ib = np.arange(xx.shape[0])  # batch index

    scale = (compute_scale(ishape[0], oshape[0], align_corners),  # y
             compute_scale(ishape[1], oshape[1], align_corners))  # x

    # Real input indices as floats
    index = (get_source_index(scale[0], np.arange(oshape[0]), half_pixel),
             get_source_index(scale[1], np.arange(oshape[1]), half_pixel))

    # Nearest input indices per axis
    index_1 = (index[0].astype(np.int32),
               index[1].astype(np.int32))
    index_2 = (np.minimum(index_1[0] + 1, ishape[0] - 1),
               np.minimum(index_1[1] + 1, ishape[1] - 1))

    # Unit distance to left and right
    dist_1 = ((index[0] - index_1[0]).reshape(1, -1, 1),  # y
              (index[1] - index_1[1]).reshape(1, 1, -1))  # x
    dist_2 = (1.0 - dist_1[0],
              1.0 - dist_1[1])

    val0 = dist_2[1] * xx[np.ix_(ib, index_1[0], index_1[1])]
    val1 = dist_1[1] * xx[np.ix_(ib, index_1[0], index_2[1])]
    val2 = dist_2[1] * xx[np.ix_(ib, index_2[0], index_1[1])]
    val3 = dist_1[1] * xx[np.ix_(ib, index_2[0], index_2[1])]

    yy = (dist_2[0] * (val0 + val1)) + (dist_1[0] * (val2 + val3))

    return yy.reshape(x.shape[:-len(oshape)] + oshape)


def ref_linear_interpolate_3d(x, output_size, align_corners, half_pixel):
    oshape = output_size          # output-depth, output-height, output-width
    ishape = x.shape[-3:]         # input-depth, input-height, input-width
    xx = x.reshape(-1, *ishape)
    ib = np.arange(xx.shape[0])  # batch index

    scale = (compute_scale(ishape[0], oshape[0], align_corners),  # z
             compute_scale(ishape[1], oshape[1], align_corners),  # y
             compute_scale(ishape[2], oshape[2], align_corners))  # x

    # Real input indices as floats
    index = (get_source_index(scale[0], np.arange(oshape[0]), half_pixel),
             get_source_index(scale[1], np.arange(oshape[1]), half_pixel),
             get_source_index(scale[2], np.arange(oshape[2]), half_pixel))

    # Nearest input indices per axis
    index_1 = (index[0].astype(np.int32),
               index[1].astype(np.int32),
               index[2].astype(np.int32))
    index_2 = (np.minimum(index_1[0] + 1, ishape[0] - 1),
               np.minimum(index_1[1] + 1, ishape[1] - 1),
               np.minimum(index_1[2] + 1, ishape[2] - 1))

    # Unit distance to left and right
    dist_1 = ((index[0] - index_1[0]).reshape(1, -1, 1, 1),  # z
              (index[1] - index_1[1]).reshape(1, 1, -1, 1),  # y
              (index[2] - index_1[2]).reshape(1, 1, 1, -1))  # x
    dist_2 = (1.0 - dist_1[0],
              1.0 - dist_1[1],
              1.0 - dist_1[2])

    val0 = dist_2[2] * xx[np.ix_(ib, index_1[0], index_1[1], index_1[2])]
    val1 = dist_1[2] * xx[np.ix_(ib, index_1[0], index_1[1], index_2[2])]
    val2 = dist_2[2] * xx[np.ix_(ib, index_1[0], index_2[1], index_1[2])]
    val3 = dist_1[2] * xx[np.ix_(ib, index_1[0], index_2[1], index_2[2])]
    val4 = dist_2[2] * xx[np.ix_(ib, index_2[0], index_1[1], index_1[2])]
    val5 = dist_1[2] * xx[np.ix_(ib, index_2[0], index_1[1], index_2[2])]
    val6 = dist_2[2] * xx[np.ix_(ib, index_2[0], index_2[1], index_1[2])]
    val7 = dist_1[2] * xx[np.ix_(ib, index_2[0], index_2[1], index_2[2])]

    val8 = (dist_2[1] * (val0 + val1)) + (dist_1[1] * (val2 + val3))
    val9 = (dist_2[1] * (val4 + val5)) + (dist_1[1] * (val6 + val7))

    yy = dist_2[0] * val8 + dist_1[0] * val9

    return yy.reshape(x.shape[:-len(oshape)] + oshape)


def ref_nearest_interpolate_1d(x, output_size, align_corners, half_pixel, half_pixel_for_nn):
    oshape = output_size          # output-width
    ishape = x.shape[-1:]         # input-width
    xx = x.reshape(-1, *ishape)
    ib = np.arange(xx.shape[0])  # batch index

    scale = (compute_scale_for_nn(
        ishape[0], oshape[0], align_corners, half_pixel_for_nn),)  # x

    # Real input indices as floats
    index = (get_source_index_for_nn(scale[0], np.arange(
        oshape[0]), half_pixel, half_pixel_for_nn),)

    # Nearest input indices per axis
    index_1 = (np.minimum(index[0].astype(np.int32), ishape[0] - 1), )

    # Nearest input
    yy = xx[np.ix_(ib, index_1[0])]

    return yy.reshape(x.shape[:-len(oshape)] + oshape)


def ref_nearest_interpolate_2d(x, output_size, align_corners, half_pixel, half_pixel_for_nn):
    oshape = output_size          # output-height, output-width
    ishape = x.shape[-2:]         # input-height, input-width
    xx = x.reshape(-1, *ishape)
    ib = np.arange(xx.shape[0])  # batch index

    scale = (compute_scale_for_nn(ishape[0], oshape[0], align_corners, half_pixel_for_nn),  # y
             compute_scale_for_nn(ishape[1], oshape[1], align_corners, half_pixel_for_nn))  # x

    # Real input indices as floats
    index = (get_source_index_for_nn(scale[0], np.arange(oshape[0]), half_pixel, half_pixel_for_nn),
             get_source_index_for_nn(scale[1], np.arange(oshape[1]), half_pixel, half_pixel_for_nn))

    # Nearest input indices per axis
    index_1 = (index[0].astype(np.int32),
               index[1].astype(np.int32))
    index_2 = (np.minimum(index_1[0], ishape[0] - 1),
               np.minimum(index_1[1], ishape[1] - 1))

    # Nearest input
    yy = xx[np.ix_(ib, index_2[0], index_2[1])]

    return yy.reshape(x.shape[:-len(oshape)] + oshape)


def ref_nearest_interpolate_3d(x, output_size, align_corners, half_pixel, half_pixel_for_nn):
    oshape = output_size          # output-depth, output-height, output-width
    ishape = x.shape[-3:]         # input-depth, input-height, input-width
    xx = x.reshape(-1, *ishape)
    ib = np.arange(xx.shape[0])  # batch index

    scale = (compute_scale_for_nn(ishape[0], oshape[0], align_corners, half_pixel_for_nn),  # z
             compute_scale_for_nn(
                 ishape[1], oshape[1], align_corners, half_pixel_for_nn),  # y
             compute_scale_for_nn(ishape[2], oshape[2], align_corners, half_pixel_for_nn))  # x

    # Real input indices as floats
    index = (get_source_index_for_nn(scale[0], np.arange(oshape[0]), half_pixel, half_pixel_for_nn),
             get_source_index_for_nn(scale[1], np.arange(
                 oshape[1]), half_pixel, half_pixel_for_nn),
             get_source_index_for_nn(scale[2], np.arange(oshape[2]), half_pixel, half_pixel_for_nn))

    # Nearest input indices per axis
    index_1 = (index[0].astype(np.int32),
               index[1].astype(np.int32),
               index[2].astype(np.int32))
    index_2 = (np.minimum(index_1[0], ishape[0] - 1),
               np.minimum(index_1[1], ishape[1] - 1),
               np.minimum(index_1[2], ishape[2] - 1))

    # Nearest input
    yy = xx[np.ix_(ib, index_2[0], index_2[1], index_2[2])]

    return yy.reshape(x.shape[:-len(oshape)] + oshape)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("inshape, outsize, scale, sdim_only", [
    # 1-dimensional
    ((3,), (6,), None, True),
    ((3,), (8,), None, True),
    ((3,), (2,), None, True),
    ((3,), (1,), None, True),
    ((3,), None, (2.5,), True),
    ((3,), None, (0.5,), True),
    ((2, 3, 4), (10,), None, False),
    ((2, 3, 5), None, (1.3,), False),
    # 2-dimensional
    ((3, 3), (8, 6), None, True),
    ((3, 3), (2, 1), None, True),
    ((3, 3), None, (2.5, 1.0), True),
    ((3, 3), None, (0.5, 0.5), True),
    ((2, 3, 4, 4), (8, 6), None, False),
    ((2, 3, 4, 4), (2, 1), None, False),
    ((2, 3, 4, 4), None, (2.5, 1.0), False),
    ((2, 3, 4, 4), None, (0.5, 0.5), False),
    ((2, 3, 4, 4), None, (0.5, 0.5), False),
    # 3-dimensional
    ((3, 3, 3), (6, 8, 6), None, True),
    ((3, 3, 3), (1, 2, 1), None, True),
    ((3, 3, 3), None, (1.5, 2.5, 1.0), True),
    ((3, 3, 3), None, (1.2, 0.5, 0.5), True),
    ((2, 3, 3, 4, 4), (6, 8, 6), None, False),
    ((2, 3, 3, 4, 4), (1, 2, 1), None, False),
    ((2, 3, 3, 4, 4), None, (1.5, 2.5, 1.0), False),
    ((2, 3, 3, 4, 4), None, (1.2, 0.5, 0.5), False),
])
@pytest.mark.parametrize('align_corners, half_pixel', [(True, False), (False, True), (False, False)])
@pytest.mark.parametrize('channel_last', [False, True])
@pytest.mark.parametrize("seed", [313])
def test_interpolate_linear_forward_backward(seed, inshape, outsize, scale, sdim_only,
                                             align_corners, half_pixel, channel_last,
                                             ctx, func_name):
    if channel_last and func_name == "Interpolate":
        pytest.skip("Interpolate with channel_last is only supported in CUDA.")
    if sdim_only and channel_last:
        pytest.skip(
            "Interpolate for spatial dimension only data is only supported for channel_first option.")

    from nbla_test_utils import function_tester
    rng = np.random.RandomState(seed)
    inputs = [rng.randn(*inshape).astype(np.float32)]
    func_args = [scale, outsize, 'linear',
                 align_corners, half_pixel, False, channel_last]
    function_tester(rng, F.interpolate, ref_interpolate, inputs,
                    func_name=func_name, func_args=func_args,
                    atol_f=1e-6, atol_b=1e-2, dstep=2e-3, ctx=ctx, disable_half_test=True)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("inshape, outsize, scale, sdim_only", [
    # 1-dimensional
    ((3,), (6,), None, True),
    ((3,), (8,), None, True),
    ((3,), (2,), None, True),
    ((3,), (1,), None, True),
    ((3,), None, (2.5,), True),
    ((3,), None, (0.5,), True),
    ((2, 3, 4), (10,), None, False),
    ((2, 3, 5), None, (1.3,), False),
    # 2-dimensional
    ((3, 3), (8, 6), None, True),
    ((3, 3), (2, 1), None, True),
    ((3, 3), None, (2.5, 1.0), True),
    ((3, 3), None, (0.5, 0.5), True),
    ((2, 3, 4, 4), (8, 6), None, False),
    ((2, 3, 4, 4), (2, 1), None, False),
    ((2, 3, 4, 4), None, (2.5, 1.0), False),
    ((2, 3, 4, 4), None, (0.5, 0.5), False),
    ((2, 3, 4, 4), None, (0.5, 0.5), False),
    # 3-dimensional
    ((3, 3, 3), (6, 8, 6), None, True),
    ((3, 3, 3), (1, 2, 1), None, True),
    ((3, 3, 3), None, (1.5, 2.5, 1.0), True),
    ((3, 3, 3), None, (1.2, 0.5, 0.5), True),
    ((1, 2, 3, 4, 4), (6, 8, 6), None, False),
    ((1, 2, 3, 4, 4), (1, 2, 1), None, False),
    ((1, 2, 3, 4, 4), None, (1.5, 2.5, 1.0), False),
    ((1, 2, 3, 4, 4), None, (1.2, 0.5, 0.5), False),
])
@pytest.mark.parametrize('align_corners, half_pixel, half_pixel_for_nn',
                         [(True, False, True),
                          (True, False, False),
                          (False, True, False),
                          (False, False, False)])
@pytest.mark.parametrize('channel_last', [False, True])
@pytest.mark.parametrize("seed", [313])
def test_interpolate_nearest_forward_backward(seed, inshape, outsize, scale, sdim_only,
                                              align_corners, half_pixel, half_pixel_for_nn,
                                              channel_last,
                                              ctx, func_name):
    if channel_last and func_name == "Interpolate":
        pytest.skip("Interpolate with channel_last is only supported in CUDA.")
    if sdim_only and channel_last:
        pytest.skip(
            "Interpolate for spatial dimension only data is only supported for channel_first option.")

    from nbla_test_utils import function_tester
    rng = np.random.RandomState(seed)
    inputs = [rng.randn(*inshape).astype(np.float32)]
    func_args = [scale, outsize, 'nearest', align_corners,
                 half_pixel, half_pixel_for_nn, channel_last]
    function_tester(rng, F.interpolate, ref_interpolate, inputs,
                    func_name=func_name, func_args=func_args,
                    atol_f=1e-6, atol_b=1e-2, dstep=2e-3, ctx=ctx)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("inshape, outsize, scale, sdim_only", [
    # 1-dimensional
    ((3,), (6,), None, True),
    ((3,), (8,), None, True),
    ((3,), (2,), None, True),
    ((3,), (1,), None, True),
    ((3,), None, (2.5,), True),
    ((3,), None, (0.5,), True),
    ((2, 3, 4), (10,), None, False),
    ((2, 3, 5), None, (1.3,), False),
    # 2-dimensional
    ((3, 3), (8, 6), None, True),
    ((3, 3), (2, 1), None, True),
    ((3, 3), None, (2.5, 1.0), True),
    ((3, 3), None, (0.5, 0.5), True),
    ((2, 3, 4, 4), (8, 6), None, False),
    ((2, 3, 4, 4), (2, 1), None, False),
    ((2, 3, 4, 4), None, (2.5, 1.0), False),
    ((2, 3, 4, 4), None, (0.5, 0.5), False),
    ((2, 3, 4, 4), None, (0.5, 0.5), False),
    # 3-dimensional
    ((3, 3, 3), (6, 8, 6), None, True),
    ((3, 3, 3), (1, 2, 1), None, True),
    ((3, 3, 3), None, (1.5, 2.5, 1.0), True),
    ((3, 3, 3), None, (1.2, 0.5, 0.5), True),
    ((2, 3, 3, 4, 4), (6, 8, 6), None, False),
    ((2, 3, 3, 4, 4), (1, 2, 1), None, False),
    ((2, 3, 3, 4, 4), None, (1.5, 2.5, 1.0), False),
    ((2, 3, 3, 4, 4), None, (1.2, 0.5, 0.5), False),
])
@pytest.mark.parametrize('align_corners, half_pixel', [(True, False), (False, True), (False, False)])
@pytest.mark.parametrize('channel_last', [False, True])
@pytest.mark.parametrize("seed", [313])
def test_interpolate_linear_double_backward(seed, inshape, outsize, scale, sdim_only,
                                            align_corners, half_pixel, channel_last, ctx, func_name):
    if channel_last and func_name == "Interpolate":
        pytest.skip("Interpolate with channel_last is only supported in CUDA.")
    if sdim_only and channel_last:
        pytest.skip(
            "Interpolate for spatial dimension only data is only supported for channel_first option.")

    from nbla_test_utils import backward_function_tester, grad_function_forward_function_output
    from nnabla.backward_function.interpolate import InterpolateDataGrad

    rng = np.random.RandomState(seed)
    inputs = [rng.randn(*inshape).astype(np.float32)]
    func_args = [scale, outsize, 'linear',
                 align_corners, half_pixel, False, channel_last]
    # 2nd-order
    backward_function_tester(rng, F.interpolate, inputs,
                             func_args=func_args,
                             ctx=ctx)
    # 3rd-order
    # F.interpolate takes scale and output_size while InterpolateDataGrad takes only output_size
    # for passing kwargs in the nn.grad, same as F.Interpolate
    import nnabla as nn
    import math
    vinputs = [nn.Variable(inp.shape)
               if inp is not None else None for inp in inputs]
    y = F.interpolate(*(vinputs + func_args))
    x = inputs[0]
    if scale:
        input_size = x.shape[-len(scale)-1:-
                             1] if channel_last else x.shape[-len(scale):]
        output_size = [int(math.floor(s * d))
                       for d, s in zip(input_size, scale)]
    else:
        output_size = outsize
    df = InterpolateDataGrad(ctx, *([output_size] + func_args[2:]))
    df.xshape = x.shape
    ginputs = [rng.randn(*y.shape)]
    backward_function_tester(rng, df,
                             ginputs, func_args=[],
                             ctx=ctx, atol_f=1e-6, atol_accum=8e-2, non_accum_check=True)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("inshape, outsize, scale, sdim_only", [
    # 1-dimensional
    ((3,), (6,), None, True),
    ((3,), (8,), None, True),
    ((3,), (2,), None, True),
    ((3,), (1,), None, True),
    ((3,), None, (2.5,), True),
    ((3,), None, (0.5,), True),
    ((2, 3, 4), (10,), None, False),
    ((2, 3, 5), None, (1.3,), False),
    # 2-dimensional
    ((3, 3), (8, 6), None, True),
    ((3, 3), (2, 1), None, True),
    ((3, 3), None, (2.5, 1.0), True),
    ((3, 3), None, (0.5, 0.5), True),
    ((2, 3, 4, 4), (8, 6), None, False),
    ((2, 3, 4, 4), (2, 1), None, False),
    ((2, 3, 4, 4), None, (2.5, 1.0), False),
    ((2, 3, 4, 4), None, (0.5, 0.5), False),
    ((2, 3, 4, 4), None, (0.5, 0.5), False),
    # 3-dimensional
    ((3, 3, 3), (6, 8, 6), None, True),
    ((3, 3, 3), (1, 2, 1), None, True),
    ((3, 3, 3), None, (1.5, 2.5, 1.0), True),
    ((3, 3, 3), None, (1.2, 0.5, 0.5), True),
    ((1, 2, 3, 4, 4), (6, 8, 6), None, False),
    ((1, 2, 3, 4, 4), (1, 2, 1), None, False),
    ((1, 2, 3, 4, 4), None, (1.5, 2.5, 1.0), False),
    ((1, 2, 3, 4, 4), None, (1.2, 0.5, 0.5), False),
])
@pytest.mark.parametrize('align_corners, half_pixel, half_pixel_for_nn',
                         [(True, False, True),
                          (True, False, False),
                          (False, True, False),
                          (False, False, False)])
@pytest.mark.parametrize('channel_last', [False, True])
@pytest.mark.parametrize("seed", [313])
def test_interpolate_nearest_double_backward(seed, inshape, outsize, scale, sdim_only,
                                             align_corners, half_pixel, half_pixel_for_nn,
                                             channel_last,
                                             ctx, func_name):
    if channel_last and func_name == "Interpolate":
        pytest.skip("Interpolate with channel_last is only supported in CUDA.")
    if sdim_only and channel_last:
        pytest.skip(
            "Interpolate for spatial dimension only data is only supported for channel_first option.")
    from nbla_test_utils import backward_function_tester, grad_function_forward_function_output
    from nnabla.backward_function.interpolate import InterpolateDataGrad

    rng = np.random.RandomState(seed)
    inputs = [rng.randn(*inshape).astype(np.float32)]
    func_args = [scale, outsize, 'nearest', align_corners,
                 half_pixel, half_pixel_for_nn, channel_last]
    # 2nd-order
    backward_function_tester(rng, F.interpolate, inputs,
                             func_args=func_args,
                             atol_f=1e-6, atol_accum=1e-2, dstep=1e-3, ctx=ctx)
    # 3rd-order
    # F.interpolate takes scale and output_size while InterpolateDataGrad takes only output_size
    # for passing kwargs in the nn.grad, same as F.Interpolate
    import nnabla as nn
    import math
    vinputs = [nn.Variable(inp.shape)
               if inp is not None else None for inp in inputs]
    y = F.interpolate(*(vinputs + func_args))
    x = inputs[0]
    if scale:
        input_size = x.shape[-len(scale)-1:-
                             1] if channel_last else x.shape[-len(scale):]
        output_size = [int(math.floor(s * d))
                       for d, s in zip(input_size, scale)]
    else:
        output_size = outsize
    df = InterpolateDataGrad(ctx, *([output_size] + func_args[2:]))
    df.xshape = x.shape
    ginputs = [rng.randn(*y.shape)]
    backward_function_tester(rng, df,
                             ginputs, func_args=[],
                             ctx=ctx, atol_f=1e-6, atol_accum=5e-2, non_accum_check=True)
