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
from nbla_test_utils import list_context

ctxs = list_context('Interpolate')


def compute_scale(isize, osize, align_corners):
    if osize > 1:
        return (isize - 1) / (osize - 1) if align_corners else isize / osize
    else:
        return 0.0


def get_source_index(scale, dst_index, align_corners):
    return (scale * dst_index if align_corners else
            np.maximum(0, scale * (dst_index + 0.5) - 0.5))


def ref_interpolate(x, scale, output_size, mode, align_corners=None):
    assert scale or output_size, 'Need either scale or output_size.'
    assert not scale or len(scale) in (1, 2, 3), 'Only 1D/2D/3D'
    assert not output_size or len(output_size) in (1, 2, 3), 'Only 1D/2D/3D'

    if not output_size:
        output_size = np.floor(np.array(scale) * x.shape[-len(scale):])
        output_size = tuple(map(int, output_size))

    if mode == "nearest":
        osize = output_size
        isize = x.shape[-len(osize):]
        scale = [i / o for i, o in zip(isize, osize)]
        index = [s * (np.arange(o) + 0.5) for s, o in zip(scale, osize)]
        index = [idx.astype(np.int32) for idx in index]
        index = [np.minimum(idx, i - 1) for idx, i in zip(index, isize)]
        xx = x.reshape(-1, *isize)
        ib = np.arange(xx.shape[0])
        yy = xx[np.ix_(ib, *index)]
        return yy.reshape(x.shape[:-len(osize)] + osize)

    if len(output_size) == 1:
        return ref_linear_interpolate_1d(x, output_size, mode, align_corners)

    if len(output_size) == 2:
        return ref_linear_interpolate_2d(x, output_size, mode, align_corners)

    if len(output_size) == 3:
        return ref_linear_interpolate_3d(x, output_size, mode, align_corners)


def ref_linear_interpolate_1d(x, output_size, mode, align_corners):
    assert mode == 'linear'

    oshape = output_size          # output-width
    ishape = x.shape[-1:]         # input-width
    xx = x.reshape(-1, *ishape)
    ib = np.arange(xx.shape[0])  # batch index

    scale = (compute_scale(ishape[0], oshape[0], align_corners),)  # x

    # Real input indices as floats
    index = (get_source_index(scale[0], np.arange(oshape[0]), align_corners),)

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


def ref_linear_interpolate_2d(x, output_size, mode, align_corners):
    assert mode == 'linear'

    oshape = output_size          # output-height, output-width
    ishape = x.shape[-2:]         # input-height, input-width
    xx = x.reshape(-1, *ishape)
    ib = np.arange(xx.shape[0])  # batch index

    scale = (compute_scale(ishape[0], oshape[0], align_corners),  # y
             compute_scale(ishape[1], oshape[1], align_corners))  # x

    # Real input indices as floats
    index = (get_source_index(scale[0], np.arange(oshape[0]), align_corners),
             get_source_index(scale[1], np.arange(oshape[1]), align_corners))

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


def ref_linear_interpolate_3d(x, output_size, mode, align_corners):
    assert mode == 'linear'

    oshape = output_size          # output-depth, output-height, output-width
    ishape = x.shape[-3:]         # input-depth, input-height, input-width
    xx = x.reshape(-1, *ishape)
    ib = np.arange(xx.shape[0])  # batch index

    scale = (compute_scale(ishape[0], oshape[0], align_corners),  # z
             compute_scale(ishape[1], oshape[1], align_corners),  # y
             compute_scale(ishape[2], oshape[2], align_corners))  # x

    # Real input indices as floats
    index = (get_source_index(scale[0], np.arange(oshape[0]), align_corners),
             get_source_index(scale[1], np.arange(oshape[1]), align_corners),
             get_source_index(scale[2], np.arange(oshape[2]), align_corners))

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


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("inshape, outsize, scale", [
    # 1-dimensional
    ((3,), (6,), None),
    ((3,), (8,), None),
    ((3,), (2,), None),
    ((3,), (1,), None),
    ((3,), None, (2.5,)),
    ((3,), None, (0.5,)),
    ((2, 3, 4), (10,), None),
    ((2, 3, 5), None, (1.3,)),
    # 2-dimensional
    ((3, 3), (8, 6), None),
    ((3, 3), (2, 1), None),
    ((3, 3), None, (2.5, 1.0)),
    ((3, 3), None, (0.5, 0.5)),
    ((2, 3, 4, 4), (8, 6), None),
    ((2, 3, 4, 4), (2, 1), None),
    ((2, 3, 4, 4), None, (2.5, 1.0)),
    ((2, 3, 4, 4), None, (0.5, 0.5)),
    # 3-dimensional
    ((3, 3, 3), (6, 8, 6), None),
    ((3, 3, 3), (1, 2, 1), None),
    ((3, 3, 3), None, (1.5, 2.5, 1.0)),
    ((3, 3, 3), None, (1.2, 0.5, 0.5)),
    ((2, 2, 3, 4, 4), (6, 8, 6), None),
    ((2, 2, 3, 4, 4), (1, 2, 1), None),
    ((2, 2, 3, 4, 4), None, (1.5, 2.5, 1.0)),
    ((2, 2, 3, 4, 4), None, (1.2, 0.5, 0.5)),
])
@pytest.mark.parametrize('align_corners', [False, True])
@pytest.mark.parametrize("seed", [313])
def test_interpolate_linear_forward_backward(seed, inshape, outsize, scale,
                                             align_corners, ctx, func_name):
    from nbla_test_utils import function_tester
    rng = np.random.RandomState(seed)
    inputs = [rng.randn(*inshape).astype(np.float32)]
    func_args = [scale, outsize, 'linear', align_corners]
    function_tester(rng, F.interpolate, ref_interpolate, inputs,
                    func_name=func_name, func_args=func_args,
                    atol_f=1e-6, atol_b=1e-2, ctx=ctx)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("inshape, outsize, scale", [
    # 1-dimensional
    ((3,), (6,), None),
    ((3,), (8,), None),
    ((3,), (2,), None),
    ((3,), (1,), None),
    ((3,), None, (2.5,)),
    ((3,), None, (0.5,)),
    ((2, 3, 4), (10,), None),
    ((2, 3, 5), None, (1.3,)),
    # 2-dimensional
    ((3, 3), (8, 6), None),
    ((3, 3), (2, 1), None),
    ((3, 3), None, (2.5, 1.0)),
    ((3, 3), None, (0.5, 0.5)),
    ((2, 3, 4, 5), (8, 6), None),
    ((2, 3, 4, 5), (2, 1), None),
    ((2, 3, 4, 5), None, (2.5, 1.0)),
    ((2, 3, 4, 5), None, (0.5, 0.5)),
    # 3-dimensional
    ((3, 3, 3), (6, 8, 6), None),
    ((3, 3, 3), (1, 2, 1), None),
    ((3, 3, 3), None, (1.5, 2.5, 1.0)),
    ((3, 3, 3), None, (1.2, 0.5, 0.5)),
    ((1, 2, 3, 4, 5), (6, 8, 6), None),
    ((1, 2, 3, 4, 5), (1, 2, 3), None),
    ((1, 2, 3, 4, 5), None, (1.5, 2.5, 1.0)),
    ((1, 2, 3, 4, 5), None, (1.2, 0.5, 0.5)),
])
@pytest.mark.parametrize("seed", [313])
def test_interpolate_nearest_forward_backward(seed, inshape, outsize, scale,
                                              ctx, func_name):
    from nbla_test_utils import function_tester
    rng = np.random.RandomState(seed)
    inputs = [rng.randn(*inshape).astype(np.float32)]
    func_args = [scale, outsize, 'nearest']
    function_tester(rng, F.interpolate, ref_interpolate, inputs,
                    func_name=func_name, func_args=func_args,
                    atol_f=1e-6, atol_b=1e-2, ctx=ctx)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("inshape, outsize, scale", [
    # 2-dimensional
    ((3, 3), (8, 6), None),
    ((3, 3), (2, 1), None),
    ((3, 3), None, (2.5, 1.0)),
    ((3, 3), None, (0.5, 0.5)),
    ((2, 3, 4, 4), (8, 6), None),
    ((2, 3, 4, 4), (2, 1), None),
    ((2, 3, 4, 4), None, (2.5, 1.0)),
    ((2, 3, 4, 4), None, (0.5, 0.5)),
    # 3-dimensional
    ((3, 3, 3), (6, 8, 6), None),
    ((3, 3, 3), (1, 2, 1), None),
    ((3, 3, 3), None, (1.5, 2.5, 1.0)),
    ((3, 3, 3), None, (1.2, 0.5, 0.5)),
    ((2, 2, 3, 4, 4), (6, 8, 6), None),
    ((2, 2, 3, 4, 4), (1, 2, 1), None),
    ((2, 2, 3, 4, 4), None, (1.5, 2.5, 1.0)),
    ((2, 2, 3, 4, 4), None, (1.2, 0.5, 0.5)),
])
@pytest.mark.parametrize('align_corners', [False, True])
@pytest.mark.parametrize("seed", [313])
def test_interpolate_linear_double_backward(seed, inshape, outsize, scale,
                                            align_corners, ctx, func_name):
    # TODO: some test fail.
    from nbla_test_utils import backward_function_tester
    rng = np.random.RandomState(seed)
    inputs = [rng.randn(*inshape).astype(np.float32)]
    func_args = [scale, outsize, 'linear', align_corners]
    backward_function_tester(rng, F.interpolate, None, inputs,
                             func_name=func_name, func_args=func_args,
                             atol_f=1e-6, atol_b=1e-2, atol_accum=1e-2, dstep=1e-3, ctx=ctx)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("inshape, outsize, scale", [
    # 2-dimensional
    ((3, 3), (8, 6), None),
    ((3, 3), (2, 1), None),
    ((3, 3), None, (2.5, 1.0)),
    ((3, 3), None, (0.5, 0.5)),
    ((2, 3, 4, 5), (8, 6), None),
    ((2, 3, 4, 5), (2, 1), None),
    ((2, 3, 4, 5), None, (2.5, 1.0)),
    ((2, 3, 4, 5), None, (0.5, 0.5)),
    # 3-dimensional
    ((3, 3, 3), (6, 8, 6), None),
    ((3, 3, 3), (1, 2, 1), None),
    ((3, 3, 3), None, (1.5, 2.5, 1.0)),
    ((3, 3, 3), None, (1.2, 0.5, 0.5)),
    ((1, 2, 3, 4, 5), (6, 8, 6), None),
    ((1, 2, 3, 4, 5), (1, 2, 3), None),
    ((1, 2, 3, 4, 5), None, (1.5, 2.5, 1.0)),
    ((1, 2, 3, 4, 5), None, (1.2, 0.5, 0.5)),
])
@pytest.mark.parametrize("seed", [313])
def test_interpolate_nearest_double_backward(seed, inshape, outsize, scale,
                                             ctx, func_name):
    from nbla_test_utils import backward_function_tester
    rng = np.random.RandomState(seed)
    inputs = [rng.randn(*inshape).astype(np.float32)]
    func_args = [scale, outsize, 'nearest']
    backward_function_tester(rng, F.interpolate, ref_interpolate, inputs,
                             func_name=func_name, func_args=func_args,
                             atol_f=1e-6, atol_b=1e-2, atol_accum=1e-2, ctx=ctx)
