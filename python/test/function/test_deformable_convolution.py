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
import copy
import numpy as np
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import refs
from nbla_test_utils import list_context
from nbla_test_utils import function_tester

ctxs = list_context('DeformableConvolution')


def ref_deformable_convolution_2d(x, w, offset, mask, b, base_axis, pad, stride,
                                  dilation, group, deformable_group, channel_last):
    if channel_last:
        assert False, "channel_last=True is not supported in ref_deformable_convolution_2d."

    assert x.shape[0:base_axis] == offset.shape[0:base_axis], "Batch sizes do not match."

    # Compute deformable convolution for each batch.
    y = []
    # Flatten the batch dimensions to pass it the reference function.
    ext_x = x.reshape((-1,) + x.shape[base_axis:])
    ext_offset = offset.reshape((-1,) + offset.shape[base_axis:])
    if mask is None:
        mask_shape = x.shape[0:base_axis] + \
            (deformable_group * w.shape[2] *
             w.shape[3],) + x.shape[base_axis + 1:]
        mask = np.ones(mask_shape).astype(np.float32)
    ext_mask = mask.reshape((-1,) + mask.shape[base_axis:])
    for xx, oo, mm in zip(ext_x, ext_offset, ext_mask):
        y += [refs.deformable_convolution_2d(xx, w, oo, mm, b, pad, stride, dilation,
                                             group, deformable_group, channel_last)[np.newaxis]]
    y = np.vstack(y)
    return y.reshape(x.shape[:base_axis] + y.shape[1:])


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("inshape, kernel, out_channels, pad, stride, dilation, group, deformable_group, with_bias", [
    # ((2, 4, 10, 10), (3, 2), 4, (0, 0), (1, 1), (1, 1), 1, 2, False), # To reduce test time
    ((2, 4, 6, 6), (3, 2), 4, (0, 0), (1, 1), (1, 1), 2, 2, True),
    ((2, 2, 5, 7), (3, 3), 2, (1, 1), (1, 2), (2, 1), 1, 1, True),
    ((2, 2, 5, 7), (3, 3), 2, (1, 1), (1, 2), (2, 1), 1, 2, False),
    ((2, 2, 5, 7), (3, 3), 2, (1, 1), (1, 2), (2, 1), 2, 1, False),
 ])
@pytest.mark.parametrize("with_mask", [True, False])
@pytest.mark.parametrize("channel_last", [True, False])
def test_forward_backward_2d(inshape, kernel, out_channels, pad, stride, dilation, group,
                             deformable_group, with_mask, channel_last, with_bias, seed, ctx, func_name):
    if channel_last:
        pytest.skip(
            'channel_last=True is not supported in any backends so far.')

    import platform
    if platform.machine().startswith("arm"):
        pytest.skip('Skip the arm platform temporarily.')

    rng = np.random.RandomState(seed)

    # Create arguments
    base_axis = len(inshape) - len(kernel) - 1
    func_args = [base_axis, pad, stride, dilation, group,
                 deformable_group, channel_last]

    # Compute shapes
    in_channels = inshape[base_axis]
    kshape = (out_channels, in_channels // group) + kernel
    offset_channels = 2 * deformable_group * kernel[0] * kernel[1]
    offset_shape = inshape[0:base_axis] + \
        (offset_channels,) + inshape[base_axis + 1:]
    mask_shape = inshape[0:base_axis] + \
        (deformable_group * kernel[0] * kernel[1],) + inshape[base_axis + 1:]
    if channel_last:
        t = refs.ChannelLastToFirstTranspose(len(inshape), len(kernel))
        inshape = tuple(inshape[i] for i in t.inv_axes)
        t = refs.ChannelLastToFirstTranspose(len(offset_shape), len(kernel))
        offset_shape = tuple(offset_shape[i] for i in t.inv_axes)
        t = refs.ChannelLastToFirstTranspose(len(kshape), len(kernel))
        kshape = tuple(kshape[i] for i in t.inv_axes)

    # Create inputs
    x = rng.randn(*inshape).astype(np.float32)
    w = rng.randn(*kshape).astype(np.float32)
    b = rng.randn(out_channels).astype(np.float32) if with_bias else None

    # Because numerical gradient cannot be calculated correctly
    # near the input boundary, offsets are generated to avoid this case.
    # 1. Generate offsets in [-1.9, 1.9].
    offsets = (3.8 * rng.rand(*offset_shape).astype(np.float32)) - 1.9
    # 2. Adhoc remove the values dstep-neighborhood of {-1, 0, 1}; selecting bad
    #    values as 0.1-neighborhood (large enough dstep-neighborhood) and shifting
    #    them +0.5 (must larger than 2 * dstep).
    offsets += np.logical_or(np.abs(offsets - np.floor(offsets)) < 0.1,
                             np.abs(offsets - np.ceil(offsets)) < 0.1).astype(np.int)*0.5

    mask = rng.rand(*mask_shape).astype(np.float32) if with_mask else None

    inputs = [x, w, offsets, mask, b]

    # Test
    atol_half = 1.0 if in_channels > 64 else 1.5e-1
    function_tester(rng, F.deformable_convolution,
                    ref_deformable_convolution_2d, inputs, func_args,
                    atol_f=1e-4, atol_b=1e-2, atol_accum=1e-5, dstep=1e-2,
                    ctx=ctx, func_name=func_name, atol_half=atol_half)
