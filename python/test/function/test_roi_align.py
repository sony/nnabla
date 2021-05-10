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
from nbla_test_utils import list_context

ctxs = list_context('RoiAlign')


def ref_roi_align(input, boxes, output_size, spatial_scale, sampling_ratio,
                  aligned, channel_last):
    assert len(input.shape) == 4
    assert len(boxes.shape) == 2

    def _roi_align(image, roi, output_size, sampling_ratio):
        channels, height, width = image.shape

        def linspace(start, stop, num):
            return np.linspace(start, stop, num, endpoint=False,
                               retstep=True, dtype=np.float32)

        steps = output_size * sampling_ratio
        x, x_step = linspace(roi[0], roi[2], steps[1])
        y, y_step = linspace(roi[1], roi[3], steps[0])
        x, y = x + 0.5 * x_step, y + 0.5 * y_step

        # compute out-of-bounds mask before clipping
        oob_x = np.where(x < -1, 0, np.where(x > width, 0, 1)).astype(bool)
        oob_y = np.where(y < -1, 0, np.where(y > height, 0, 1)).astype(bool)
        oob_mask = oob_y.reshape(-1, 1) @ oob_x.reshape(1, -1)

        x, y = np.clip(x, 0, width - 1), np.clip(y, 0, height - 1)

        # left/right x and top/bottom y coordinates
        lx, rx = np.floor(x).astype(int), np.ceil(x).astype(int)
        ty, by = np.floor(y).astype(int), np.ceil(y).astype(int)

        # distance values around sampling point
        dlx = np.broadcast_to(x - lx, (channels, len(y), len(x)))
        dty = np.broadcast_to(
            y - ty, (channels, len(x), len(y))).swapaxes(1, 2)
        dlx, dty = dlx.astype(np.float32), dty.astype(np.float32)
        drx, dby = 1 - dlx, 1 - dty

        # image values around sampling point
        tl = tuple(np.meshgrid(range(channels), ty, lx, indexing='ij'))
        tr = tuple(np.meshgrid(range(channels), ty, rx, indexing='ij'))
        bl = tuple(np.meshgrid(range(channels), by, lx, indexing='ij'))
        br = tuple(np.meshgrid(range(channels), by, rx, indexing='ij'))

        # bilinear interpolate
        result = (image[tl] * drx * dby + image[tr] * dlx * dby +
                  image[bl] * drx * dty + image[br] * dlx * dty)
        # nullify out-of-bounds
        result = result * oob_mask
        # pool the sampling ratio
        result = (result.reshape(-1, output_size[0], sampling_ratio[0],
                                 output_size[1], sampling_ratio[1])
                  .mean(axis=(2, 4)))
        return result

    if channel_last:
        input = np.transpose(input, (0, 3, 1, 2))

    output = list()
    for box in boxes:
        img = input[int(box[0])]
        roi = box[1:] * (2 * spatial_scale[::-1]) - 0.5 * aligned
        if not aligned:
            roi[2:] = roi[:2] + np.maximum(roi[2:] - roi[:2], 1)
        if sampling_ratio > 0:
            _sampling_ratio = np.array([sampling_ratio, sampling_ratio])
        else:
            roi_size = np.maximum(roi[2:] - roi[:2], 1)[::-1]  # flip x,y
            _sampling_ratio = np.ceil(roi_size / output_size).astype(int)
        output.append(_roi_align(img, roi, output_size, _sampling_ratio))
    output = np.stack(output)

    if channel_last:
        output = np.transpose(output, (0, 2, 3, 1))

    return output


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("inshape, boxes", [
    ((2, 3, 10, 10), ([[0, 0, 0, 10, 10], [1, 0, 0, 10, 10]])),
    ((2, 3, 11, 12), ([[1, 1, 2, 8, 7], [0, -2, 0, 11, 8]])),
])
@pytest.mark.parametrize('output_size', [(10, 10), (5, 15), (13, 8)])
@pytest.mark.parametrize('spatial_scale', [(1.0, 1.5), (0.7, 1.0)])
@pytest.mark.parametrize('sampling_ratio', [-1, 0, 1, 2])
@pytest.mark.parametrize('aligned', [False, True])
@pytest.mark.parametrize('channel_last', [False, True])
@pytest.mark.parametrize("seed", [313])
def test_roi_align_forward_backward(seed, inshape, boxes, output_size,
                                    spatial_scale, sampling_ratio, aligned,
                                    channel_last, ctx, func_name):
    from nbla_test_utils import function_tester
    if channel_last and not func_name.endswith('Cuda'):
        pytest.skip('channel_last=True is only supported in CUDA backend')
    rng = np.random.RandomState(seed)
    inputs = [
        rng.randn(*inshape).astype(np.float32),
        np.array(boxes, dtype=np.float32)
    ]
    if channel_last:
        inputs[0] = np.transpose(inputs[0], (0, 2, 3, 1))
    func_args = [
        #output_size, 2 * [spatial_scale], sampling_ratio, aligned, channel_last,
        output_size, spatial_scale, sampling_ratio, aligned, channel_last,
    ]
    function_tester(rng, F.roi_align, ref_roi_align, inputs, func_args,
                    atol_f=1e-4, atol_b=1e-2, backward=[True, False],
                    ctx=ctx, func_name=func_name)
