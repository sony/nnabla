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
from nbla_test_utils import list_context

from refs import (generate_transformation_2d, generate_transformation_3d,
                  affine_grid_2d, affine_grid_3d, ChannelLastToFirstTranspose)

from itertools import product

ctxs = list_context('WarpByGrid')


def create_inputs(rng, batch_size, channels, size_out, size_inp, align_corners):
    if len(size_out) == 2:
        inp = rng.randn(batch_size, channels,
                        size_inp[0], size_inp[1]).astype(np.float32)
        affine = generate_transformation_2d(rng, batch_size)
        grid_s = affine_grid_2d(affine, size_out, align_corners)
    elif len(size_out) == 3:
        inp = rng.randn(batch_size, channels,
                        size_inp[0], size_inp[1], size_inp[2]).astype(np.float32)
        affine = generate_transformation_3d(rng, batch_size)
        grid_s = affine_grid_3d(affine, size_out, align_corners)
    return inp, grid_s


def unnormalize_grid(s, S, align_corners):
    if align_corners:
        # [-1, 1] <--> [0, S - 1]
        return (s + 1.0) * (S - 1) / 2.0
    else:
        # [-1, 1] <--> [0.5, S - 0.5] = [0 + 0.5, S - 1 + 0.5]
        return ((s + 1.0) * S - 1.0) / 2.0


def get_src_findex_by_pad(s, S, padding_mode, align_corners):
    if padding_mode == "zero":
        return get_src_findex_with_zero_pad(s, S)
    elif padding_mode == "reflect":
        if align_corners:
            return get_src_findex_with_reflect_pad(s, S, True)
        else:
            sf = get_src_findex_with_reflect_pad(s, S, False)
            return get_src_findex_with_repeat_pad(sf, S)
    elif padding_mode == "repeat":
        return get_src_findex_with_repeat_pad(s, S)


def get_src_findex_with_zero_pad(s, S):
    return s


def get_src_findex_with_repeat_pad(s, S):
    if s < 0:
        return 0
    elif s > S - 1:
        return S - 1
    else:
        return s


def get_src_findex_with_reflect_pad(s, S, align_corners):
    def reflect(s, L, U):
        length = (U - L)
        if s < L:
            d = L - s
            nf = d / length
            n = int(np.floor(nf))
            r = d - n * length
            if n % 2 == 0:
                return L + r
            else:
                return U - r
        elif (s > U):
            d = s - U
            nf = d / length
            n = int(np.floor(nf))
            r = d - n * length
            if n % 2 == 0:
                return U - r
            else:
                return L + r
        else:
            return s

    if align_corners:
        return reflect(s, 0.0, S - 1.0)
    else:
        sf = reflect(2 * s, -1, 2 * S - 1)
        sf *= 0.5
        return sf


def get_pixel_value_2d(inp, b, c, h, w, H, W):
    if (h >= 0 and h < H) and (w >= 0 and w < W):
        return inp[b, c, h, w]
    else:
        return 0.0


def get_pixel_value_3d(inp, b, c, d, h, w, D, H, W):
    if (d >= 0 and d < D) and (h >= 0 and h < H) and (w >= 0 and w < W):
        return inp[b, c, d, h, w]
    else:
        return 0.0


def warp_by_grid_linear_2d(inp, grid, mode, padding_mode, align_corners, channel_last):
    B, C, Hi, Wi = inp.shape
    B, Ho, Wo, _ = grid.shape
    out = np.zeros((B, C, Ho, Wo))
    for b, c, h, w in product(range(B), range(C), range(Ho), range(Wo)):
        xf = unnormalize_grid(grid[b, h, w, 0], Wi, align_corners)
        yf = unnormalize_grid(grid[b, h, w, 1], Hi, align_corners)
        xf = get_src_findex_by_pad(xf, Wi, padding_mode, align_corners)
        yf = get_src_findex_by_pad(yf, Hi, padding_mode, align_corners)
        xi0 = np.floor(xf)
        yi0 = np.floor(yf)
        xi0 = int(xi0)
        yi0 = int(yi0)
        xi1 = xi0 + 1
        yi1 = yi0 + 1
        px0 = xf - xi0
        py0 = yf - yi0
        px1 = 1.0 - px0
        py1 = 1.0 - py0

        v_y0x0 = get_pixel_value_2d(inp, b, c, yi0, xi0, Hi, Wi)
        v_y0x1 = get_pixel_value_2d(inp, b, c, yi0, xi1, Hi, Wi)
        v_y1x0 = get_pixel_value_2d(inp, b, c, yi1, xi0, Hi, Wi)
        v_y1x1 = get_pixel_value_2d(inp, b, c, yi1, xi1, Hi, Wi)

        val = (v_y0x0 * py1 * px1) + (v_y0x1 * py1 * px0) \
            + (v_y1x0 * py0 * px1) + (v_y1x1 * py0 * px0)
        out[b, c, h, w] = val
    return out


def warp_by_grid_linear_3d(inp, grid, mode, padding_mode, align_corners, channel_last):
    B, C, Di, Hi, Wi = inp.shape
    B, Do, Ho, Wo, _ = grid.shape
    out = np.zeros((B, C, Do, Ho, Wo))
    for b, c, d, h, w in product(range(B), range(C), range(Do), range(Ho), range(Wo)):
        xf = unnormalize_grid(grid[b, d, h, w, 0], Wi, align_corners)
        yf = unnormalize_grid(grid[b, d, h, w, 1], Hi, align_corners)
        zf = unnormalize_grid(grid[b, d, h, w, 2], Di, align_corners)
        xf = get_src_findex_by_pad(xf, Wi, padding_mode, align_corners)
        yf = get_src_findex_by_pad(yf, Hi, padding_mode, align_corners)
        zf = get_src_findex_by_pad(zf, Di, padding_mode, align_corners)
        xi0 = np.floor(xf)
        yi0 = np.floor(yf)
        zi0 = np.floor(zf)
        xi0 = int(xi0)
        yi0 = int(yi0)
        zi0 = int(zi0)
        xi1 = xi0 + 1
        yi1 = yi0 + 1
        zi1 = zi0 + 1
        px0 = xf - xi0
        py0 = yf - yi0
        pz0 = zf - zi0
        px1 = 1.0 - px0
        py1 = 1.0 - py0
        pz1 = 1.0 - pz0

        v_z0y0x0 = get_pixel_value_3d(inp, b, c, zi0, yi0, xi0, Di, Hi, Wi)
        v_z0y0x1 = get_pixel_value_3d(inp, b, c, zi0, yi0, xi1, Di, Hi, Wi)
        v_z0y1x0 = get_pixel_value_3d(inp, b, c, zi0, yi1, xi0, Di, Hi, Wi)
        v_z0y1x1 = get_pixel_value_3d(inp, b, c, zi0, yi1, xi1, Di, Hi, Wi)
        v_z1y0x0 = get_pixel_value_3d(inp, b, c, zi1, yi0, xi0, Di, Hi, Wi)
        v_z1y0x1 = get_pixel_value_3d(inp, b, c, zi1, yi0, xi1, Di, Hi, Wi)
        v_z1y1x0 = get_pixel_value_3d(inp, b, c, zi1, yi1, xi0, Di, Hi, Wi)
        v_z1y1x1 = get_pixel_value_3d(inp, b, c, zi1, yi1, xi1, Di, Hi, Wi)

        val = v_z0y0x0 * pz1 * py1 * px1 \
            + v_z0y0x1 * pz1 * py1 * px0 \
            + v_z0y1x0 * pz1 * py0 * px1 \
            + v_z0y1x1 * pz1 * py0 * px0 \
            + v_z1y0x0 * pz0 * py1 * px1 \
            + v_z1y0x1 * pz0 * py1 * px0 \
            + v_z1y1x0 * pz0 * py0 * px1 \
            + v_z1y1x1 * pz0 * py0 * px0
        out[b, c, d, h, w] = val
    return out


def warp_by_grid_nearest_2d(inp, grid, mode, padding_mode, align_corners, channel_last):
    B, C, Hi, Wi = inp.shape
    B, Ho, Wo, _ = grid.shape
    out = np.zeros((B, C, Ho, Wo))
    for b, c, h, w in product(range(B), range(C), range(Ho), range(Wo)):
        xf = unnormalize_grid(grid[b, h, w, 0], Wi, align_corners)
        yf = unnormalize_grid(grid[b, h, w, 1], Hi, align_corners)
        xf = get_src_findex_by_pad(xf, Wi, padding_mode, align_corners)
        yf = get_src_findex_by_pad(yf, Hi, padding_mode, align_corners)
        xi = np.floor(xf + 0.5)
        yi = np.floor(yf + 0.5)
        xi = int(xi)
        yi = int(yi)

        val = get_pixel_value_2d(inp, b, c, yi, xi, Hi, Wi)
        out[b, c, h, w] = val
    return out


def warp_by_grid_nearest_3d(inp, grid, mode, padding_mode, align_corners, channel_last):
    B, C, Di, Hi, Wi = inp.shape
    B, Do, Ho, Wo, _ = grid.shape
    out = np.zeros((B, C, Do, Ho, Wo))
    for b, c, d, h, w in product(range(B), range(C), range(Do), range(Ho), range(Wo)):
        xf = unnormalize_grid(grid[b, d, h, w, 0], Wi, align_corners)
        yf = unnormalize_grid(grid[b, d, h, w, 1], Hi, align_corners)
        zf = unnormalize_grid(grid[b, d, h, w, 2], Di, align_corners)
        xf = get_src_findex_by_pad(xf, Wi, padding_mode, align_corners)
        yf = get_src_findex_by_pad(yf, Hi, padding_mode, align_corners)
        zf = get_src_findex_by_pad(zf, Di, padding_mode, align_corners)
        xi = np.floor(xf + 0.5)
        yi = np.floor(yf + 0.5)
        zi = np.floor(zf + 0.5)
        xi = int(xi)
        yi = int(yi)
        zi = int(zi)

        val = get_pixel_value_3d(inp, b, c, zi, yi, xi, Di, Hi, Wi)
        out[b, c, d, h, w] = val
    return out


def ref_warp_by_grid(inp, grid, mode, padding_mode, align_corners, channel_last):
    if channel_last:
        size = grid.shape[1:-1]
        n_sdim = 2 if len(size) == 2 else 3
        t = ChannelLastToFirstTranspose(inp.ndim, n_sdim)
        inp = t(inp)

    if mode == "linear":
        if len(inp.shape) == 4:
            out = warp_by_grid_linear_2d(
                inp, grid, mode, padding_mode, align_corners, channel_last)
        if len(inp.shape) == 5:
            out = warp_by_grid_linear_3d(
                inp, grid, mode, padding_mode, align_corners, channel_last)
    if mode == "nearest":
        if len(inp.shape) == 4:
            out = warp_by_grid_nearest_2d(
                inp, grid, mode, padding_mode, align_corners, channel_last)
        if len(inp.shape) == 5:
            out = warp_by_grid_nearest_3d(
                inp, grid, mode, padding_mode, align_corners, channel_last)

    out = t.inv(out) if channel_last else out
    return out


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("batch_size, channels, size_inp, size_out",
                         [  # 2D
                          (1, 1, (5, 5), (6, 6)),
                          (2, 1, (6, 6), (5, 5)),
                          (1, 2, (5, 4), (5, 4)),
                          (1, 1, (4, 5), (4, 5)),
                          (2, 2, (4, 4), (4, 4)),
                          # 3D
                          (1, 1, (8, 8, 8), (4, 4, 4)),
                          ])
@pytest.mark.parametrize("mode", ["linear", "nearest"])
@pytest.mark.parametrize("padding_mode", ["zero", "reflect", "repeat"])
@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("channel_last", [False, True])
def test_warp_by_grid_forward_backward(seed, ctx, func_name, size_out, size_inp, channels, batch_size,
                                       mode, padding_mode, align_corners, channel_last):
    if channel_last and func_name == "WarpByGrid":
        pytest.skip("channel_last=True is not supported in CPU.")

    if mode == "nearest":
        backward = [True, False]
    else:
        backward = [True, True]

    from nbla_test_utils import function_tester
    rng = np.random.RandomState(seed)
    inp, grid_s = create_inputs(
        rng, batch_size, channels, size_out, size_inp, align_corners)

    if channel_last:
        inp = inp.transpose((0, 2, 3, 1)) if len(
            size_inp) == 2 else inp.transpose((0, 2, 3, 4, 1))

    inputs = [inp]
    inputs += [grid_s]
    func_args = [mode, padding_mode, align_corners, channel_last]
    function_tester(rng, F.warp_by_grid, ref_warp_by_grid, inputs, func_args=func_args,
                    ctx=ctx, func_name=func_name, disable_half_test=True,
                    backward=backward, atol_b=1e-2, atol_accum=1e-2)
