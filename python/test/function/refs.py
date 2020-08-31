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
from six.moves import range

import itertools
import numpy as np


def get_conv_out_size(w, k, p, s, d=1):
    return (w + 2 * p - (d * (k - 1) + 1)) // s + 1


def get_deconv_out_size(w, k, p, s, d):
    return s * (w - 1) - 2 * p + (d * (k - 1) + 1)


def get_pool_out_size(w, k, p, s, ignore_border):
    return (w + p - ((k - p) if ignore_border else 1)) // s + 1


class ChannelLastToFirstTranspose(object):

    def __init__(self, dim, kdim):
        base_axis = dim - kdim - 1
        up_to_base = tuple(range(0, base_axis))
        self.axes = up_to_base + (dim - 1,) + tuple(range(base_axis, dim - 1))
        self.inv_axes = up_to_base + \
            tuple(range(base_axis + 1, dim)) + (base_axis,)

    def __call__(self, x):
        return x.transpose(self.axes).copy()

    def inv(self, x):
        return x.transpose(self.inv_axes).copy()


def convolution_1d(x, w, b, pad, stride, dilation, group, dtype=np.float32):
    """
    """
    C, H = x.shape
    K, Cg, M = w.shape

    Ho = get_conv_out_size(H, M, pad[0], stride[0], dilation[0])
    x_pad = np.zeros((C, H + pad[0] * 2), dtype=dtype)
    x_pad[:, pad[0]:pad[0] + H] = x
    y = np.zeros((K, Ho), dtype=dtype)
    for k in range(K):
        g = int(k // (K // group))
        for ho in range(Ho):
            hi = ho * stride[0] + np.arange(0, M) * dilation[0]
            ci = np.arange(g * Cg, (g + 1) * Cg)
            y[k, ho] = (w[k] * x_pad[np.ix_(ci, hi)]).sum()
    if b is not None:
        y += b[..., np.newaxis]
    return y


def convolution_2d(x, w, b, pad, stride, dilation, group, dtype=np.float32):
    """
    """
    C, H, W = x.shape
    K, Cg, M, N = w.shape

    Ho = get_conv_out_size(H, M, pad[0], stride[0], dilation[0])
    Wo = get_conv_out_size(W, N, pad[1], stride[1], dilation[1])
    x_pad = np.zeros((C, H + pad[0] * 2, W + pad[1] * 2), dtype=dtype)
    x_pad[:, pad[0]:pad[0] + H, pad[1]:pad[1] + W] = x
    y = np.zeros((K, Ho, Wo), dtype=dtype)
    for k in range(K):
        g = int(k // (K // group))
        for ho in range(Ho):
            for wo in range(Wo):
                hi = ho * stride[0] + np.arange(0, M) * dilation[0]
                wi = wo * stride[1] + np.arange(0, N) * dilation[1]
                ci = np.arange(g * Cg, (g + 1) * Cg)
                y[k, ho, wo] = (w[k] * x_pad[np.ix_(ci, hi, wi)]).sum()
    if b is not None:
        y += b[..., np.newaxis, np.newaxis]
    return y


def convolution_nd(x, w, b, pad, stride, dilation, group, dtype=np.float32):
    """
    """
    C = x.shape[0]
    inshape = x.shape[1:]
    ndim = len(inshape)
    assert w.ndim == ndim + 2
    K, Cg = w.shape[:2]
    kshape = w.shape[2:]

    def get_conv_out_size_recursive(d, ndim):
        if d == ndim:
            return []
        s = get_conv_out_size(
            inshape[d], kshape[d], pad[d], stride[d], dilation[d])
        return [s] + get_conv_out_size_recursive(d + 1, ndim)

    outshape = get_conv_out_size_recursive(0, ndim)
    inshape_pad = [C] + [inshape[d] + 2 * pad[d] for d in range(ndim)]
    x_pad = np.zeros(inshape_pad, dtype=dtype)
    x_pad[[slice(None,)] + [slice(pad[d], pad[d] + inshape[d])
                            for d in range(ndim)]] = x
    y = np.zeros([K] + outshape, dtype=dtype)
    for k in range(K):
        g = int(k // (K // group))
        for outindex in itertools.product(*map(range, outshape)):
            inindex = [outindex[d] * stride[d] +
                       np.arange(0, kshape[d]) * dilation[d] for d in range(ndim)]
            ci = np.arange(g * Cg, (g + 1) * Cg)
            y[(k,) + tuple(outindex)] = (w[k] *
                                         x_pad[np.ix_(ci, *inindex)]).sum()
    if b is not None:
        y += b[[Ellipsis] + [np.newaxis for d in range(ndim)]]
    return y


def deconvolution_1d(x, w, b, pad, stride, dilation, group, dtype=np.float32,
                     output_padding=(0,)):
    y = x
    K, Ho = y.shape
    K, Cg, M = w.shape
    C = Cg * group

    H = (get_deconv_out_size(Ho, M, pad[0], stride[0], dilation[0])
         + output_padding[0])
    x_pad = np.zeros((C, H + pad[0] * 2), dtype=dtype)
    for k in range(K):
        g = int(k // (K // group))
        for ho in range(Ho):
            hi = ho * stride[0] + np.arange(0, M) * dilation[0]
            ci = np.arange(g * Cg, (g + 1) * Cg)
            x_pad[np.ix_(ci, hi)] += w[k] * y[k, ho]
    x = x_pad[:, pad[0]:pad[0] + H]
    if b is not None:
        x += b[..., np.newaxis]
    return x


def deconvolution_2d(x, w, b, pad, stride, dilation, group, dtype=np.float32,
                     output_padding=(0, 0)):
    y = x
    K, Ho, Wo = y.shape
    K, Cg, M, N = w.shape
    C = Cg * group

    H = (get_deconv_out_size(Ho, M, pad[0], stride[0], dilation[0])
         + output_padding[0])
    W = (get_deconv_out_size(Wo, N, pad[1], stride[1], dilation[1])
         + output_padding[1])
    x_pad = np.zeros((C, H + pad[0] * 2, W + pad[1] * 2), dtype=dtype)
    for k in range(K):
        g = int(k // (K // group))
        for ho in range(Ho):
            for wo in range(Wo):
                hi = ho * stride[0] + np.arange(0, M) * dilation[0]
                wi = wo * stride[1] + np.arange(0, N) * dilation[1]
                ci = np.arange(g * Cg, (g + 1) * Cg)
                x_pad[np.ix_(ci, hi, wi)] += w[k] * y[k, ho, wo]
    x = x_pad[:, pad[0]:pad[0] + H, pad[1]:pad[1] + W]
    if b is not None:
        x += b[..., np.newaxis, np.newaxis]
    return x


def pooling_2d(x, mode, kernel, stride, pad, ignore_border=True,
               including_pad=True, dtype=np.float32):
    """
    """
    assert mode in ['average', 'sum', 'max']

    C, H, W = x.shape
    Ho = get_pool_out_size(H, kernel[0], pad[0], stride[0], ignore_border)
    Wo = get_pool_out_size(W, kernel[1], pad[1], stride[1], ignore_border)
    Hi = H + pad[0] + (pad[0] if ignore_border else kernel[0] - 1)
    Wi = W + pad[1] + (pad[1] if ignore_border else kernel[1] - 1)

    x_pad = np.ones((C, Hi, Wi), dtype=dtype)
    x_pad *= x.min() if mode == 'max' else 0
    x_pad[:, pad[0]:pad[0] + H, pad[1]:pad[1] + W] = x

    if mode == 'average':
        b_pad = np.zeros((C, Hi, Wi), dtype=np.uint)
        h_beg = int(not including_pad) * pad[0]
        w_beg = int(not including_pad) * pad[1]
        h_end = H + (1 + int(including_pad)) * pad[0]
        w_end = W + (1 + int(including_pad)) * pad[1]
        b_pad[:, h_beg:h_end, w_beg:w_end] = 1

    y = np.zeros((C, Ho, Wo), dtype=dtype)

    for c in range(C):
        for ho in range(Ho):
            for wo in range(Wo):
                hi = ho * stride[0] + np.arange(0, kernel[0])
                wi = wo * stride[1] + np.arange(0, kernel[1])
                yy = y[c]
                xx = x_pad[c]
                if mode == "max":
                    yy[ho, wo] = xx[np.ix_(hi, wi)].max()
                elif mode == "sum":
                    yy[ho, wo] = xx[np.ix_(hi, wi)].sum()
                elif mode == "average":
                    pad_sum = xx[np.ix_(hi, wi)].sum()
                    pad_cnt = b_pad[c][np.ix_(hi, wi)].sum()
                    yy[ho, wo] = pad_sum / pad_cnt
    return y


def pooling_3d(x, mode, kernel, stride, pad, ignore_border=True,
               including_pad=True, dtype=np.float32):
    """
    """
    assert mode in ['average', 'sum', 'max']

    C, Z, H, W = x.shape
    Zo = get_pool_out_size(Z, kernel[0], pad[0], stride[0], ignore_border)
    Ho = get_pool_out_size(H, kernel[1], pad[1], stride[1], ignore_border)
    Wo = get_pool_out_size(W, kernel[2], pad[2], stride[2], ignore_border)
    Zi = Z + pad[0] + (pad[0] if ignore_border else kernel[0] - 1)
    Hi = H + pad[1] + (pad[1] if ignore_border else kernel[1] - 1)
    Wi = W + pad[2] + (pad[2] if ignore_border else kernel[2] - 1)

    x_pad = np.ones((C, Zi, Hi, Wi), dtype=dtype)
    x_pad *= x.min() if mode == 'max' else 0
    x_pad[:, pad[0]:pad[0] + Z, pad[1]:pad[1] + H, pad[2]:pad[2] + W] = x

    if mode == 'average':
        b_pad = np.zeros((C, Zi, Hi, Wi), dtype=np.uint)
        z_beg = int(not including_pad) * pad[0]
        h_beg = int(not including_pad) * pad[1]
        w_beg = int(not including_pad) * pad[2]
        z_end = Z + (1 + int(including_pad)) * pad[0]
        h_end = H + (1 + int(including_pad)) * pad[1]
        w_end = W + (1 + int(including_pad)) * pad[2]
        b_pad[:, z_beg:z_end, h_beg:h_end, w_beg:w_end] = 1
        #b_pad[:, pad[0]:pad[0] + Z, pad[1]:pad[1] + H, pad[2]:pad[2] + W] = 1

    y = np.zeros((C, Zo, Ho, Wo), dtype=dtype)

    for c in range(C):
        for zo in range(Zo):
            for ho in range(Ho):
                for wo in range(Wo):
                    zi = zo * stride[0] + np.arange(0, kernel[0])
                    hi = ho * stride[1] + np.arange(0, kernel[1])
                    wi = wo * stride[2] + np.arange(0, kernel[2])
                    yy = y[c]
                    xx = x_pad[c]
                    if mode == "max":
                        yy[zo, ho, wo] = xx[np.ix_(zi, hi, wi)].max()
                    elif mode == "sum":
                        yy[zo, ho, wo] = xx[np.ix_(zi, hi, wi)].sum()
                    elif mode == "average":
                        pool_sum = xx[np.ix_(zi, hi, wi)].sum()
                        pool_cnt = b_pad[c][np.ix_(zi, hi, wi)].sum()
                        yy[zo, ho, wo] = pool_sum / pool_cnt
    return y


def generate_rotation_2d(rng, B):
    rotates = []
    for i in range(B):
        degree = 2 * np.pi * (2.0 * rng.rand() - 1.0)
        c, s = np.cos(degree), np.sin(degree)
        rotate = np.asarray([[c, -s],
                             [s, c]])
        rotates.append(rotate)
    return np.asarray(rotates)


def generate_rotation_3d(rng, B):
    rotates = []
    for i in range(B):
        alpha = np.pi * (2.0 * rng.rand() - 1.0)
        beta = np.pi / 2.0 * (2.0 * rng.rand() - 1.0)
        gamma = np.pi * (2.0 * rng.rand() - 1.0)

        c, s = np.cos(alpha), np.sin(alpha)
        Ra = np.asarray([[c, -s, 0],
                         [s, c, 0],
                         [0, 0, 1]])
        c, s = np.cos(beta), np.sin(beta)
        Rb = np.asarray([[c, 0, s],
                         [0, 1, 0],
                         [-s, 0, c]])
        c, s = np.cos(gamma), np.sin(gamma)
        Rg = np.asarray([[1, 0, 0],
                         [0, c, -s],
                         [0, s, c]])
        rotate = Ra.dot(Rb).dot(Rg)
        rotates.append(rotate)
    return np.asarray(rotates)


def generate_transformation_2d(rng, batch_size):
    rotate = generate_rotation_2d(rng, batch_size)
    translate = (2.0 * rng.rand(batch_size, 2, 1) - 1.0) * 0.001
    theta = np.concatenate([rotate, translate], axis=2)
    return theta.astype(np.float32)


def generate_transformation_3d(rng, batch_size):
    rotate = generate_rotation_3d(rng, batch_size)
    translate = (2.0 * rng.rand(batch_size, 3, 1) - 1.0) * 0.001
    theta = np.concatenate([rotate, translate], axis=2)
    return theta.astype(np.float32)


def generate_normalized_grid_2d(B, size, align_corners):
    H, W = size
    hgrid = np.linspace(-1.0, 1.0, H)
    wgrid = np.linspace(-1.0, 1.0, W)
    hgrid = hgrid if align_corners else hgrid * (H - 1) / H
    wgrid = wgrid if align_corners else wgrid * (W - 1) / W
    w, h = np.meshgrid(wgrid, hgrid)

    x = w.reshape(-1)
    y = h.reshape(-1)
    t = np.ones(len(x))
    normalized_grid = np.stack((x, y, t), axis=1)
    normalized_grid = normalized_grid.reshape(H, W, 3)
    normalized_grid = np.repeat(
        normalized_grid[np.newaxis, :, :, :], B, axis=0)
    return normalized_grid.astype(np.float32)


def generate_normalized_grid_3d(B, size, align_corners):
    D, H, W = size
    dgrid = np.linspace(-1.0, 1.0, D)
    hgrid = np.linspace(-1.0, 1.0, H)
    wgrid = np.linspace(-1.0, 1.0, W)
    dgrid = dgrid if align_corners else dgrid * (D - 1) / D
    hgrid = hgrid if align_corners else hgrid * (H - 1) / H
    wgrid = wgrid if align_corners else wgrid * (W - 1) / W
    h, d, w = np.meshgrid(hgrid, dgrid, wgrid)

    x = w.reshape(-1)
    y = h.reshape(-1)
    z = d.reshape(-1)
    t = np.ones(len(x))
    normalized_grid = np.stack((x, y, z, t), axis=1)
    normalized_grid = normalized_grid.reshape(D, H, W, 4)
    normalized_grid = np.repeat(
        normalized_grid[np.newaxis, :, :, :, :], B, axis=0)
    return normalized_grid.astype(np.float32)


def affine_grid_2d(affine, size, align_corners):
    B = affine.shape[0]
    H, W = size
    grid_t = generate_normalized_grid_2d(B, size, align_corners)
    grid_s = np.matmul(grid_t.reshape(B, H * W, 3),
                       affine.transpose((0, 2, 1)))
    grid_s = grid_s.reshape(B, H, W, 2)
    return grid_s.astype(np.float32)


def affine_grid_3d(affine, size, align_corners):
    B = affine.shape[0]
    D, H, W = size
    grid_t = generate_normalized_grid_3d(B, size, align_corners)
    grid_s = np.matmul(grid_t.reshape(B, D * H * W, 4),
                       affine.transpose((0, 2, 1)))
    grid_s = grid_s.reshape(B, D, H, W, 3)
    return grid_s.astype(np.float32)


def pad_sequence(sequences, batch_first):
    # sequences: list of nparray
    # sequences[i]: (T_i, D_1, ..., D_M)
    Ds = () if len(sequences[0].shape) == 1 else sequences[0].shape[1:]
    B = len(sequences)
    T = max([seq.shape[0] for seq in sequences])
    data = np.zeros((B, T) + Ds) if batch_first else np.zeros((T, B) + Ds)
    for b, seq in enumerate(sequences):
        l = seq.shape[0]
        if batch_first:
            data[b, :l] = seq
        else:
            data[:l, b] = seq
    return data
