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

import numpy as np


def get_conv_out_size(w, k, p, s, d=1):
    return (w + 2 * p - (d * (k - 1) + 1)) / s + 1


def get_deconv_out_size(w, k, p, s, d):
    return s * (w - 1) - 2 * p + (d * (k - 1) + 1)


def get_pool_ignore_border_out_size(w, k, p, s):
    return int(np.ceil((w + 2 * p) * 1.0 / s))


def get_pool_ignore_border_in_out_size(w, k, p, s):
    o = get_pool_ignore_border_out_size(w, k, p, s)
    i = k + (o - 1) * s
    return i, o


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


def deconvolution_2d(x, w, b, pad, stride, dilation, group, dtype=np.float32):
    y = x
    K, Ho, Wo = y.shape
    K, Cg, M, N = w.shape
    C = Cg * group

    H = get_deconv_out_size(Ho, M, pad[0], stride[0], dilation[0])
    W = get_deconv_out_size(Wo, N, pad[1], stride[1], dilation[1])
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


def pooling_2d(x, mode, kernel, stride, pad, ignore_border=True, including_pad=True, dtype=np.float32):
    """
    """
    assert mode in ['average', 'sum', 'max']

    C, H, W = x.shape
    if ignore_border:
        Ho = get_conv_out_size(H, kernel[0], pad[0], stride[0])
        Wo = get_conv_out_size(W, kernel[1], pad[1], stride[1])
        Hi = H + 2 * pad[0]
        Wi = W + 2 * pad[1]
    else:
        Hi, Ho = get_pool_ignore_border_in_out_size(
            H, kernel[0], pad[0], stride[0])
        Wi, Wo = get_pool_ignore_border_in_out_size(
            W, kernel[1], pad[1], stride[1])
    if mode == 'max':
        x_pad = np.ones((C, Hi, Wi), dtype=dtype) * x.min()
    else:
        x_pad = np.zeros((C, Hi, Wi), dtype=dtype) * x.min()
    x_pad[:, pad[0]:pad[0] + H, pad[1]:pad[1] + W] = x

    if mode == 'average' and not including_pad:
        b_pad = np.zeros((C, Hi, Wi), dtype=np.uint8)
        b_pad[:, pad[0]:pad[0] + H, pad[1]:pad[1] + W] = 1
    y = np.zeros((C, Ho, Wo), dtype=dtype)
    for ho in range(Ho):
        for wo in range(Wo):
            for c in range(C):
                hi = ho * stride[0] + np.arange(0, kernel[0])
                wi = wo * stride[1] + np.arange(0, kernel[1])
                yy = y[c]
                xx = x_pad[c]
                if mode == "max":
                    yy[ho, wo] = xx[np.ix_(hi, wi)].max()
                elif mode == "average":
                    if including_pad:
                        yy[ho, wo] = xx[np.ix_(hi, wi)].mean()
                    else:
                        yy[ho, wo] = xx[np.ix_(hi, wi)].sum(
                        ) / b_pad[c][np.ix_(hi, wi)].sum()
                elif mode == "sum":
                    yy[ho, wo] = xx[np.ix_(hi, wi)].sum()
                else:
                    raise ValueError("Unknown mode.")
    return y
