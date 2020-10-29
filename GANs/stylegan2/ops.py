# Copyright (c) 2020 Sony Corporation. All Rights Reserved.
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
import nnabla as nn
import nnabla.functions as F


"""
    these operations are based on the official implementations of styleGAN2.
    which can be found at https://github.com/NVlabs/stylegan2.
"""


def _setup_kernel(k):
    k = np.asarray(k, dtype=np.float32)
    if k.ndim == 1:
        k = np.outer(k, k)
    k /= np.sum(k)
    assert k.ndim == 2
    assert k.shape[0] == k.shape[1]
    return k


def lerp(a, b, t):
    """
        linear interpolation.
    """
    return a + (b - a) * t


def _simple_upfirdn_2d(x, k, up=1, down=1, pad0=0, pad1=0):
    assert x.ndim == 4
    y = x
    y = F.reshape(y, [-1, y.shape[2], y.shape[3], 1], inplace=True)
    y = upfirdn_2d(y, k, upx=up, upy=up, downx=down, downy=down,
                   padx0=pad0, padx1=pad1, pady0=pad0, pady1=pad1)
    y = F.reshape(y, [-1, x.shape[1], y.shape[1], y.shape[2]], inplace=True)
    return y


def upfirdn_2d(x, k, upx=1, upy=1, downx=1, downy=1, padx0=0, padx1=0, pady0=0, pady1=0):
    assert isinstance(x, nn.Variable) or isinstance(x, nn.NdArray)
    k = np.asarray(k, dtype=np.float32)
    assert x.ndim == 4
    inH = x.shape[1]
    inW = x.shape[2]
    minorDim = x.shape[3]
    kernelH, kernelW = k.shape
    assert inW >= 1 and inH >= 1
    assert kernelW >= 1 and kernelH >= 1
    assert isinstance(upx, int) and isinstance(upy, int)
    assert isinstance(downx, int) and isinstance(downy, int)
    assert isinstance(padx0, int) and isinstance(padx1, int)
    assert isinstance(pady0, int) and isinstance(pady1, int)

    # Upsample (insert zeros).
    x = F.reshape(x, [-1, inH, 1, inW, 1, minorDim], inplace=True)
    x = F.pad(x, [0, 0, 0, 0, 0, upy - 1, 0, 0, 0, upx - 1, 0, 0])
    x = F.reshape(x, [-1, inH * upy, inW * upx, minorDim], inplace=True)

    # Pad (crop if negative).
    x = F.pad(x, [0, 0, max(pady0, 0), max(pady1, 0),
                  max(padx0, 0), max(padx1, 0), 0, 0])
    x = x[:, max(-pady0, 0): x.shape[1] - max(-pady1, 0),
          max(-padx0, 0): x.shape[2] - max(-padx1, 0), :]

    # Convolve with filter.
    x = F.transpose(x, [0, 3, 1, 2])
    x = F.reshape(x, [-1, 1, inH * upy + pady0 + pady1,
                      inW * upx + padx0 + padx1], inplace=True)
    w = nn.Variable.from_numpy_array(k[np.newaxis, np.newaxis, ::-1, ::-1])
    x = F.convolution(x, w)
    x = F.reshape(x, [-1, minorDim, inH * upy + pady0 + pady1 - kernelH +
                      1, inW * upx + padx0 + padx1 - kernelW + 1], inplace=True)
    x = F.transpose(x, [0, 2, 3, 1])

    # Downsample (throw away pixels).
    return x[:, ::downy, ::downx, :]


def upsample_2d(x, k=None, factor=2, gain=1):
    assert isinstance(factor, int) and factor >= 1
    if k is None:
        k = [1] * factor
    k = _setup_kernel(k) * (gain * (factor ** 2))
    p = k.shape[0] - factor
    return _simple_upfirdn_2d(x, k, up=factor, pad0=(p + 1) // 2 + factor - 1, pad1=p // 2)


def upsample_conv_2d(x, w, k=None, factor=2, gain=1, group=1):
    assert isinstance(factor, int) and factor >= 1

    # Check weight shape.
    assert w.ndim == 4
    convH = w.shape[2]
    convW = w.shape[3]
    assert convW == convH

    # Setup filter kernel.
    if k is None:
        k = [1] * factor
    k = _setup_kernel(k) * (gain * (factor ** 2))
    p = (k.shape[0] - factor) - (convW - 1)

    # Execute.
    w = w[:, :, ::-1, ::-1]
    x = F.deconvolution(x, w, stride=(factor, factor), group=group)
    x = F.reshape(x, (group, -1, x.shape[2], x.shape[3]), inplace=True)
    return _simple_upfirdn_2d(x, k, pad0=(p + 1)//2 + factor - 1, pad1=p//2 + 1)


def convert_images_to_uint8(images, drange=[-1, 1]):
    """
        convert float32 -> uint8
    """
    if isinstance(images, nn.Variable):
        images = images.d
    if isinstance(images, nn.NdArray):
        images = images.data

    scale = 255 / (drange[1] - drange[0])
    images = images * scale + (0.5 - drange[0] * scale)
    return np.uint8(np.clip(images, 0, 255))
