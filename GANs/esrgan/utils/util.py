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


def array_to_image(array):
    array = array.squeeze().astype(float).clip(0, 1)  # clamp
    n_dim = array.ndim
    if n_dim == 3:
        img_np = np.transpose(array[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 2:
        img_np = array
    else:
        raise TypeError(
            'Only support 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))

    img_np = (img_np * 255.0).round()
    # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(np.uint8)


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))
