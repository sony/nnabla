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
import nnabla as nn
from imageio import imread
from collections import namedtuple


def calculate_scale(in_size, out_size, align_corners):
    if align_corners and out_size > 1:
        scale = np.float32(in_size - 1) / np.float32((out_size - 1))
    else:
        scale = np.float32(in_size) / np.float32(out_size)
    return scale


def half_pixel_scaler(x: int, scale: float):
    return (np.float32(x) + np.float32(0.5)) * scale - np.float32(0.5)


def legacyscaler(x: int, scale: float):
    return np.float32(x) * scale


def compute_interpolation_weights(out_size: int, in_size: int, scale: float, half_pixel_centers: bool):

    if half_pixel_centers:
        scaler = half_pixel_scaler
    else:
        scaler = legacyscaler

    Interpolation = namedtuple('Interpolation', 'lower upper lerp')
    # example = Interpolation(30, 43, 0.3) returns Interpolation(lower=30, upper=43, lerp=0.3)

    lerp_weight = list()
    lerp_weight.append(Interpolation(0, 0, 0))

    for i in range(out_size - 1, -1, -1):
        in_ = scaler(i, scale)
        in_f = np.floor(in_)
        lower = np.maximum(np.int64(in_f), np.int64(0))
        upper = np.minimum(np.int64(np.ceil(in_)), in_size - 1)
        lerp = in_ - in_f
        lerp_weight.append(Interpolation(lower, upper, lerp))
    lerp_weight = lerp_weight[::-1]

    return lerp_weight


def compute_lerp(top_left: float, top_right: float, bottom_left: float, bottom_right: float, x_lerp: float, y_lerp: float):
    top = top_left + (top_right - top_left) * x_lerp
    bottom = bottom = bottom_left + (bottom_right - bottom_left) * x_lerp
    return top + (bottom - top) * y_lerp


def tf_resizebilinear(x, scale=None, output_size=(299, 299), align_corners=False, half_pixel_centers=False):
    """Same interpolation as TensorFlow's ResizeBilinear.
    """
    # expect batched inputs.
    assert len(x.shape) == 4
    assert scale or output_size, 'Need either scale or output_size.'

    if x.shape[3] == 3:
        x = np.transpose(x, (0, 3, 1, 2))  # NHWC -> NCHW

    if not output_size:
        output_size = np.floor(np.array(scale) * x.shape[-len(scale):])
        output_size = tuple(map(int, output_size))

    in_height, in_width = x.shape[2:]
    out_height, out_width = output_size

    height_scale = calculate_scale(in_height, out_height, align_corners)
    width_scale = calculate_scale(in_width, out_width, align_corners)

    ys = compute_interpolation_weights(
        out_height, in_height, height_scale, half_pixel_centers)
    xs = compute_interpolation_weights(
        out_width, in_width, width_scale, half_pixel_centers)

    output = np.zeros(x.shape[:2] + output_size, dtype=np.float32)

    for _y in range(out_height):
        for _x in range(out_width):
            output[:, :, _y, _x] = compute_lerp(
                    x[:, :, ys[_y].lower, xs[_x].lower],
                    x[:, :, ys[_y].lower, xs[_x].upper],
                    x[:, :, ys[_y].upper, xs[_x].lower],
                    x[:, :, ys[_y].upper, xs[_x].upper],
                    xs[_x].lerp,
                    ys[_y].lerp)
    return output


def im2ndarray(image_paths, imsize=(299, 299), normalize=True):
    """
        retrieve image paths first, then convert each image
        to nn.Variable. len(image_paths) must be the same as batch_size.

        Args: 
            image_paths (list): list containing paths of images.
            imsize (tuple of int): resized image height and width.
            normalize (bool): if True (by default), normalize images
                              so that the values are within [-1., +1.]. 
        Returns:
            _ (nn.Variable): Variable converted from images.
        TODO: enable imresize, accept multi resolution images.
    """
    for i, image_path in enumerate(image_paths):
        image = imread(image_path)
        image = image.astype(np.float32)  # cast to float
        if i == 0:
            images = np.expand_dims(image, 0)
        else:
            image = np.expand_dims(image, 0)  # not images
            images = np.concatenate([images, image])

    images = tf_resizebilinear(
        images, output_size=imsize, align_corners=False, half_pixel_centers=False)
    if normalize:
        images = (images - 128.) / 128.

    return nn.NdArray.from_numpy_array(images)
