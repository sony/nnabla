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

from __future__ import absolute_import

import numpy as np
from PIL import Image
import imageio

from .common import rescale_pixel_intensity

interpolations_map = {
    "nearest": Image.NEAREST,
    "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC,
}

if hasattr(Image, "HAMMING"):  # version >3.4.0
    interpolations_map["hamming"] = Image.HAMMING

if hasattr(Image, "BOX"):  # version >3.4.0
    interpolations_map["box"] = Image.BOX

if hasattr(Image, "LANCZOS"):  # version >1.1.3
    interpolations_map["lanczos"] = Image.LANCZOS


def get_byte_image(img):
    if img.dtype == np.uint8:
        return img.copy()
    else:
        return rescale_pixel_intensity(img, input_low=0, input_high=1, output_low=0, output_high=255,
                                       output_type=np.uint8)


def imread(path, grayscale=False, size=None, interpolate="bilinear", channel_first=False):
    """

    :param path: str of filename or file object
    :param grayscale: bool
    :param size: tupple of int (width, height)
        if None, resize is not done and img shape depends on the files.
    :param interpolate: str
        must be one of ["nearest", "box", "bilinear", "hamming", "bicubic", "lanczos"]
    :param channel_first:
        This argument specifies the shape of img is whether (height, width, channel) or (channel, height, width).
        Default value is False, which means the img shape is (height, width, channel)
    :return: numpy.ndarray
    """
    if Image is None:
        raise ImportError("cannot import PIL.Image")

    img = Image.open(path, mode="r")

    if grayscale:
        img = img.convert("L")

    if size is not None:
        img = imresize(img, size, interpolate=interpolate)

    img = np.array(img)

    if channel_first and len(img.shape) == 3:
        img = img.transpose((2, 0, 1))

    return img


def imsave(path, img, channel_first=False):
    """

    :param path: str, output filename
    :param img: numpy.ndarray whose shape is (channel, height, width)
    :param channel_first:
        This argument specifies the shape of img is whether (height, width, channel) or (channel, height, width).
        Default value is False, which means the img shape is (height, width, channel)
    """
    if channel_first:
        img = img.transpose((1, 2, 0))

    imageio.imwrite(path, img)


def imresize(img, size, interpolate="bilinear", channel_first=False):
    """

    :param img: numpy.ndarray whose shape is (height, width, channel) or (height, width)
    :param size: tupple of int (width, height).
   :param interpolate: str
        must be one of ["nearest", "box", "bilinear", "hamming", "bicubic", "lanczos"]
    :param channel_first: bool
        This argument specifies the shape of img is whether (height, width, channel) or (channel, height, width).
        Default value isyou can get the array whose shape is  False, which means the img shape is (height, width, channel)
    :return: numpy.ndarray whose shape is (size[1], size[0], channel) or (size[1], size[0])
    """
    if not isinstance(size, (list, tuple)):
        raise ValueError("size must be list or tuple")

    if interpolate not in interpolations_map.keys():
        raise ValueError(
            "unknown interpolation type. must be [{}]".format(", ".join(list(interpolations_map.keys()))))

    high = 255

    if isinstance(img, np.ndarray):
        if channel_first and len(img.shape) == 3:
            img = img.transpose((1, 2, 0))
        byte_image = get_byte_image(img)
        pil_image = Image.fromarray(byte_image)
        output_type = img.dtype
        if output_type != np.uint8:
            high = 1
    elif isinstance(img, Image.Image):
        pil_image = img
        output_type = np.uint8
    else:
        raise ValueError(
            "the type of input img is unknown. img must be numpy.ndarray or PIL.Image")

    resized = np.array(pil_image.resize(
        size, resample=interpolations_map[interpolate]))

    if channel_first and len(img.shape) == 3:
        resized = resized.transpose((2, 0, 1))

    return rescale_pixel_intensity(resized, input_low=0, input_high=255,
                                   output_low=0, output_high=high, output_type=output_type)
