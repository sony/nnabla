# Copyright 2020,2021 Sony Corporation.
# Copyright 2021 Sony Group Corporation.
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

from __future__ import division, absolute_import

import numpy as np
from nnabla.logger import logger


def normalize_pixel_intensity(img, input_low, input_high):
    denominator = input_high - input_low

    if denominator == 0:
        # all values are same
        if input_low == 0:
            return img  # output all values are 0
        else:
            return img / input_low  # output all values are 1

    return (img - input_low) / denominator


def rescale_pixel_intensity(img, input_low=0, input_high=255, output_low=0, output_high=255, output_type=None):
    if not isinstance(img, np.ndarray):
        raise ValueError(
            "rescale_pixel_intensity() supports only numpy.ndarray as input")

    if output_type is None:
        output_type = img.dtype

    if input_low == output_low and output_low == output_high:
        return np.asarray(img, output_type)

    # [input_low, input_high] -> [0, input_high - input_low] -> [0, 1]
    normalized = normalize_pixel_intensity(img, input_low, input_high)

    # [0, 1] -> [0, output_high - output_low] -> [output_low, output_high]
    scaled = normalized * (output_high - output_low) + output_low

    return scaled.astype(output_type)


def upscale_uint8_image(img, as_uint16):
    if as_uint16:
        return np.asarray(img, np.uint16) * 256

    return img


def upscale_float_image(img, as_uint16):
    if as_uint16:
        return np.asarray((img * 65535), np.uint16)

    return np.asarray((img * 255), np.uint8)


def upscale_pixel_intensity(img, as_uint16):
    if img.dtype == np.uint16:
        return img

    if img.dtype == np.uint8:
        return upscale_uint8_image(img, as_uint16)

    return upscale_float_image(img, as_uint16)


def check_type_and_cast_if_necessary(img, as_uint16):
    if not as_uint16 and img.dtype == np.uint16:
        raise ValueError("Input img type is uint16, but as_uint16 is False."
                         " If you want to save img as uint8, be sure all pixel values are in the range of [0, 255]"
                         " and cast it before you call imsave function.")

    if as_uint16 and img.dtype != np.uint16:
        # in the case of as_uint16 is True and upscale is False. Save uint8 as uint16 without scaling.
        logger.warning(
            "Save image as uint16, but the original input image type is not uint16")
        img = np.asarray(img, np.uint16)

    return img


def _imread_before(grayscale, num_channels):
    # pre-process for imread. check arguments.
    if num_channels not in [-1, 0, 1, 3, 4]:
        raise ValueError("Currently num_channels supports [-1, 0, 1, 3, 4].")

    if not grayscale and num_channels in [0, 1]:
        raise ValueError("If grayscale=False, num_channels must be [-1, 3, 4]."
                         "If you want to get image as gray-scale, try grayscale=True ")


def _imread_after(img, size, interpolate, channel_first, imresize):
    # after-process for imread. resize and transpose img.
    if size is not None:
        img = imresize(img, size, interpolate=interpolate)

    if channel_first and len(img.shape) == 3:
        img = img.transpose((2, 0, 1))

    return img


def _imsave_before(img, channel_first, auto_scale):
    if not isinstance(img, np.ndarray):
        raise ValueError("the input img for imsave must be numpy.ndarray.")

    if len(img.shape) not in [2, 3]:
        raise ValueError(
            "Invalid dimension size of input image. (dims: {})".format(len(img.shape)))

    if img.dtype not in [np.uint8, np.uint16]:
        # consider dtype is float.
        if not auto_scale:
            raise ValueError("If you want to save image with float values, "
                             "'auto_scale' must be True.")

        if np.any(img < 0) or np.any(img > 1):
            raise ValueError("Except for uint8 or uint16, "
                             "all the pixel values of input must be in the range of [0, 1]")

        img = np.asarray(img, np.float32)

    if channel_first and len(img.shape) == 3:
        img = img.transpose((1, 2, 0))

    if len(img.shape) == 3 and img.shape[-1] not in [1, 2, 3, 4]:
        raise ValueError(
            f"Invalid channel size of input image. (channel size: {img.shape[-1]})")

    return img


def _imresize_before(img, size, channel_first, interpolate, interpolations_map):
    if not isinstance(img, np.ndarray):
        raise ValueError("the input img for imresize must be numpy.ndarray.")

    if not isinstance(size, (list, tuple)):
        raise ValueError("size must be list or tuple")

    if len(img.shape) not in [2, 3]:
        raise ValueError(
            "Invalid dimension size of input image. (dims: {})".format(len(img.shape)))

    if interpolate not in interpolations_map:
        raise ValueError(
            "unknown interpolation type."
            " In this backend, you can use only one of [{}]".format(", ".join(interpolations_map)))

    if img.dtype not in [np.uint8, np.uint16]:
        img = np.asarray(img, np.float32)

    if channel_first and len(img.shape) == 3:
        img = img.transpose((1, 2, 0))

    if len(img.shape) == 3 and np.prod(img.shape[:-1]) == 1:
        cur_dtype = img.dtype
        img = (img * np.ones((2, 2, 1))).astype(cur_dtype)

    return img


def _imresize_after(img, channel_first):
    if channel_first and len(img.shape) == 3:
        img = img.transpose((2, 0, 1))

    return img
