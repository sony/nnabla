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

from __future__ import division, absolute_import
import warnings

import numpy as np
import skimage
import skimage.io
import skimage.transform

from .common import rescale_pixel_intensity

try:
    import png
except ImportError:
    png = None


def imread_skimage(path, grayscale=False, size=None, interpolate="gaussian"):
    img = skimage.io.imread(path, as_gray=grayscale)

    if size is not None:
        img = imresize(img, size, interpolate)

    if grayscale:  # pivel values are [0, 1]
        img = rescale_pixel_intensity(
            img, input_low=0, input_high=1, output_low=0, output_high=255, output_type=np.uint8)

    return img


def imread_pypng(path, size=None, interpolate="gaussian"):
    f = path if hasattr(path, "read") else open(path, "rb")

    r = png.Reader(file=f)
    width, height, pixels, metadata = r.asRGB8()
    img = np.array(list(pixels)).reshape((height, width, -1))

    if size is not None:
        out_height, out_width = size
        img = imresize(img, (out_height, out_width), interpolate)

    return img


def imread(path, grayscale=False, size=None, interpolate="gaussian", channel_first=False):
    """

    :param path: str of filename or file object
    :param grayscale: bool
    :param size: tupple of int (width, height)
        if None, resize is not done and img shape depends on the files.
    :param interpolate: "gaussian" or None
        currently skimage support only gaussian interpolation.
        if None, no interpolation is done. This is used for resize.
    :param channel_first:
        This argument specifies the shape of img is whether (height, width, channel) or (channel, height, width).
        Default value is False, which means the img shape is (height, width, channel)
    :return: numpy.ndarray
    """
    if grayscale or png is None:
        img = imread_skimage(path, grayscale, size, interpolate)
    else:
        img = imread_pypng(path, size, interpolate)

    return img.transpose((2, 0, 1)) if channel_first and len(img.shape) == 3 else img


def imsave(path, img, channel_first=False):
    """

    :param path: str, output filename
    :param img: numpy.ndarray whose shape is (channel, height, width)
    :param channel_first:
        This argument specifies the shape of img is whether (height, width, channel) or (channel, height, width).
        Default value is False, which means the img shape is (height, width, channel)
    """

    if channel_first and len(img.shape) == 3:
        img = img.transpose((1, 2, 0))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        skimage.io.imsave(path, img)


def imresize(img, size, interpolate="gaussian", channel_first=False):
    """

    :param img: numpy.ndarray whose shape is (height, width, channel) or (height, width)
    :param size: tupple of int (width, height).
    :param interpolate: "gaussian" or None
        currently skimage support only gaussian interpolation.
        if None, no interpolation is done.
    :param channel_first: bool
        This argument specifies the shape of img is whether (height, width, channel) or (channel, height, width).
        Default value is False, which means the img shape is (height, width, channel)
    :return: numpy.ndarray whose shape is (size[1], size[0], channel) or (size[1], size[0])
    """
    # cv2.resize required (h, w, c) as size of image and (h, w) as size

    if interpolate is not None and interpolate != "gaussian":
        raise ValueError("Currently, skimage support only gaussian interpolation."
                         " if you want to resize img without any interpolation, specify None to 'interpolate'")

    if channel_first and len(img.shape) == 3:
        img = img.transpose((1, 2, 0))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if skimage.__version__ >= "0.14":
            resized = skimage.transform.resize(img, (size[1], size[0]),
                                               preserve_range=img.dtype == np.uint8,
                                               anti_aliasing=interpolate == "gaussian")
        else:
            resized = skimage.transform.resize(img, (size[1], size[0]),
                                               preserve_range=img.dtype == np.uint8)

    if img.dtype == np.uint8:
        resized = resized.astype(np.uint8)

    if channel_first and len(img.shape) == 3:
        resized = resized.transpose((2, 0, 1))

    return resized
