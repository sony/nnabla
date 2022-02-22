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

from __future__ import absolute_import, division

import cv2
import numpy as np
from nnabla.logger import logger

from .common import upscale_pixel_intensity, check_type_and_cast_if_necessary, \
    _imread_before, _imread_after, _imsave_before, _imresize_before, _imresize_after
from .image_utils_backend import ImageUtilsBackend


class Cv2Backend(ImageUtilsBackend):
    _interpolations_map = {
        "nearest": cv2.INTER_NEAREST,
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4,
    }

    def __init__(self):
        ImageUtilsBackend.__init__(self)

    @staticmethod
    def _imread_helper(path, r_mode):
        if hasattr(path, "read"):
            img = cv2.imdecode(np.asarray(bytearray(path.read())), r_mode)
        else:
            img = cv2.imread(path, r_mode)

        return img

    @staticmethod
    def convert_channel_from_gray(img, num_channels):
        if num_channels in [-1, 0]:
            return img

        elif num_channels == 1:
            return img[..., np.newaxis]

        elif num_channels == 3:  # GRAY => RGB (just expand)
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        elif num_channels == 4:  # GRAY => RGBA (just expand)
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGBA)

        raise ValueError("num_channels must be [-1, 0, 1, 3, 4]")

    @staticmethod
    def convert_channel_from_bgr(img, num_channels):
        if num_channels in [0, 1]:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if num_channels == 1:
                img = img[..., np.newaxis]

            return img

        elif num_channels in [-1, 3]:  # BGR => RGB
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB, dst=img)

        elif num_channels == 4:  # BGR => RGBA
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)

        raise ValueError("num_channels must be [-1, 0, 1, 3, 4]")

    @staticmethod
    def convert_channel_from_bgra(img, num_channels):
        if num_channels in [0, 1]:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            if num_channels == 1:
                img = img[..., np.newaxis]

            return img

        elif num_channels == 3:  # BGRA => RGB
            return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

        elif num_channels in [-1, 4]:  # BGRA => RGBA
            return cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA, dst=img)

        raise ValueError("num_channels must be [-1, 0, 1, 3, 4]")

    @staticmethod
    def _cvtColor_helper(img, num_channels):
        # convert num_channels which means (h, w, c) -> (h, w, num_channels)
        if len(img.shape) == 2:
            return Cv2Backend.convert_channel_from_gray(img, num_channels)

        elif img.shape[-1] == 3:
            return Cv2Backend.convert_channel_from_bgr(img, num_channels)

        elif img.shape[-1] == 4:
            return Cv2Backend.convert_channel_from_bgra(img, num_channels)

        raise ValueError("Bad image shape. ({})".format(img.shape))

    def accept(self, path, ext, operator):
        if operator in ['resize', 'save']:
            return "OK"
        else:
            if ext in ['.bmp', '.dib', '.ppm', '.pbm', '.pgm', '.sr', '.ras']:
                return "Recommended"
            elif ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
                return "OK"
            else:
                return "NG"

    def imread(self, path, grayscale=False, size=None, interpolate="bilinear",
               channel_first=False, as_uint16=False, num_channels=-1):
        """
        Read image by cv2 module.

        Args:
            path (str or 'file object'): File path or object to read.
            grayscale (bool):
            size (tupple of int):
                (width, height).
                If None, output img shape depends on the files to read.
            channel_first (bool):
                This argument specifies the shape of img is whether (height, width, channel) or (channel, height, width).
                Default value is False, which means the img shape is (height, width, channel).
            interpolate (str):
                must be one of ["nearest", "bilinear", "bicubic", "lanczos"].
            as_uint16 (bool):
                If True, this function reads image as uint16.
            num_channels (int):
                channel size of output array.
                Default is -1 which preserves raw image shape.

        Returns:
            numpy.ndarray
    """

        _imread_before(grayscale, num_channels)

        r_mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_UNCHANGED
        img = self._imread_helper(path, r_mode)

        if as_uint16 and img.dtype != np.uint16:
            if img.dtype == np.uint8:
                logger.warning("You want to read image as uint16, but the original bit-depth is 8 bit."
                               "All pixel values are simply increased by 256 times.")
                img = img.astype(np.uint16) * 256
            else:
                logger.warning(
                    "casting {} to uint16 is not safe.".format(img.dtype))
                return self.next_available(path).imread(path, grayscale=grayscale, size=size, interpolate=interpolate,
                                                        channel_first=channel_first, as_uint16=as_uint16, num_channels=num_channels)
        try:
            img = self._cvtColor_helper(img, num_channels)
        except:
            return self.next_available(path).imread(path, grayscale=grayscale, size=size, interpolate=interpolate,
                                                    channel_first=channel_first, as_uint16=as_uint16, num_channels=num_channels)

        img = _imread_after(img, size, interpolate,
                            channel_first, self.imresize)

        return img

    def imsave(self, path, img, channel_first=False, as_uint16=False, auto_scale=True):
        """
        Save image by cv2 module.
        Args:
            path (str): output filename
            img (numpy.ndarray): Image array to save. Image shape is considered as (height, width, channel) by default.
            channel_first:
                This argument specifies the shape of img is whether (height, width, channel) or (channel, height, width).
                Default value is False, which means the img shape is (height, width, channel)
            as_uint16 (bool):
                If True, save image as uint16.
            auto_scale (bool) :
                Whether upscale pixel values or not.
                If you want to save float image, this argument must be True.
                In cv2 backend, all below are supported.
                    - float ([0, 1]) to uint8 ([0, 255])  (if img.dtype==float and auto_scale==True and as_uint16==False)
                    - float to uint16 ([0, 65535]) (if img.dtype==float and auto_scale==True and as_uint16==True)
                    - uint8 to uint16 are supported (if img.dtype==np.uint8 and auto_scale==True and as_uint16==True)
        """

        img = _imsave_before(img, channel_first, auto_scale)

        if auto_scale:
            img = upscale_pixel_intensity(img, as_uint16)

        img = check_type_and_cast_if_necessary(img, as_uint16)

        # revert channel order to opencv`s one.
        if len(img.shape) == 3:
            if img.shape[-1] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif img.shape[-1] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)

        cv2.imwrite(path, img)

    def imresize(self, img, size, interpolate="bilinear", channel_first=False):
        """
        Resize image by cv2 module.
        Args:
            img (numpy.ndarray): Image array to save.
                Image shape is considered as (height, width, channel) for RGB or (height, width) for gray-scale by default.
            size (tupple of int): (width, height).
            channel_first: bool
                This argument specifies the shape of img is whether (height, width, channel) or (channel, height, width).
                Default value is False, which means the img shape is (height, width, channel)
            interpolate (str):
                must be one of ["nearest", "bilinear", "bicubic", "lanczos"]
        Returns:
            numpy.ndarray whose shape is (size[1], size[0], channel) or (size[1], size[0])
        """

        img = _imresize_before(img, size, channel_first,
                               interpolate, list(self._interpolations_map.keys()))

        dtype = img.dtype

        expand_flag = len(img.shape) == 3 and img.shape[-1] == 1
        # if img shape is (h, w, 1),
        # cv2.reshape automatically shrink the last channel and the shape of returned img is (h, w)

        resized = cv2.resize(
            img, size, interpolation=self._interpolations_map[interpolate])

        if expand_flag:
            resized = np.expand_dims(resized, axis=-1)

        return _imresize_after(resized, channel_first).astype(dtype)
