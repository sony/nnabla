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

import numpy as np
from PIL import Image
from nnabla.logger import logger

from .common import _imread_before, _imread_after, _imsave_before, _imresize_before, _imresize_after
from .image_utils_backend import ImageUtilsBackend


class PilBackend(ImageUtilsBackend):
    _interpolations_map = {
        "nearest": Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
    }

    def __init__(self):
        ImageUtilsBackend.__init__(self)
        if hasattr(Image.Resampling, "HAMMING"):  # version >3.4.0
            self._interpolations_map["hamming"] = Image.Resampling.HAMMING

        if hasattr(Image.Resampling, "BOX"):  # version >3.4.0
            self._interpolations_map["box"] = Image.Resampling.BOX

        if hasattr(Image.Resampling, "LANCZOS"):  # version >1.1.3
            self._interpolations_map["lanczos"] = Image.Resampling.LANCZOS

    @staticmethod
    def convert_pil(pil_image, grayscale, num_channels, return_palette_indices):
        if pil_image.mode == "I":
            raise ValueError(
                "Input img type seems int32. Currently we don`t support int32 image in pillow backend.")

        # Note:
        #   This code block below is copied from
        #   https://github.com/scipy/scipy/blob/maintenance/1.2.x/scipy/misc/pilutil.py.
        if pil_image.mode == 'P' and not return_palette_indices:
            # Mode 'P' means there is an indexed "palette".  If we leave the mode
            # as 'P', then when we do `a = array(im)` below, `a` will be a 2-D
            # containing the indices into the palette, and not a 3-D array
            # containing the RGB or RGBA values.
            if 'transparency' in pil_image.info:
                pil_image = pil_image.convert('RGBA')
            else:
                pil_image = pil_image.convert('RGB')

        if grayscale:
            ret = np.asarray(pil_image.convert("L"))

            if num_channels > 0:
                ret = np.broadcast_to(
                    ret[..., np.newaxis], ret.shape + (num_channels,))

            return ret
        elif num_channels == 3:
            return pil_image.convert("RGB")

        elif num_channels == 4:
            return pil_image.convert("RGBA")

        return pil_image

    @staticmethod
    def pil_image_to_ndarray(pil_image, grayscale, num_channels, return_palette_indices):
        ret = PilBackend.convert_pil(pil_image, grayscale, num_channels,
                                     return_palette_indices)
        return np.asarray(ret).astype(np.uint8)

    @staticmethod
    def pil_resize_from_ndarray(arr, size, resample):
        mode = "F" if arr.dtype == np.float32 else None

        pil_image = Image.fromarray(arr, mode=mode)
        resized_image = pil_image.resize(size, resample=resample)

        return np.asarray(resized_image)

    def accept(self, path, ext, operator):
        if operator in ['resize', 'save']:
            return "OK"
        else:
            if ext in ['.bmp', '.dib', '.eps', '.gif', '.icns', '.ico', '.jpeg', '.jpg', '.msp', '.png', '.ppm', '.pbm', '.pgm', '.pnm', '.tif', '.tiff']:
                return "OK"
            else:
                return "NG"

    def imread(self, path, grayscale=False, size=None, interpolate="bilinear",
               channel_first=False, as_uint16=False, num_channels=-1, return_palette_indices=False):
        """
        Read image by PIL module.
        Notice that PIL only supports uint8 for RGB (not uint16).
        So this imread function returns only uint8 array for both RGB and gray-scale.
        (Currently ignore "I" mode for gray-scale (32bit integer).)

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
                must be one of ["nearest", "box", "bilinear", "hamming", "bicubic", "lanczos"].
            as_uint16 (bool):
                If you specify this argument, you can use only False for pil backend.
            num_channels (int):
                channel size of output array.
                Default is -1 which preserves raw image shape.
            return_palette_indices (bool):
                Whether to return a raw palette indices without any conversion or not.
                If this flag is True and read Image has the mode "P",
                then this function returns 2-D array containing the indices into palette.
                We recommend that this flag should be False unless you intend to use the raw palette indices.

        Returns:
            numpy.ndarray
        """

        if as_uint16:
            logger.warning("pillow only supports uint8 for RGB image."
                           " If you want to load image as uint16,"
                           " install pypng or cv2 and"
                           " nnabla.utils.image_utils automatically change backend to use these module.")
            return self.next_available(path).imread(path, grayscale=grayscale, size=size, interpolate=interpolate,
                                                    channel_first=channel_first, as_uint16=as_uint16, num_channels=num_channels)

        _imread_before(grayscale, num_channels)

        pil_img = Image.open(path, mode="r")

        try:
            img = self.pil_image_to_ndarray(
                pil_img, grayscale, num_channels, return_palette_indices)
        except:
            return self.next_available(path).imread(path, grayscale=grayscale, size=size, interpolate=interpolate,
                                                    channel_first=channel_first, as_uint16=as_uint16, num_channels=num_channels)

        return _imread_after(img, size, interpolate, channel_first, self.imresize)

    def imsave(self, path, img, channel_first=False, as_uint16=False, auto_scale=True):
        """
        Save image by pillow module.
        Currently, pillow supports only uint8 to save.

        Args:
            path (str): output filename
            img (numpy.ndarray): Image array to save. Image shape is considered as (height, width, channel) by default.
            channel_first (bool):
                This argument specifies the shape of img is whether (height, width, channel) or (channel, height, width).
                Default value is False, which means the img shape is considered as (height, width, channel)
            as_uint16 (bool):
                In this backend, this argument is always False because pillow dose not support uint16.
                If True, exception will be raised.
            auto_scale (bool) :
                Whether upscale pixel values or not.
                If you want to save float image, this argument must be True.
                In pillow backend, only float ([0, 1]) to uint8 ([0, 255]) is supported.
        """
        img = _imsave_before(img, channel_first, auto_scale)

        if img.dtype == np.uint16 or as_uint16:
            logger.warning("Pillow only supports uint8 image to save. Cast img to uint8."
                           "If you want to save image as uint16, install pypng or cv2 "
                           "and nnabla.utils.image_utils automatically change backend to use these module.")
            return self.next_available(path).imsave(path, img, channel_first=channel_first, as_uint16=as_uint16, auto_scale=auto_scale)

        if auto_scale and img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)

        if len(img.shape) == 3 and img.shape[-1] == 1:
            img = np.squeeze(img, axis=-1)

        Image.fromarray(img).save(path)

    def imresize(self, img, size, interpolate="bilinear", channel_first=False):
        """
        Resize image by pil module.
        Args:
            img (numpy.ndarray): Image array to save.
                Image shape is considered as (height, width, channel) for RGB or (height, width) for gray-scale by default.
            size (tupple of int): (width, height).
            channel_first (bool):
                This argument specifies the shape of img is whether (height, width, channel) or (channel, height, width).
                Default value isyou can get the array whose shape is False, which means the img shape is (height, width, channels)
            interpolate (str):
                must be one of ["nearest", "box", "bilinear", "hamming", "bicubic", "lanczos"]
        Returns:
            numpy.ndarray whose shape is ('size'[1], 'size'[0], channel) or (size[1], size[0])
        """

        img = _imresize_before(img, size, channel_first,
                               interpolate, list(self._interpolations_map.keys()))

        expand_flag = False
        if len(img.shape) == 3 and img.shape[-1] == 1:
            # (h, w, 1) can not be handled by pil.Image, temporally reshape to (h, w)
            img = img.reshape(img.shape[0], img.shape[1])
            expand_flag = True

        resample = self._interpolations_map[interpolate]

        if img.dtype == np.uint8:
            resized = self.pil_resize_from_ndarray(img, size, resample)
        else:
            dtype = img.dtype
            img_float32 = np.asarray(img, np.float32)
            if len(img.shape) == 3:
                resized = np.stack([self.pil_resize_from_ndarray(img_float32[..., i], size, resample)
                                    for i in range(img.shape[-1])], axis=2)
            else:
                resized = self.pil_resize_from_ndarray(
                    img_float32, size, resample)

            resized = np.asarray(resized, dtype)

        if expand_flag:
            resized = resized[..., np.newaxis]

        return _imresize_after(resized, channel_first)
