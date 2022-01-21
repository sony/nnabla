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
import png
from nnabla.logger import logger

from .common import upscale_pixel_intensity, check_type_and_cast_if_necessary, \
    _imread_before, _imread_after, _imsave_before
from .image_utils_backend import ImageUtilsBackend


class PngBackend(ImageUtilsBackend):
    def __init__(self):
        ImageUtilsBackend.__init__(self)

    @staticmethod
    def rgb2gray(arr):
        # return shape is 2d of (y, x)
        return np.dot(arr[..., :3], [0.299, 0.587, 0.114])

    @staticmethod
    def convert_num_channles(img, num_channels):
        src_channles = 0 if len(img.shape) == 2 else img.shape[-1]
        if src_channles == num_channels or num_channels == -1:
            return img

        if src_channles == 0:
            return np.broadcast_to(img[..., np.newaxis], img.shape + (num_channels,))

        if num_channels == 3:
            return img[..., :3]

        if num_channels == 4:
            fill = 65535 if img.dtype == np.uint16 else 255
            alpha = np.ones(img.shape[:-1] + (1,)).astype(img.dtype) * fill
            return np.concatenate((img, alpha), axis=-1)

        raise ValueError("invalid number of channels")

    @staticmethod
    def read_result_to_ndarray(pixels, width, height, metadata, grayscale, as_uint16, num_channels):
        if metadata["bitdepth"] == 16 and not as_uint16:
            raise ValueError("cannot convert 16bit image to 8bit image."
                             " Original range of pixel values is unknown.")

        output_type = np.uint16 if as_uint16 else np.uint8
        img = np.asarray(list(pixels), output_type)

        if metadata["bitdepth"] == 8 and as_uint16:
            logger.warning("You want to read image as uint16, but the original bit-depth is 8 bit."
                           "All pixel values are simply increased by 256 times.")
            img *= 256

        # shape is (height, width * planes), planes = 1 (gray), 3 (rgb) or 4 (rgba)

        if not metadata["greyscale"]:
            # read image is rgb or rgba
            img = img.reshape((height, width, -1))

            if grayscale:
                img = PngBackend.rgb2gray(img).astype(output_type)

        img = PngBackend.convert_num_channles(img, num_channels)

        return img

    def accept(self, path, ext, operator):
        if operator == "resize":
            return 'NG'
        elif operator == "save":
            return 'OK'
        else:
            if ext == '.png':
                f = path if hasattr(path, "read") else open(path, "rb")

                r = png.Reader(file=f)
                width, height, pixels, metadata = r.asDirect()
                f.seek(0)
                bit_depth = metadata.get("bitdepth")

                if bit_depth not in [8, 16]:
                    return "NG"
                else:
                    return "Recommended"
            else:
                return "NG"

    def imread(self, path, grayscale=False, size=None, interpolate="bilinear",
               channel_first=False, as_uint16=False, num_channels=-1):
        """
        Read image by pypng module.

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
                If True, this function reads image as uint16.
            num_channels (int):
                channel size of output array.
                Default is -1 which preserves raw image shape.

        Returns:
            numpy.ndarray
        """

        _imread_before(grayscale, num_channels)

        f = path if hasattr(path, "read") else open(path, "rb")

        r = png.Reader(file=f)
        width, height, pixels, metadata = r.asDirect()

        bit_depth = metadata.get("bitdepth")

        if bit_depth not in [8, 16]:
            logger.warning("The bit-depth of the image you want to read is unsupported ({}bit)."
                           "Currently, pypng backend`s imread supports only [8, 16] bit-depth."
                           "the path for this image is {}".format(bit_depth, path))
            return self.next_available(path).imread(path, grayscale=grayscale, size=size, interpolate=interpolate,
                                                    channel_first=channel_first, as_uint16=as_uint16, num_channels=num_channels)

        try:
            img = self.read_result_to_ndarray(
                pixels, width, height, metadata, grayscale, as_uint16, num_channels)
        except:
            return self.next_available(path).imread(path, grayscale=grayscale, size=size, interpolate=interpolate,
                                                    channel_first=channel_first, as_uint16=as_uint16, num_channels=num_channels)

        return _imread_after(img, size, interpolate, channel_first, self.imresize)

    def imsave(self, path, img, channel_first=False, as_uint16=False, auto_scale=True):
        """
        Save image by pypng module.

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
                In pypng backend, all below are supported.
                    - float ([0, 1]) to uint8 ([0, 255])  (if img.dtype==float and upscale==True and as_uint16==False)
                    - float to uint16 ([0, 65535]) (if img.dtype==float and upscale==True and as_uint16==True)
                    - uint8 to uint16 are supported (if img.dtype==np.uint8 and upscale==True and as_uint16==True)
        """

        img = _imsave_before(img, channel_first, auto_scale)

        if auto_scale:
            img = upscale_pixel_intensity(img, as_uint16)

        img = check_type_and_cast_if_necessary(img, as_uint16)

        bitdepth = 8 if img.dtype == np.uint8 else 16
        grayscale = True if len(img.shape) == 2 or (
            len(img.shape) == 3 and img.shape[-1] == 1) else False

        writer = png.Writer(img.shape[1], img.shape[0],
                            greyscale=grayscale, bitdepth=bitdepth)

        writer.write(open(path, "wb"), img.reshape(img.shape[0], -1))
