# Copyright 2018,2019,2020,2021 Sony Corporation.
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

from .backend_manager import backend_manager
from .backend_events.common import rescale_pixel_intensity


def set_backend(backend):
    """
    Set utils.image_utils` backend module.
    If the backend which you specify is not installed in your environment, Exception is raised.

    Args:
        backend (str): the name of image_utils` backend
    """

    backend_manager.set_backend(backend)


def get_backend():
    """
    Get image_utils` backend module used now.

    Returns:
        str
    """

    return backend_manager.get_backend()


def get_available_backends():
    """
    Get all available image_utils` backend modules.

    Returns:
        list of str
    """

    return backend_manager.get_available_backends()


def minmax_auto_scale(img, as_uint16):
    """
    Utility function for rescaling all pixel values of input image to fit the range of uint8.
    Rescaling method is min-max, which is all pixel values are normalized to [0, 1] by using img.min() and img.max()
    and then are scaled up by 255 times.
    If the argument `as_uint16` is True, output image dtype is np.uint16 and the range of pixel values is [0, 65535] (scaled up by 65535 after normalized to [0, 1]).

    :param img (numpy.ndarray): input image.
    :param as_uint16: If True, output image dtype is uint16.
    :return: numpy.ndarray
    """

    if as_uint16:
        output_high = 65535
        output_type = np.uint16
    else:
        output_high = 255
        output_type = np.uint8

    return rescale_pixel_intensity(img, input_low=img.min(), input_high=img.max(),
                                   output_low=0, output_high=output_high, output_type=output_type)


def imread(path, grayscale=False, size=None, interpolate="bilinear",
           channel_first=False, as_uint16=False, num_channels=-1, **kwargs):
    """
    Read image from ``path``.
    If you specify the ``size``, the output array is resized.
    Default output shape is (height, width, channel) for RGB image and (height, width) for gray-scale image.

    Args:
        path (String or File Object): Input image path.
        grayscale (bool): If True, the img is rescaled to gray-scale. Default is False.
        size (tuple of int): Output shape. The order is (width, height). If None, the image is not resized. Default is None.
        interpolate (str): Interpolation method.
            This argument is depend on the backend.
            If you want to specify this argument, you should pay much attention to which backend you use now.
            What you can select is below:
             - pil backend: ["nearest", "box", "bilinear", "hamming", "bicubic", "lanczos"].
             - cv2 backend: ["nearest", "bilinear", "bicubic", "lanczos"].
            Default is "bilinear" for both backends.
        channel_first (bool): If True, the shape of the output array is (channel, height, width) for RGB image. Default is False.
        as_uint16 (bool): If True, this function tries to read img as np.uint16. Default is False.
        num_channels (int): channel size of output array.
            Default is -1 which preserves raw image shape.
        return_palette_indices (bool):
            This argument can be used only by pil backend.
            On pil backend, if this flag is True and PIL.Image has the mode "P",
            then this function returns 2-D array containing the indices into palette.
            Otherwise, 3-D array of "RGB" or "RGBA" (it depends on an image info) will be returned.
            Default value is False.

    Returns:
         numpy.ndarray :
         if as_uint16=True output dtype is np.uint16, else np.uint8 (default).
    """

    best_backend = backend_manager.get_best_backend(path, "load")
    return best_backend.imread(path, grayscale=grayscale, size=size, interpolate=interpolate,
                               channel_first=channel_first, as_uint16=as_uint16, num_channels=num_channels,
                               **kwargs)


def imsave(path, img, channel_first=False, as_uint16=False, auto_scale=True, **kwargs):
    """
    Save ``img`` to the file specified by ``path``.
    As default, the shape of ``img`` has to be (height, width, channel).

    Args:
        path (str): Output path.
        img (numpy.ndarray):
            Input image.
            All pixel values must be positive and in the range [0, 255] of int for uint8, [0, 65535] of int for uint16 or [0, 1] for float.
            When you pass float image, you must set `auto_scale` as True (If not, exception will be raised).
            If img with negative values is passed as input, exception will be raised.
        channel_first (bool):
            If True, you can input the image whose shape is (channel, height, width). Default is False.
        as_uint16 (bool):
            If True, cast image to uint16 before save. Default is False.
        auto_scale (bool):
            Whether the range of pixel values are scaled up or not.
            The range of upscaled pixel values depends on output dtype, which is [0, 255] as uint8 and [0, 65535] as uint16.
    """

    best_backend = backend_manager.get_best_backend(path, "save")
    best_backend.imsave(
        path, img, channel_first=channel_first, as_uint16=as_uint16, auto_scale=auto_scale, **kwargs)


def imresize(img, size, interpolate="bilinear", channel_first=False, **kwargs):
    """
    Resize ``img`` to ``size``.
    As default, the shape of input image has to be (height, width, channel).

    Args:
        img (numpy.ndarray): Input image.
        size (tuple of int): Output shape. The order is (width, height).
        interpolate (str): Interpolation method.
            This argument is depend on the backend.
            If you want to specify this argument, you should pay much attention to which backend you use now.
            What you can select is below:
             - pil backend: ["nearest", "box", "bilinear", "hamming", "bicubic", "lanczos"].
             - cv2 backend: ["nearest", "bilinear", "bicubic", "lanczos"].
            Default is "bilinear" for both backends.
        channel_first (bool):
            If True, the shape of the output array is (channel, height, width) for RGB image. Default is False.
    Returns:
         numpy.ndarray
    """

    best_backend = backend_manager.get_best_backend(img, "resize")
    return best_backend.imresize(img, size, interpolate=interpolate, channel_first=channel_first, **kwargs)


# alias
imwrite = imsave
imload = imread
