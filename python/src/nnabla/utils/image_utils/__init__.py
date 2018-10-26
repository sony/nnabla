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

from nnabla.logger import logger

# check if cv2 module exists
try:
    from . import cv2_utils
except ImportError:
    cv2_utils = None

# check if skimage module exists
try:
    from . import skimage_utils
except ImportError:
    skimage_utils = None

# check if imageio and pillow exist
try:
    from . import pil_utils
except ImportError:
    raise ImportError(
        "imageio and pillow must be installed. Check your environment.")

if cv2_utils is not None:
    module = cv2_utils
    backend = "cv2"

elif skimage_utils is not None:
    module = skimage_utils
    backend = "skimage"

else:
    module = pil_utils
    backend = "pillow"


def imread(path, grayscale=False, size=None, interpolate="bilinear", channel_first=False):
    """


    Read image from ``path``.
    If you specify the ``size``, the output array is resized.
    Default output shape is (height, width, channel) for RGB image and (height, width) for gray-scale image.

    Args:
        path (String or File Object): Input image path.
        grayscale (bool): If True, the img is rescaled to gray-scale. Default is False.
        size (tuple of int): Output shape. The order is (width, height). If None, the image is not resized. Default is None.
        channel_first (bool): If True, the shape of the output array is (channel, height, width) for RGB image. Default is False.

    Returns:
         numpy.ndarray
    """

    return module.imread(path, grayscale=grayscale, size=size, interpolate=interpolate, channel_first=channel_first)


def imsave(path, img, channel_first=False):
    """

    Save ``img`` to the file specified by ``path``.
    As default, the shape of ``img`` has to be (height, width, channel).

    Args:
        path (str): Output path.
        img (numpy.ndarray): Input image.
        channel_first (bool): If True, you can input the image whose shape is (channel, height, width).
    """

    module.imsave(path, img, channel_first=channel_first)


def imresize(img, size, interpolate="bilinear", channel_first=False):
    """
    Resize ``img`` to ``size``.
    As default, the shape of input image has to be (height, width, channel).

    Args:
        img (numpy.ndarray): Input image.
        size (tuple of int): Output shape. The order is (width, height).
        channel_first (bool): If True, the shape of the output array is (channel, height, width) for RGB image. Default is False.
    Returns:
         numpy.ndarray
    """

    return module.imresize(img, size, interpolate=interpolate, channel_first=channel_first)


# alias
imwrite = imsave
imload = imread

logger.info("use {} for the backend of image utils".format(backend))
