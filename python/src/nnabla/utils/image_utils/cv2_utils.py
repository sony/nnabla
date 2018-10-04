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


import cv2
import numpy as np

interpolations_map = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "lanczos": cv2.INTER_LANCZOS4,
}


def imread(path, grayscale=False, size=None, interpolate="bilinear", channel_first=False):
    """

    :param path: str of filename or file object
    :param grayscale: bool
    :param size: tupple of int (width, height)
        if None, resize is not done and img shape depends on the files.
    :param interpolate: str
        must be one of ["nearest", "bilinear", "bicubic", "lanczos"]
    :param channel_first:
        This argument specifies the shape of img is whether (height, width, channel) or (channel, height, width).
        Default value is False, which means the img shape is (height, width, channel)
    :return: numpy.ndarray
    """
    if hasattr(path, "read"):
        path.seek(0)
        img = cv2.imdecode(np.asarray(bytearray(path.read()),
                                      dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    else:
        img = cv2.imread(path)

    if grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif len(img.shape) == 3:
        img = img[:, :, ::-1]  # BGR -> RGB

    if size is not None:
        img = imresize(img, size, interpolate)

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

    if channel_first and len(img.shape) == 3:
        img = img.transpose(1, 2, 0)

    if len(img.shape) == 3:
        img = img[..., ::-1]  # RGB -> BGR

    cv2.imwrite(path, img)


def imresize(img, size, interpolate="bilinear", channel_first=False):
    """

    :param img: numpy.ndarray whose shape is (height, width, channel) or (height, width)
    :param size: tupple of int (width, height).
    :param interpolate: "gaussian" or None
        must be one of ["nearest", "bilinear", "bicubic", "lanczos"]
    :param channel_first: bool
        This argument specifies the shape of img is whether (height, width, channel) or (channel, height, width).
        Default value is False, which means the img shape is (height, width, channel)
    :return: numpy.ndarray whose shape is (size[1], size[0], channel) or (size[1], size[0])
    """
    if interpolate not in interpolations_map.keys():
        raise ValueError(
            "unknown interpolation type. must be [{}]".format(", ".join(list(interpolations_map.keys()))))
    else:
        interpolation = interpolations_map[interpolate]

    if img.dtype is not np.uint8:
        img = img.astype(float)

    if channel_first and len(img.shape) == 3:
        img = img.transpose((1, 2, 0))

    resized = cv2.resize(img, size, interpolation=interpolation)

    if channel_first and len(img.shape) == 3:
        resized = resized.transpose((2, 0, 1))

    return resized
