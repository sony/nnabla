# Copyright 2019,2020,2021 Sony Corporation.
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
from nnabla.utils.image_utils import imresize


def resize(image, desired_size):
    old_size = image.shape[:2]  # old_size is in (height, width) format
    ratio = min(np.divide(desired_size, old_size))
    new_size = (int(old_size[0]*ratio), int(old_size[1]*ratio))

    # new_size should be in (width, height) format
    if image.shape[2] == 1:
        image = imresize(
            image, (new_size[1], new_size[0]), interpolate='nearest')
        return image
    image = imresize(image, (new_size[1], new_size[0]))

    return image


def zero_mean_unit_range(image):
    return ((2.0 / 255.0) * image - 1.0)


def cast(image):
    image = image.astype(np.float32)
    return image


def pad(image, desired_size=(513, 513), value=None):

    # mean pixel value
    if value == None:
        value = 127.5
    old_size = image.shape[:2]  # old_size is in (height, width) format

    new_size = tuple([x + max(ds - x, 0)
                      for x, ds in zip(old_size, desired_size)])

    # Pad image with mean pixel value
    new_im = np.pad(image, ((0, (new_size[0]-old_size[0])), (0, new_size[1] -
                                                             old_size[1]), (0, 0)), mode='constant', constant_values=value)
    return new_im


def preprocess_image_and_label(image, target_width=513, target_height=513):
    '''
    image - 4d array (batch, ch, h, w)
    label - 4d array (batch, ch, h, w)

    '''
    crop_w = target_width
    crop_h = target_height
    image = cast(image)
    image = resize(image, (target_height, target_width))
    image = pad(image, desired_size=(crop_h, crop_w), value=None)
    image = zero_mean_unit_range(image)
    return image
