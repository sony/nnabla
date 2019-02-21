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
import random
import numpy as np


def convert_to_rgb(image, label=None):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if label.ndim == 2:
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
    return image, label


def resize(image, desired_size):

    old_size = image.shape[:2]  # old_size is in (height, width) format
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format
    image = cv2.resize(image, (new_size[1], new_size[0]))

    return image


def random_flip(image, label=None):
    # random no from 0-1(exclude 1)
    rand = random.uniform(0, 1)
    prob = 0.5
    if rand <= prob:
        image = cv2.flip(image, 1)
        label = cv2.flip(label, 1)
        if label is not None:
            label = np.expand_dims(label, axis=2)

    return image, label


def random_crop(image, label, crop_h, crop_w):
    h, w, c = image.shape
    if h >= crop_h and w >= crop_w:
        max_offset_h = h - crop_h + 1
        max_offset_w = w - crop_w + 1

        # rand no between 0 to max_offset --> int
        offset_h = int(random.uniform(0, max_offset_h))
        offset_w = int(random.uniform(0, max_offset_w))

        # crop image to reach crop_h begin from offset_h, crop_w begin from offset_w
        crop_img = image[offset_h:crop_h+offset_h, offset_w:crop_w+offset_w]
        crop_lab = label[offset_h:crop_h+offset_h, offset_w:crop_w+offset_w]

    return crop_img, crop_lab


def random_scale(image, label):

    min_scale_factor = 0.5
    max_scale_factor = 2.
    scale_factor_step_size = 0.25

    num_steps = int((max_scale_factor - min_scale_factor) /
                    scale_factor_step_size + 1)
    scale_factors = np.linspace(min_scale_factor, max_scale_factor, num_steps)
    random.shuffle(scale_factors)

    scale = scale_factors[0].astype(np.float32)

    scaled_image = np.zeros(
        (int(image.shape[0]*scale), int(image.shape[1]*scale), image.shape[2]), dtype=np.float32)
    scaled_label = np.zeros(
        (int(label.shape[0]*scale), int(label.shape[1]*scale), label.shape[2]), dtype=np.int32)

    scaled_image = cv2.resize(image, (int(
        image.shape[1]*scale), int(image.shape[0]*scale)), interpolation=cv2.INTER_LINEAR)

    scaled_label = cv2.resize(label, (int(
        label.shape[1]*scale), int(label.shape[0]*scale)), interpolation=cv2.INTER_NEAREST)
    if scaled_label.ndim == 2:
        scaled_label = np.expand_dims(scaled_label, axis=2)

    return scaled_image, scaled_label


def zero_mean_unit_range(image):
    return ((2.0 / 255.0) * image - 1.0)


def cast(image, label=None):
    image = image.astype(np.float32)
    if label is not None:
        label = label.astype(np.int32)
    return image, label


def pad(image, label=None, desired_size=513):

    # mean pixel value
    color = [127.5, 127.5, 127.5]

    old_size = image.shape[:2]  # old_size is in (height, width) format
    ratio = float(desired_size)/max(old_size)

    new_size = tuple([x + max(desired_size-x, 0) for x in old_size])

    # Pad image with mean pixel value
    new_im = cv2.copyMakeBorder(
        image, 0, new_size[0]-old_size[0], 0, new_size[1]-old_size[1], cv2.BORDER_CONSTANT, value=color)
    if label is not None:
        label = cv2.copyMakeBorder(
            label, 0, new_size[0]-old_size[0], 0, new_size[1]-old_size[1], cv2.BORDER_CONSTANT, value=[255, 255, 255])
        if label.ndim == 2:
            label = np.expand_dims(label, axis=2)

    return new_im, label


def create_mask(label):
    # cretae masks of 1, 0 for ignore labels
    mask = (label != 255)
    mask = mask.astype(np.int32)
    label[label == 255] = 0
    label = label.astype(np.int32)
    return label, mask


def preprocess_image_and_label(image, label=None, target_width=513, train=True):
    '''
    image - 4d array (batch, ch, h, w)
    label - 4d array (batch, ch, h, w)

    '''

    crop_w = target_width
    crop_h = target_width

    if train == True:
        image, label = cast(image, label)

        image, label = random_scale(image, label)

        image, label = pad(image, label, desired_size=target_width)

        image, label = random_crop(image, label, target_width, target_width)

        image, label = random_flip(image, label)

        image = zero_mean_unit_range(image)
        if label is not None:
            label, mask = create_mask(label)

        return image, label, mask

    else:
        image, label = cast(image)
        image = resize(image, target_width)
        image, label = pad(image, desired_size=crop_w)
        image = zero_mean_unit_range(image)
        return image
