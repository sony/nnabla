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
import nnabla as nn
from nnabla.utils.data_iterator import data_iterator_simple
import numpy as np
import random


def read_image(img):
    img = cv2.imread(img, cv2.IMREAD_UNCHANGED)
    img = img.astype(np.float32) / 255.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img


def channel_convert(channel, img, color):
    if channel == 1 and color == 'RGB':  # gray/y to BGR
        return [cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)]
    else:
        return img


def modcrop(img_in, scale):
    # img_in: Numpy, HWC or HW
    img = np.copy(img_in)
    if img.ndim == 2:
        H, W = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r]
    elif img.ndim == 3:
        H, W, C = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r, :]
    else:
        raise ValueError('Wrong img ndim: [{:d}].'.format(img.ndim))
    return img


def augment(img, hflip, rot90, vflip):
    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    return _augment(img)


def data_iterator_sr(num_examples, batch_size, gt_image, lq_image, train, shuffle, rng=None):
    from args import get_config
    conf = get_config()

    def dataset_load_func(i):
        # get images from the list
        scale = conf.train.scale
        gt_size = conf.train.gt_size
        gt_img = read_image(gt_image[i])
        lq_img = read_image(lq_image[i])
        if not train:
            gt_img = modcrop(gt_img, scale)
        gt_img = channel_convert(gt_img.shape[2], gt_img, color="RGB")
        if train:
            # randomly crop
            H, W, C = lq_img.shape
            lq_size = gt_size//scale
            rnd_h = random.randint(0, max(0, H - lq_size))
            rnd_w = random.randint(0, max(0, W - lq_size))
            lq_img = lq_img[rnd_h:rnd_h + lq_size, rnd_w:rnd_w + lq_size, :]
            rnd_h_gt, rnd_w_gt = int(rnd_h * scale), int(rnd_w * scale)
            gt_img = gt_img[rnd_h_gt:rnd_h_gt +
                            gt_size, rnd_w_gt:rnd_w_gt + gt_size, :]
            # horizontal and vertical flipping and rotation
            hflip, rot = [True, True]
            hflip = hflip and random.random() < 0.5
            vflip = rot and random.random() < 0.5
            rot90 = rot and random.random() < 0.5
            lq_img = augment(lq_img, hflip, rot90, vflip)
            gt_img = augment(gt_img, hflip, rot90, vflip)
            lq_img = channel_convert(C, [lq_img], color="RGB")[0]
        # BGR to RGB and HWC to CHW
        if gt_img.shape[2] == 3:
            gt_img = gt_img[:, :, [2, 1, 0]]
            lq_img = lq_img[:, :, [2, 1, 0]]

        gt_img = np.ascontiguousarray(np.transpose(gt_img, (2, 0, 1)))
        lq_img = np.ascontiguousarray(np.transpose(lq_img, (2, 0, 1)))
        return gt_img, lq_img
    return data_iterator_simple(dataset_load_func, num_examples, batch_size, shuffle=shuffle, rng=rng,
                                with_file_cache=False, with_memory_cache=False)
