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

import sys
import argparse
import csv
import nnabla as nn
from read_pfm import readPFM
from nnabla.utils.data_iterator import data_iterator_simple
from nnabla.utils.image_utils import imread
import numpy as np
import random


def data_iterator(num_examples, batch_size, img_left, img_right, img_disp, train, shuffle, dataset, rng=None):
    def dataset_load_func(i):
        # get images from the list
        image_left = imread(img_left[i]).astype('float32')
        image_right = imread(img_right[i]).astype('float32')
        # print(img_disp)
        if dataset == "SceneFlow":
            from main import parser
            args = parser.parse_args()
            image_disp, scale = readPFM(img_disp[i])
            image_disp = np.ascontiguousarray(image_disp, dtype=np.float32)
        elif dataset == "Kitti":
            from finetune import parser
            args = parser.parse_args()
            image_disp = imread(img_disp[i]).astype('float32')
        image_disp = image_disp.reshape(
            image_disp.shape[0], image_disp.shape[1], 1)

        mean_imagenet = np.asarray([0.485, 0.456, 0.406]).astype(
            np.float32).reshape(3, 1, 1)
        std_imagenet = np.asarray([0.229, 0.224, 0.225]).astype(
            np.float32).reshape(3, 1, 1)

        if train:
            w, h = image_left.shape[1], image_left.shape[0]
            th, tw = args.crop_height, args.crop_width
            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)
            # crop
            image_left = image_left[y1:y1 + th, x1:x1 + tw]
            image_right = image_right[y1:y1 + th, x1:x1 + tw]
            if dataset == "Kitti":
                image_disp = np.ascontiguousarray(
                    image_disp, dtype=np.float32)/256
            image_disp = image_disp[y1:y1 + th, x1:x1 + tw]
            # normalize with mean and std
            image_left, image_right, image_disp = np.rollaxis(
                image_left, 2), np.rollaxis(image_right, 2), np.rollaxis(image_disp, 2)
            image_left = (image_left/255).astype(np.float32)
            image_right = (image_right/255).astype(np.float32)
            image_left -= mean_imagenet
            image_left /= std_imagenet
            image_right -= mean_imagenet
            image_right /= std_imagenet
        else:
            # crop
            if dataset == "SceneFlow":
                image_left = image_left[:args.im_height, :args.im_width, :]
                image_right = image_right[:args.im_height, :args.im_width, :]
                image_disp = image_disp[:args.im_height, :args.im_width, :]
            elif dataset == "Kitti":
                w, h = image_left.shape[1], image_left.shape[0]
                image_left = image_left[h -
                                        args.im_height:h, w-args.im_width:w, :]
                image_right = image_right[h -
                                          args.im_height:h, w-args.im_width:w, :]
                image_disp = image_disp[h -
                                        args.im_height:h, w-args.im_width:w, :]
                image_disp = np.ascontiguousarray(
                    image_disp, dtype=np.float32)/256
            # normalize
            image_left, image_right, image_disp = np.rollaxis(
                image_left, 2), np.rollaxis(image_right, 2), np.rollaxis(image_disp, 2)
            image_left = (image_left/255).astype(np.float32)
            image_right = (image_right/255).astype(np.float32)
            image_left -= mean_imagenet
            image_left /= std_imagenet
            image_right -= mean_imagenet
            image_right /= std_imagenet

        return image_left, image_right, image_disp
    return data_iterator_simple(dataset_load_func, num_examples, batch_size, shuffle=shuffle, rng=rng,
                                with_file_cache=False, with_memory_cache=False)
