# Copyright (c) 2019 Sony Corporation. All Rights Reserved.
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


import os
import numpy as np
from nnabla.utils.image_utils import imread, imresize
import random


def get_data_dict(attr_path, selected_attrs):
    """
    Based on `selected_attrs`,
    create an attribute truth table and attribute dictionaries.
    """
    with open(attr_path, 'r') as f:
        lines = f.readlines()
    lines = [line.rstrip() for line in lines]
    all_attr_names = lines[1].split()
    attr2idx, idx2attr = dict(), dict()
    dataset = list()
    for i, attr_name in enumerate(all_attr_names):
        attr2idx[attr_name] = i
        idx2attr[i] = attr_name

    lines = lines[2:]
    random.seed(1234)
    random.shuffle(lines)
    for i, line in enumerate(lines):
        split = line.split()
        filename = split[0]
        values = split[1:]

        label = []
        for attr_name in selected_attrs:
            idx = attr2idx[attr_name]
            label.append(values[idx] == '1')

        dataset.append([filename, label])

    return dataset, attr2idx, idx2attr


def stargan_load_func(i, dataset, image_dir, image_size, crop_size):
    '''
    Load an image and label from dataset.
    This function assumes that there are two set of domains in the dataset.
    For example, CelebA has 40 attributes.
    Args:
        dataset: a list containing image paths and attribute lists.
        image_dir: path to the directory containing raw images.
        image_size: image size (height and width) after getting resized.
        crop_size: crop size.
    Returns:
        image, label: an image and a label to be fed to nn.Variables.
    '''
    def center_crop_numpy(image, crop_size_h, crop_size_w):
        # naive implementation.
        assert len(image.shape) == 3  # (c, h, w)
        start_h = (image.shape[1] - crop_size_h) // 2
        stop_h = image.shape[1] - start_h
        start_w = (image.shape[2] - crop_size_w) // 2
        stop_w = image.shape[2] - start_w
        cropped_image = image[:, start_h:start_h +
                              crop_size_h, start_w:start_w+crop_size_w]
        return cropped_image

    img_path, label = dataset[i][0], dataset[i][1]
    # Load image and labels.
    # Unlike original implementation, crop and resize are executed here.
    image = imread(os.path.join(image_dir, img_path),
                   num_channels=3, channel_first=True)
    if image.dtype == np.uint8:
        # Clip image's value from [0, 255] -> [0.0, 1.0]
        image = image / 255.0
    image = (image - 0.5) / 0.5  # Normalize.
    image = center_crop_numpy(image, crop_size, crop_size)
    image = imresize(image, (image_size, image_size),
                     interpolate='bilinear', channel_first=True)

    return np.asarray(image), np.asarray(label)
