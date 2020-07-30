# Copyright (c) 2020 Sony Corporation. All Rights Reserved.
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

import nnabla as nn
import numpy as np
import nnabla.monitor as nm

from nnabla.utils.image_utils import imread, imsave, imresize


class MonitorManager(object):
    """
        docstring for MonitorManager
        input_dict = {str1: Variable1, str2: Variable2, ... } 
    """

    def __init__(self, key2var_dict, monitor, interval):
        super(MonitorManager, self).__init__()
        self.key2var_dict = key2var_dict
        self.monitor_dict = dict()
        for k, v in self.key2var_dict.items():
            assert isinstance(v, nn.Variable), "invalid inputs?"
            self.monitor_dict[k] = nm.MonitorSeries(
                k, monitor, interval=interval)

    def add(self, iteration):
        for k, v in self.monitor_dict.items():
            var = self.key2var_dict[k]
            self.monitor_dict[k].add(iteration, var.d.item())


def combine_images(images):
    """
        src_bod: np.ndarray shape of (B, 15, H, W)
    """

    def postprocess_img(img):
        """
            takes image (either bod_map or bod_img),
            returns Batched 3ch image. (B, C, H, W)
        """

        if img.shape[1] > 3:
            img_for_save = np.clip(np.max(img, axis=1, keepdims=True), 0, 1)
            img_for_save = np.concatenate(
                [img_for_save, img_for_save, img_for_save], axis=1)
        else:
            img_normalized = (img + 1) / 2.  # [-1., +1.] -> [0, 1]
            img_for_save = img_normalized
        return img_for_save

    def force_resize(image, target_shape):
        resized_image = np.zeros(
            image.shape[:2] + target_shape)  # (B, C, H, W)
        for i in range(image.shape[0]):
            resized_image[i] = imresize(
                image[i], target_shape, channel_first=True)
        return resized_image

    batch_size = images[0].shape[0]
    target_height, target_width = images[0].shape[2:]
    preprocessed_images = [postprocess_img(image) for image in images]

    out_image = force_resize(
        preprocessed_images[0], (target_height, target_width))

    for image_to_combine in preprocessed_images[1:]:
        resized_image = force_resize(
            image_to_combine, (target_height, target_width))
        out_image = np.concatenate([out_image, resized_image], axis=3)

    return out_image
