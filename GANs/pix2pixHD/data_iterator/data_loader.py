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

import numpy as np
from nnabla.utils.data_iterator import data_iterator
from nnabla.utils.data_source import DataSource
from nnabla.utils.image_utils import imread, imresize


def load_function(image_path, inst_path, label_path, image_shape):
    # naive image read implementation
    image = imread(image_path, channel_first=True)

    inst_map = imread(inst_path, as_uint16=True)

    label_map = imread(label_path)

    if image.shape[1:] != image_shape:
        # imresize takes (width, height) as shape.
        resize_shape = (image_shape[1], image_shape[0])
        image = imresize(image, resize_shape, channel_first=True)
        inst_map = imresize(inst_map, resize_shape)
        label_map = imresize(label_map, resize_shape)

    # normalize
    image = (image - 127.5) / 127.5  # -> [-1, 1]

    return image, inst_map, label_map


class RawIterator(DataSource):
    def __init__(self, data_list, image_shape=(1024, 2048), shuffle=True, rng=None, flip=True):
        super(RawIterator, self).__init__(shuffle=shuffle, rng=rng)

        self._data_list = data_list  # [[image, inst, label], ...]
        self._image_shape = image_shape
        self._size = len(self._data_list)
        self._variables = ("image", "instance_id", "label_id")
        self.flip = flip

        self.reset()

    def reset(self):
        self._idxs = self._rng.permutation(
            self._size) if self.shuffle else np.arange(self._size)

        super(RawIterator, self).reset()

    def __iter__(self):
        self.reset()
        return self

    def _get_data(self, position):

        i = self._idxs[position]
        image_path, inst_path, label_path = self._data_list[i]

        image, inst_map, label_map = load_function(
            image_path, inst_path, label_path, self._image_shape)

        if self.flip:
            if np.random.rand() > 0.5:
                image = image[..., ::-1]
                inst_map = inst_map[..., ::-1]
                label_map = label_map[..., ::-1]

        return image, inst_map, label_map


def create_data_iterator(batch_size, data_list, image_shape, shuffle=True, rng=None,
                         with_memory_cache=False, with_parallel=False, with_file_cache=False, flip=True):
    return data_iterator(RawIterator(data_list, image_shape, shuffle=shuffle, rng=rng, flip=flip),
                         batch_size,
                         with_memory_cache,
                         with_parallel,
                         with_file_cache)
