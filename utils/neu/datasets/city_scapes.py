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
import os
import fnmatch

from nnabla.utils.data_iterator import data_iterator
from nnabla.utils.data_source import DataSource
from nnabla.utils.image_utils import imread, imresize

from . import _get_sliced_data_source


image_extentions = [".png"]
file_type_id = {"leftImg8bit": 0, "instanceIds": 1, "labelIds": 2}

##################################
# preprocessing
##################################


def get_cityscape_datalist(args, data_type="train", save_file=False):
    list_path = os.path.abspath(
        "./cityscapes_data_list_{}.txt".format(data_type))
    if os.path.exists(list_path):
        with open(list_path, "r") as f:
            lines = f.readlines()

        return [line.strip().split(",") for line in lines]

    root_dir_path = os.path.abspath(args.data_dir)
    if not os.path.exists(root_dir_path):
        raise ValueError(
            "path for data_dir doesn't exist. ({})".format(args.data_dir))

    collections = {}
    for dirpath, dirnames, filenames in os.walk(root_dir_path):
        # really naive...
        if not fnmatch.fnmatch(dirpath, "*{}*".format(data_type)):
            continue

        images = [filename for filename in filenames if filename.endswith(
            *image_extentions)]

        if len(images) > 0:
            for image in images:
                key = "_".join(image.split("_")[:3])
                file_type = image.split("_")[-1].split(".")[0]

                if file_type not in file_type_id:
                    continue

                image_path = os.path.join(dirpath, image)
                if key not in collections:
                    collections[key] = [None, None, None]

                collections[key][file_type_id[file_type]] = image_path

    outs = collections.values()

    if save_file:
        write_outs = []
        for path_list in outs:
            if None in path_list:
                raise ValueError(
                    "unexpected error is happened during setting up dataset.")

            write_outs.append(",".join(path_list))

        with open(list_path, "w") as f:
            f.write("\n".join(write_outs))

    return list(outs)


##################################################
# data loader / iterator
##################################################

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


class CityScapesIterator(DataSource):
    def __init__(self, data_list, image_shape=(1024, 2048), shuffle=True, rng=None, flip=True):
        super(CityScapesIterator, self).__init__(shuffle=shuffle, rng=rng)

        self._data_list = data_list  # [[image, inst, label], ...]
        self._image_shape = image_shape
        self._size = len(self._data_list)
        self._variables = ("image", "instance_id", "label_id")
        self.flip = flip

        self.reset()

    def reset(self):
        self._idxs = self._rng.permutation(
            self._size) if self.shuffle else np.arange(self._size)

        super(CityScapesIterator, self).reset()

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


def create_data_iterator(batch_size, data_list, image_shape, comm=None, shuffle=True, rng=None,
                         with_memory_cache=False, with_parallel=False, with_file_cache=False, flip=True):
    ds = CityScapesIterator(data_list, image_shape,
                            shuffle=shuffle, rng=rng, flip=flip)

    ds = _get_sliced_data_source(ds, comm, shuffle=shuffle)

    return data_iterator(ds,
                         batch_size,
                         with_memory_cache,
                         with_parallel,
                         with_file_cache)
