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

from nnabla.utils.data_iterator import data_iterator
from nnabla.utils.data_source import DataSource, SlicedDataSource
from nnabla.utils.image_utils import imread

from . import _get_sliced_data_source


def get_slice_start_end(size, n_slices, rank):
    _size = size // n_slices
    amount = size % n_slices
    slice_start = _size * rank
    if rank < amount:
        slice_start += rank
    else:
        slice_start += amount

    slice_end = slice_start + _size
    if slice_end > size:
        slice_start -= (slice_end - size)
        slice_end = size

    return slice_start, slice_end


##################################
# preprocessing
##################################


def get_label_to_id_maps(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()[1:]

    ret = {}
    for line in lines:
        _id, _, _, _, _labels = line.strip().split("\t")
        _label = _labels.split(",")[0]  # just use first word for each id.
        ret[_label] = int(_id)

    # id 0 is unknown class on ade20k dataset
    assert 0 not in ret
    ret[0] = "[UNK]"

    return ret


def get_ade20k_datalist(conf, data_type="train", save_file=False):
    d_name = "ade20k"
    if conf.outdoor_only:
        d_name += "_outdoor"

    list_path = os.path.abspath(
        "./{}_data_list_{}.txt".format(d_name, data_type))
    if os.path.exists(list_path):
        with open(list_path, "r") as f:
            lines = f.readlines()

        return [line.strip().split(",") for line in lines]

    root_dir_path = os.path.abspath(conf.data_dir)
    if not os.path.exists(root_dir_path):
        raise ValueError(
            "path for data_dir doesn't exist. ({})".format(conf.data_dir))

    # extract subset ids
    label2id = get_label_to_id_maps(
        os.path.join(root_dir_path, "objectInfo150.txt"))

    subset_ids = []
    if conf.outdoor_only:
        conf.subset_labels = tuple(conf.subset_labels)
        for l in conf.subset_labels:
            subset_ids.append(label2id[l])

    # fix data_type so as to match directory name
    assert data_type in ["train", "val", "training", "validation"]
    if data_type == "train":
        data_type = "training"
    if data_type == "val":
        data_type = "validation"

    # collect files included in subset (.png).
    outs = []
    write_outs = []
    ann_path = os.path.join(root_dir_path, "annotations", data_type)
    img_path = os.path.join(root_dir_path, "images", data_type)
    ann_files = os.listdir(ann_path)
    for filename in ann_files:
        file_path = os.path.join(ann_path, filename)

        # check subset condition if needed
        if len(subset_ids) > 0:
            ann_image = imread(file_path)
            flag = False
            for i in subset_ids:
                if i in ann_image:
                    flag = True
                    break

            if not flag:  # id is not included in the loaded image.
                continue

        ann = file_path
        # images are jpg format.
        img = os.path.join(img_path, os.path.splitext(filename)[0] + ".jpg")

        elm = [img, ann]
        outs.append(elm)
        write_outs.append(",".join(elm))

    if save_file:
        with open(list_path, "w") as f:
            f.write("\n".join(write_outs))

    return outs


##################################################
# data loader / iterator
##################################################

def _crop(x, pos, size, channel_last=False):
    assert not channel_last, "channel_last is not supported"
    return x[..., pos[0]:pos[0] + size[0], pos[1]:pos[1] + size[1]]


def load_function(image_path, label_path, load_shape, crop_shape):
    # naive implementation of loading image.
    _load_shape = (load_shape[1], load_shape[0])
    image = imread(image_path, size=_load_shape,
                   interpolate="bicubic", channel_first=True, num_channels=3)
    label_map = imread(label_path, size=_load_shape, interpolate="nearest")

    if load_shape != crop_shape:
        pos_y = np.random.randint(0, max(0, load_shape[0] - crop_shape[0]))
        pos_x = np.random.randint(0, max(0, load_shape[1] - crop_shape[1]))

        image = _crop(image, (pos_y, pos_x), crop_shape)
        label_map = _crop(label_map, (pos_y, pos_x), crop_shape)

    # normalize
    image = (image - 127.5) / 127.5  # -> [-1, 1]

    return image, label_map


class Ade20kIterator(DataSource):
    def __init__(self, data_list, load_shape=(286, 286), crop_shape=(256, 256),
                 shuffle=True, rng=None, flip=True):
        super(Ade20kIterator, self).__init__(shuffle=shuffle, rng=rng)

        self.load_shape = load_shape
        self.crop_shape = crop_shape
        self.flip = flip

        # mandatory variables of DataSource
        self._data_list = data_list  # [[image, label], ...]
        self._image_shape = crop_shape
        self._size = len(self._data_list)
        self._variables = ("image", "label_id")

        self.reset()

    def reset(self):
        self._idxs = self._rng.permutation(
            self._size) if self.shuffle else np.arange(self._size)

        super(Ade20kIterator, self).reset()

    def __iter__(self):
        self.reset()
        return self

    def _get_data(self, position):
        i = self._idxs[position]
        image_path, label_path = self._data_list[i]

        image, label_map = load_function(
            image_path, label_path, self.load_shape, self.crop_shape)

        if self.flip:
            if np.random.rand() > 0.5:
                image = image[..., ::-1]
                label_map = label_map[..., ::-1]

        return image, label_map


def create_data_iterator(batch_size, data_list, load_shape, crop_shape, comm=None, shuffle=True, rng=None,
                         with_memory_cache=False, with_parallel=False, with_file_cache=False, flip=True):

    ds = Ade20kIterator(data_list, load_shape, crop_shape,
                        shuffle=shuffle, rng=rng, flip=flip)

    # ds.slice turns withMemoryCache flag on forcibly.
    # For data augmentation, this is not desirable and ds.slice is not used here.
    ds = _get_sliced_data_source(ds, comm, shuffle)

    return data_iterator(ds,
                         batch_size,
                         with_memory_cache,
                         with_parallel,
                         with_file_cache)
