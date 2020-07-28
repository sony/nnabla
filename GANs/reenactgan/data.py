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

import os
import glob
import numpy as np

import nnabla.logger as logger
from nnabla.utils.data_iterator import data_iterator
from nnabla.utils.data_source import DataSource
from nnabla.utils.image_utils import imread

from utils import get_points_list, get_bod_map


def celebv_data_iterator(dataset_mode=None, celeb_name=None, data_dir=None, ref_dir=None,
                         mode="all", batch_size=1, shuffle=False, rng=None,
                         with_memory_cache=False, with_file_cache=False,
                         resize_size=(64, 64), line_thickness=3, gaussian_kernel=(5, 5), gaussian_sigma=3
                         ):

    if dataset_mode == 'transformer':
        if ref_dir:
            assert os.path.exists(ref_dir), f'{ref_dir} not found.'
            logger.info(
                'CelebV Dataiterator using reference .npz file for Transformer is created.')
            return data_iterator(CelebVDataRefSource(
                                celeb_name=celeb_name, data_dir=data_dir, ref_dir=ref_dir,
                                need_image=False, need_heatmap=True, need_resized_heatmap=False,
                                mode=mode, shuffle=shuffle, rng=rng),
                                batch_size, rng, with_memory_cache, with_file_cache)

        else:
            logger.info('CelebV Dataiterator for Transformer is created.')
            return data_iterator(CelebVDataSource(
                            celeb_name=celeb_name, data_dir=data_dir,
                            need_image=False, need_heatmap=True, need_resized_heatmap=False,
                            mode=mode, shuffle=shuffle, rng=rng,
                            resize_size=resize_size, line_thickness=line_thickness,
                            gaussian_kernel=gaussian_kernel, gaussian_sigma=gaussian_sigma),
                            batch_size, rng, with_memory_cache, with_file_cache)

    elif dataset_mode == 'decoder':
        if ref_dir:
            assert os.path.exists(ref_dir), f'{ref_dir} not found.'
            logger.info(
                'CelebV Dataiterator using reference .npz file for Decoder is created.')
            return data_iterator(CelebVDataRefSource(
                                celeb_name=celeb_name, data_dir=data_dir, ref_dir=ref_dir,
                                need_image=True, need_heatmap=True, need_resized_heatmap=True,
                                mode=mode, shuffle=shuffle, rng=rng),
                                batch_size, rng, with_memory_cache, with_file_cache)

        else:
            logger.info('CelebV Dataiterator for Decoder is created.')
            return data_iterator(CelebVDataSource(
                            celeb_name=celeb_name, data_dir=data_dir,
                            need_image=True, need_heatmap=True, need_resized_heatmap=True,
                            mode=mode, shuffle=shuffle, rng=rng,
                            resize_size=resize_size, line_thickness=line_thickness,
                            gaussian_kernel=gaussian_kernel, gaussian_sigma=gaussian_sigma),
                            batch_size, rng, with_memory_cache, with_file_cache)

    else:
        logger.error(
            'Specified Dataitaretor is wrong?  given: {}'.format(dataset_mode))
        import sys
        sys.exit()


def wflw_data_iterator(data_dir=None, dataset_mode="encoder_ref", mode="train", use_reference=False,
                       batch_size=1, shuffle=True, rng=None, with_memory_cache=False, with_file_cache=False, transform=None):
    if use_reference:
        logger.info(
            'WFLW Dataset for Encoder using reference .npz file is created.')
        return data_iterator(WFLWDataEncoderRefSource(data_dir, shuffle=shuffle, rng=rng, transform=transform, mode=mode),
                             batch_size, rng, with_memory_cache, with_file_cache)

    else:
        logger.info('WFLW Dataset for Encoder is created.')
        return data_iterator(WFLWDataEncoderSource(data_dir, shuffle=shuffle, rng=rng, transform=transform, mode=mode),
                             batch_size, rng, with_memory_cache, with_file_cache)


class CelebVBaseDatahandler(object):
    """docstring for CelebVBaseDatahandler"""

    def __init__(self, celeb_name=None, data_dir=None, mode="all", shuffle=True, rng=None, resize_size=(64, 64), line_thickness=3, gaussian_kernel=(5, 5), gaussian_sigma=3):

        self.resize_size = resize_size
        self.line_thickness = line_thickness
        self.gaussian_kernel = gaussian_kernel
        self.gaussian_sigma = gaussian_sigma

        celeb_name_list = ['Donald_Trump', 'Emmanuel_Macron',
                           'Jack_Ma', 'Kathleen', 'Theresa_May']
        assert celeb_name in celeb_name_list
        self.data_dir = data_dir
        self._shuffle = shuffle
        self.mode = mode
        self.celeb_name = celeb_name

        self.imgs_root_path = os.path.join(self.data_dir, self.celeb_name)
        if not os.path.exists(self.imgs_root_path):
            logger.error('{} is not exists.'.format(self.imgs_root_path))

        # use an annotation file to know how many images are needed.
        self.ant, self._size = self.get_ant_and_size(
            self.imgs_root_path, self.mode)
        logger.info(f'the number of images for {self.mode}: {self._size}')

        self._variables = list()
        self._ref_files = dict()

        self.reset()

    def get_img_path(self, imgs_root_path, img_name):
        img_path = os.path.join(imgs_root_path, 'Image', img_name)
        return img_path

    def get_img_name(self, ant):
        name = ant.split(' ')[-1].split('\n')[0]
        return name

    def get_img(self, img_path):
        img = imread(img_path, num_channels=3, channel_first=True)
        return img  # (3, 256, 256)

    def normalize_img(self, img, normalize_type="image"):
        if normalize_type == "image":
            # [0, 255] -> [-1, +1]
            img = (img / 255.0) * 2.0 - 1.0
        elif normalize_type == "heatmap":
            # [0, 255] -> [0., 1.]
            img = img / 255.0
        else:
            raise TypeError
        return img

    def get_ant_and_size(self, imgs_root_path, mode="all"):
        if mode == "all":
            filename = 'all_98pt.txt'
        elif mode == "train":
            filename = 'train_98pt.txt'
        else:
            filename = 'test_98pt.txt'
        # get the annotation txt file path
        txt_path = os.path.join(imgs_root_path, filename)
        logger.info(f'checking {txt_path}...')
        with open(txt_path, "r", encoding="utf-8") as f:
            ant = f.readlines()  # read the annotation data from the txt
        size = len(ant)  # the number of training images
        return ant, size

    def _get_data(self, position):
        return NotImplemented

    def reset(self):
        # reset method initialize self._indexes
        if self._shuffle:
            self._indexes = np.arange(self._size)
            np.random.shuffle(self._indexes)
        else:
            self._indexes = np.arange(self._size)


class CelebVDataSource(DataSource, CelebVBaseDatahandler):
    def __init__(self, celeb_name=None, data_dir=None,
                 need_image=False, need_heatmap=True, need_resized_heatmap=False,
                 mode="all", shuffle=True, rng=None,
                 resize_size=(64, 64), line_thickness=3, gaussian_kernel=(5, 5), gaussian_sigma=3):
        super(CelebVDataSource, self).__init__()
        CelebVBaseDatahandler.__init__(self, celeb_name=celeb_name, data_dir=data_dir,
                                       mode=mode, shuffle=shuffle, rng=rng,
                                       resize_size=resize_size, line_thickness=line_thickness, gaussian_kernel=gaussian_kernel, gaussian_sigma=gaussian_sigma)

        if need_image:
            self._variables.append('image')

        if need_heatmap:
            self._variables.append('heatmap')

        if need_resized_heatmap:
            self._variables.append('resized_heatmap')

        assert self._variables  # must contain at least one element
        self._variables = tuple(self._variables)

        self.reset()

    def _get_data(self, position):
        idx = self._indexes[position]
        data = self.get_required_data(idx)
        return data

    def get_required_data(self, idx):
        img_name = self.get_img_name(self.ant[idx])
        img, bod_map, bod_map_resized = None, None, None

        img_path = self.get_img_path(self.imgs_root_path, img_name)
        _img = self.get_img(img_path)  # (3, 256, 256)

        if 'heatmap' in self._variables or 'resized_heatmap' in self._variables:
            # len(x_list)=98, len(y_list)=98
            x_list, y_list = get_points_list(self.ant[idx])

        if 'image' in self._variables:
            img = self.normalize_img(_img)

        if 'heatmap' in self._variables:
            bod_map = get_bod_map(_img, x_list, y_list, resize_size=self.resize_size, line_thickness=self.line_thickness,
                                  gaussian_kernel=self.gaussian_kernel, gaussian_sigma=self.gaussian_sigma)  # (15, 64, 64)
            bod_map = self.normalize_img(bod_map, normalize_type="heatmap")

        if 'resized_heatmap' in self._variables:
            bod_map_resized = get_bod_map(_img, x_list, y_list, resize_size=(256, 256), line_thickness=self.line_thickness,
                                          gaussian_kernel=self.gaussian_kernel, gaussian_sigma=self.gaussian_sigma)  # (15, 256, 256)
            bod_map_resized = self.normalize_img(
                bod_map_resized, normalize_type="heatmap")

        return [_ for _ in (img, bod_map, bod_map_resized) if _ is not None]

    def reset(self):
        # reset method initialize self._indexes
        if self._shuffle:
            self._indexes = np.arange(self._size)
            np.random.shuffle(self._indexes)
        else:
            self._indexes = np.arange(self._size)
        super(CelebVDataSource, self).reset()


class CelebVDataRefSource(DataSource, CelebVBaseDatahandler):
    def __init__(self, celeb_name="Donald_Trump",
                 data_dir="./datasets/CelebV", ref_dir=None,
                 need_image=False, need_heatmap=True, need_resized_heatmap=False,
                 mode="all", shuffle=True, rng=None):
        super(CelebVDataRefSource, self).__init__()
        CelebVBaseDatahandler.__init__(self, celeb_name=celeb_name, data_dir=data_dir,
                                       mode=mode, shuffle=shuffle, rng=rng)
        self.ref_dir = ref_dir

        if need_image:
            self._assign_variable_and_load_ref('image')

        if need_heatmap:
            self._assign_variable_and_load_ref('heatmap')

        if need_resized_heatmap:
            self._assign_variable_and_load_ref('resized_heatmap')

        assert self._variables  # must contain at least one element
        self._variables = tuple(self._variables)

        self.reset()

    def _assign_variable_and_load_ref(self, data):
        assert data in ('image', 'heatmap', 'resized_heatmap')
        self._variables.append(data)
        _ref_path = os.path.join(self.ref_dir, f'{self.celeb_name}_{data}.npz')
        assert _ref_path, f"{_ref_path} does not exist."
        self._ref_files[data] = np.load(_ref_path)
        logger.info(f'loaded {_ref_path}.')

    def _get_data(self, position):
        idx = self._indexes[position]
        data = self.get_required_data(idx)
        return data

    def get_required_data(self, idx):
        img_name = self.get_img_name(self.ant[idx])
        img, bod_map, bod_map_resized = None, None, None

        if 'image' in self._variables:
            img = self._ref_files['image'][img_name]  # uint8, [0, 255]
            img = self.normalize_img(img)

        if 'heatmap' in self._variables:
            bod_map = self._ref_files['heatmap'][img_name]
            bod_map = self.normalize_img(bod_map, normalize_type="heatmap")

        if 'resized_heatmap' in self._variables:
            # uint8, [0, 255]
            bod_map_resized = self._ref_files['resized_heatmap'][img_name]
            bod_map_resized = self.normalize_img(
                bod_map_resized, normalize_type="heatmap")

        return [_ for _ in (img, bod_map, bod_map_resized) if _ is not None]

    def reset(self):
        # reset method initialize self._indexes
        if self._shuffle:
            self._indexes = np.arange(self._size)
            np.random.shuffle(self._indexes)
        else:
            self._indexes = np.arange(self._size)
        super(CelebVDataRefSource, self).reset()


class WFLWDataEncoderSource(DataSource):
    def __init__(self, data_dir=None, shuffle=True, rng=None, transform=None, mode="train"):
        super(WFLWDataEncoderSource, self).__init__()
        self.ref_dir = data_dir  # './datasets/WFLW_heatmaps'
        self.mode = mode
        self.img_dir = os.path.join(
            self.ref_dir, "WFLW_cropped_images", self.mode)
        self.bod_dir = os.path.join(
            self.ref_dir, "WFLW_landmark_images", self.mode)

        self._size = len(glob.glob(f"{self.img_dir}/*.png"))
        self._shuffle = shuffle
        self.transform = transform
        self._variables = ('x', 'y')
        self.reset()

    def _get_data(self, position):
        idx = self._indexes[position]

        # image load
        img = self.get_img(os.path.join(self.img_dir, f"train_{idx}.png"))
        img = self.normalize_img(img)

        # pose
        # uint8, [0, 255]
        bod_map = self.get_img(os.path.join(self.bod_dir, f"train_{idx}.png"))
        bod_map = self.normalize_img(bod_map, normalize_type="heatmap")

        if self.transform is not None:
            img = np.transpose(img, (1, 2, 0))
            bod_map = np.transpose(bod_map, (1, 2, 0))
            aug = self.transform(image=img, mask=bod_map)
            img = aug['image']
            bod_map = aug['mask']
            img = np.transpose(img, (2, 0, 1))
            bod_map = np.transpose(bod_map, (2, 0, 1))

        return img, bod_map

    def reset(self):
        # reset method initialize self._indexes
        if self._shuffle:
            self._indexes = np.arange(self._size)
            np.random.shuffle(self._indexes)
        else:
            self._indexes = np.arange(self._size)
        super(WFLWDataEncoderSource, self).reset()

    def get_img(self, img_path):
        img = imread(img_path, num_channels=3, channel_first=True)
        return img  # (3, 256, 256)

    def normalize_img(self, img, normalize_type="image"):
        if normalize_type == "image":
            # [0, 255] -> [-1, +1]
            img = (img / 255.0) * 2.0 - 1.0
        elif normalize_type == "heatmap":
            # [0, 255] -> [0., 1.]
            img = img / 255.0
        else:
            raise TypeError
        return img


class WFLWDataEncoderRefSource(DataSource):
    def __init__(self, data_dir=None, shuffle=True, rng=None, transform=None, mode="train"):
        super(WFLWDataEncoderRefSource, self).__init__()
        self.ref_dir = data_dir  # './datasets/WFLW_heatmaps'
        self.mode = mode
        ref_img_path = os.path.join(self.ref_dir, f'WFLW_cropped_image_{self.mode}.npz')
        ref_bod_path = os.path.join(self.ref_dir, f'WFLW_heatmap_{self.mode}.npz')
        self.ref_img = np.load(ref_img_path)
        self.ref_bod = np.load(ref_bod_path)

        self._size = len(self.ref_img.files)
        self._shuffle = shuffle
        self.transform = transform
        self._variables = ('x', 'y')
        self.reset()

    def _get_data(self, position):
        idx = self._indexes[position]

        # image load
        img = self.ref_img[f"{self.mode}_{idx}.png"]
        img = self.normalize_img(img)

        # pose
        bod_map = self.ref_bod[f"{self.mode}_{idx}.png"]  # uint8, [0, 255]
        bod_map = self.normalize_img(bod_map, normalize_type="heatmap")

        if self.transform is not None:
            img = np.transpose(img, (1, 2, 0))
            bod_map = np.transpose(bod_map, (1, 2, 0))
            aug = self.transform(image=img, mask=bod_map)
            img = aug['image']
            bod_map = aug['mask']
            img = np.transpose(img, (2, 0, 1))
            bod_map = np.transpose(bod_map, (2, 0, 1))

        return img, bod_map

    def reset(self):
        # reset method initialize self._indexes
        if self._shuffle:
            self._indexes = np.arange(self._size)
            np.random.shuffle(self._indexes)
        else:
            self._indexes = np.arange(self._size)
        super(WFLWDataEncoderRefSource, self).reset()

    def get_img(self, img_path):
        img = imread(img_path, num_channels=3, channel_first=True)
        return img  # (3, 256, 256)

    def normalize_img(self, img, normalize_type="image"):
        if normalize_type == "image":
            # [0, 255] -> [-1, +1]
            img = (img / 255.0) * 2.0 - 1.0
        elif normalize_type == "heatmap":
            # [0, 255] -> [0., 1.]
            img = img / 255.0
        else:
            raise TypeError
        return img
