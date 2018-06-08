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

import os
import glob
import shutil
import numpy as np
import six
import requests
import tqdm
import zipfile
from PIL import Image

import nnabla.logger as logger
from nnabla.utils.data_iterator import data_iterator
from nnabla.utils.data_source import DataSource


class FacadeDataSource(DataSource):
    DATASET_NAME = 'facades'
    URLs = ['http://cmp.felk.cvut.cz/~tylecr1/facade/CMP_facade_DB_base.zip',
            'http://cmp.felk.cvut.cz/~tylecr1/facade/CMP_facade_DB_extended.zip']
    DEFAULT_DATASET_ROOT_PATH = './datasets'
    DEFAULT_TRAIN_ROOT_PATH = DEFAULT_DATASET_ROOT_PATH + '/' + DATASET_NAME + '/train'
    DEFAULT_VAL_ROOT_PATH = DEFAULT_DATASET_ROOT_PATH + '/' + DATASET_NAME + '/val'
    DEFAULT_TEST_ROOT_PATH = DEFAULT_DATASET_ROOT_PATH + '/' + DATASET_NAME + '/test'

    def __init__(self, images_root_path=None, cropsize=(256, 256), random_crop=True, shuffle=True, rng=None):
        super(FacadeDataSource, self).__init__(shuffle=shuffle, rng=rng)
        if images_root_path is None:
            images_root_path = FacadeDataSource.DEFAULT_TRAIN_ROOT_PATH

        if not os.path.exists(images_root_path):
            self.download()

        self._facade_images = glob.glob(images_root_path + '/*.jpg')
        self._crop_size = cropsize
        self._random_crop = random_crop
        self._size = len(self._facade_images)

        self._variables = ('x', 'y')  # should be defined
        self.reset()  # should be called

    def download(self, train=400, val=100):
        # Load Facade Image Data
        facade_raw = os.path.join(
            FacadeDataSource.DEFAULT_DATASET_ROOT_PATH, FacadeDataSource.DATASET_NAME, 'raw')
        if (not os.path.exists(FacadeDataSource.DEFAULT_TRAIN_ROOT_PATH) or
                not os.path.exists(FacadeDataSource.DEFAULT_VAL_ROOT_PATH) or
                not os.path.exists(FacadeDataSource.DEFAULT_TEST_ROOT_PATH)):
            # Make Default Dataset Root Directory
            if not os.path.exists(FacadeDataSource.DEFAULT_DATASET_ROOT_PATH):
                os.makedirs(FacadeDataSource.DEFAULT_DATASET_ROOT_PATH)
            # Download
            for url in FacadeDataSource.URLs:
                logger.info('Downloading Dataset from {0:s}...'.format(url))
                download_path = os.path.join(
                    FacadeDataSource.DEFAULT_DATASET_ROOT_PATH, os.path.basename(url))
                # Make Download Progress Bar
                tsize = int(requests.get(
                    url, stream=True).headers['Content-Length'])
                with tqdm.tqdm(total=tsize) as bar:
                    update_bar = lambda bcount, bsize, total, bar=bar: bar.update(
                        bsize)
                    six.moves.urllib.request.urlretrieve(
                        url, download_path, reporthook=update_bar)
                # Extract Dataset
                with zipfile.ZipFile(download_path, 'r') as zfile:
                    zfile.extractall(path=facade_raw)
            # split into train/val/test dataset
            image_paths = []
            for d in [_d for _d in glob.glob(facade_raw+'/*') if os.path.isdir(_d)]:
                image_paths.extend(glob.glob(d + '/*.jpg'))
            self._rng.shuffle(image_paths)
            test = len(image_paths) - \
                (train + val) if len(image_paths) > (train + val) else 0
            for n, d in zip(
                    [train, val, test],
                    [FacadeDataSource.DEFAULT_TRAIN_ROOT_PATH,
                        FacadeDataSource.DEFAULT_VAL_ROOT_PATH,
                        FacadeDataSource.DEFAULT_TEST_ROOT_PATH]):
                if not os.path.exists(d):
                    os.makedirs(d)
                for i in range(n):
                    image_p = image_paths.pop()
                    label_p = os.path.splitext(image_p)[0] + '.png'
                    shutil.copy(image_p, d)
                    shutil.copy(label_p, d)
        else:
            logger.info('Facade Dataset is already downloaded.')

    def _get_data(self, position):
        # Load from Filename
        idx = self._indexes[position]
        label_name = os.path.splitext(self._facade_images[idx])[0] + '.png'
        image = Image.open(self._facade_images[idx])
        _label = Image.open(label_name)

        # Resize Images
        w, h = image.size
        r = 286. / min(w, h)
        image = image.resize((int(r*w), int(r*h)), Image.BILINEAR)
        image = np.asarray(image).astype('f').transpose(2, 0, 1)
        _label = _label.resize((int(r*w), int(r*h)), Image.NEAREST)
        _label = np.asarray(_label) - 1

        # Normalize Image
        image = image / 128.0 - 1.0

        # random crop
        c, h, w = image.shape
        ch, cw = self._crop_size
        top_left = (self._rng.randint(0, h-ch), self._rng.randint(0,
                                                                  w-cw)) if self._random_crop else (0, 0)
        image = image[:, top_left[0]:top_left[0] +
                      ch, top_left[1]:top_left[1]+cw]
        _label = _label[top_left[0]:top_left[0]+ch, top_left[1]:top_left[1]+cw]

        # Make Label Image
        label = np.zeros((12, image.shape[1], image.shape[2])).astype("i")
        for i in range(12):
            label[i, :] = _label == i
        return (image, label)

    def reset(self):
        # reset method initilize self._indexes
        if self._shuffle:
            self._indexes = self._rng.permutation(self._size)
        else:
            self._indexes = np.arange(self._size)
        super(FacadeDataSource, self).reset()


def facade_data_iterator(
        images_root_path,
        batch_size,
        random_crop=True,
        shuffle=True,
        rng=None,
        with_memory_cache=True,
        with_parallel=False,
        with_file_cache=False):
    return data_iterator(FacadeDataSource(images_root_path, random_crop=random_crop, shuffle=shuffle, rng=rng),
                         batch_size,
                         with_memory_cache,
                         with_file_cache)
