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

'''
Provide data iterator for Caltech101 examples.
'''
import numpy as np
import tarfile

from nnabla.logger import logger
from nnabla.utils.data_iterator import data_iterator
from nnabla.utils.data_source import DataSource
from nnabla.utils.data_source_loader import download, get_data_home

from nnabla.utils.image_utils import imresize, imread


class Caltech101DataSource(DataSource):
    '''
    Get data directly from caltech101 dataset from Internet.
    '''

    def _resize_image(self, im, width, height, padding):
        # resize
        h = im.shape[0]
        w = im.shape[1]
        if w != width or h != height:
            # resize image
            if not padding:
                # trimming mode
                if float(h) / w > float(height) / width:
                    target_h = int(float(w) / width * height)
                    im = im[(h - target_h) // 2:h -
                            (h - target_h) // 2, ::]
                else:
                    target_w = int(float(h) / height * width)
                    im = im[::, (w - target_w) // 2:w -
                            (w - target_w) // 2]
            else:
                # padding mode
                if float(h) / w < float(height) / width:
                    target_h = int(float(height) / width * w)
                    pad = (((target_h - h) // 2, target_h -
                            (target_h - h) // 2 - h), (0, 0))
                else:
                    target_w = int(float(width) / height * h)
                    pad = ((0, 0), ((target_w - w) // 2,
                                    target_w - (target_w - w) // 2 - w))
                pad = pad + ((0, 0),)
                im = np.pad(im, pad, 'constant')
            im = imresize(im, (width, height))
        x = np.array(im, dtype=np.uint8).transpose((2, 0, 1))
        return x

    def _get_data(self, position):
        image = self._images[self._indexes[position]]
        label = self._labels[self._indexes[position]]
        return (image, label)

    def __init__(self, width, height, padding, train=True, shuffle=False, rng=None):
        super(Caltech101DataSource, self).__init__(shuffle=shuffle, rng=rng)
        data_uri = "http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz"
        logger.info('Getting labeled data from {}.'.format(data_uri))
        r = download(data_uri)  # file object returned
        label_dict = dict()
        with tarfile.open(fileobj=r, mode="r:gz") as fpin:
            images = []
            labels = []
            for name in fpin.getnames():
                if ".jpg" not in name or "Google" in name:
                    continue
                label, filename = name.split("/")[-2:]
                if label not in label_dict:
                    label_dict[label] = len(label_dict)
                im = imread(fpin.extractfile(name), num_channels=3)
                arranged_images = self._resize_image(
                    im, width, height, padding)
                images.append(arranged_images)
                labels.append(label_dict[label])
            self._size = len(images)
            self._images = np.array(images)
            self._labels = np.array(labels).reshape(-1, 1)
        r.close()
        logger.info('Getting labeled data from {}.'.format(data_uri))

        self._size = self._labels.size
        self._variables = ('x', 'y')
        if rng is None:
            rng = np.random.RandomState(313)
        self.rng = rng
        self._indexes = rng.permutation(self._size)

    def reset(self):
        if self._shuffle:
            self._indexes = self.rng.permutation(self._size)
        else:
            self._indexes = self._indexes
        super(Caltech101DataSource, self).reset()

    @property
    def images(self):
        """Get copy of whole data with a shape of (N, 3, H, W)."""
        return self._images.copy()

    @property
    def labels(self):
        """Get copy of whole label with a shape of (N, 3)."""
        return self._labels.copy()


def data_iterator_caltech101(batch_size,
                             train=True,
                             rng=None,
                             shuffle=True,
                             width=128,
                             height=128,
                             padding=True,
                             with_memory_cache=False,
                             with_file_cache=False):
    '''
    Provide DataIterator with :py:class:`Caltech101DataSource`
    with_memory_cache and with_file_cache option's default value is all False,
    because :py:class:`Caltech101DataSource` is able to store all data into memory.
    '''
    return data_iterator(Caltech101DataSource(width=width, height=height, padding=padding, train=train, shuffle=shuffle, rng=rng),
                         batch_size,
                         rng,
                         with_memory_cache,
                         with_file_cache)
