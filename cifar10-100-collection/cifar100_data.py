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


'''
Provide data iterator for Cifar100 examples.
'''
from contextlib import contextmanager
import numpy as np
import struct
import tarfile
import zlib
import time
import os
import errno

from nnabla.logger import logger
from nnabla.utils.data_iterator import data_iterator
from nnabla.utils.data_source import DataSource
from nnabla.utils.data_source_loader import download, get_data_home


class Cifar100DataSource(DataSource):
    '''
    Get data directly from cifar100 dataset from Internet(yann.lecun.com).
    '''

    def _get_data(self, position):
        image = self._images[self._indexes[position]]
        label = self._labels[self._indexes[position]]
        return (image, label)

    def __init__(self, train=True, shuffle=False, rng=None):
        super(Cifar100DataSource, self).__init__(shuffle=shuffle)

        self._train = train
        data_uri = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
        logger.info('Getting labeled data from {}.'.format(data_uri))
        r = download(data_uri)  # file object returned
        with tarfile.open(fileobj=r, mode="r:gz") as fpin:
            # Training data
            if train:
                images = []
                labels = []
                for member in fpin.getmembers():
                    if "train" not in member.name:
                        continue
                    fp = fpin.extractfile(member)
                    data = np.load(fp, encoding="bytes", allow_pickle=True)
                    images = data[b"data"]
                    labels = data[b"fine_labels"]
                self._size = 50000
                self._images = images.reshape(self._size, 3, 32, 32)
                self._labels = np.array(labels).reshape(-1, 1)
            # Validation data
            else:
                for member in fpin.getmembers():
                    if "test" not in member.name:
                        continue
                    fp = fpin.extractfile(member)
                    data = np.load(fp, encoding="bytes", allow_pickle=True)
                    images = data[b"data"]
                    labels = data[b"fine_labels"]
                self._size = 10000
                self._images = images.reshape(self._size, 3, 32, 32)
                self._labels = np.array(labels).reshape(-1, 1)
        r.close()
        logger.info('Getting labeled data from {} done.'.format(data_uri))

        self._size = self._labels.size
        self._variables = ('x', 'y')
        if rng is None:
            rng = np.random.RandomState(313)
        self.rng = rng
        self.reset()

    def reset(self):
        if self._shuffle:
            self._indexes = self.rng.permutation(self._size)
        else:
            self._indexes = np.arange(self._size)
        super(Cifar100DataSource, self).reset()

    @property
    def images(self):
        """Get copy of whole data with a shape of (N, 1, H, W)."""
        return self._images.copy()

    @property
    def labels(self):
        """Get copy of whole label with a shape of (N, 1)."""
        return self._labels.copy()


def data_iterator_cifar100(batch_size,
                           train=True,
                           rng=None,
                           shuffle=True,
                           with_memory_cache=False,
                           with_file_cache=False):
    '''
    Provide DataIterator with :py:class:`Cifar100DataSource`
    with_memory_cache and with_file_cache option's default value is all False,
    because :py:class:`Cifar100DataSource` is able to store all data into memory.
    '''
    return data_iterator(Cifar100DataSource(train=train, shuffle=shuffle, rng=rng),
                         batch_size,
                         rng,
                         with_memory_cache,
                         with_file_cache)
