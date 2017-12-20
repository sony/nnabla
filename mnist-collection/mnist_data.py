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
Provide data iterator for MNIST examples.
'''
import numpy
import struct
import zlib

from nnabla.logger import logger
from nnabla.utils.data_iterator import data_iterator
from nnabla.utils.data_source import DataSource
from nnabla.utils.data_source_loader import download


def load_mnist(train=True):
    '''
    Load MNIST dataset images and labels from the original page by Yan LeCun or the cache file.

    Args:
        train (bool): The testing dataset will be returned if False. Training data has 60000 images, while testing has 10000 images.

    Returns:
        numpy.ndarray: A shape of (#images, 1, 28, 28). Values in [0.0, 1.0].
        numpy.ndarray: A shape of (#images, 1). Values in {0, 1, ..., 9}.

    '''
    if train:
        image_uri = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
        label_uri = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
    else:
        image_uri = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
        label_uri = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
    logger.info('Getting label data from {}.'.format(label_uri))
    # With python3 we can write this logic as following, but with
    # python2, gzip.object does not support file-like object and
    # urllib.request does not support 'with statement'.
    #
    #   with request.urlopen(label_uri) as r, gzip.open(r) as f:
    #       _, size = struct.unpack('>II', f.read(8))
    #       labels = numpy.frombuffer(f.read(), numpy.uint8).reshape(-1, 1)
    #
    r = download(label_uri)
    data = zlib.decompress(r.read(), zlib.MAX_WBITS | 32)
    _, size = struct.unpack('>II', data[0:8])
    labels = numpy.frombuffer(data[8:], numpy.uint8).reshape(-1, 1)
    r.close()
    logger.info('Getting label data done.')

    logger.info('Getting image data from {}.'.format(image_uri))
    r = download(image_uri)
    data = zlib.decompress(r.read(), zlib.MAX_WBITS | 32)
    _, size, height, width = struct.unpack('>IIII', data[0:16])
    images = numpy.frombuffer(data[16:], numpy.uint8).reshape(
        size, 1, height, width)
    r.close()
    logger.info('Getting image data done.')

    return images, labels


class MnistDataSource(DataSource):
    '''
    Get data directly from MNIST dataset from Internet(yann.lecun.com).
    '''

    def _get_data(self, position):
        image = self._images[self._indexes[position]]
        label = self._labels[self._indexes[position]]
        return (image, label)

    def __init__(self, train=True, shuffle=False, rng=None):
        super(MnistDataSource, self).__init__(shuffle=shuffle)
        self._train = train

        self._images, self._labels = load_mnist(train)

        self._size = self._labels.size
        self._variables = ('x', 'y')
        if rng is None:
            rng = numpy.random.RandomState(313)
        self.rng = rng
        self.reset()

    def reset(self):
        if self._shuffle:
            self._indexes = self.rng.permutation(self._size)
        else:
            self._indexes = numpy.arange(self._size)
        super(MnistDataSource, self).reset()

    @property
    def images(self):
        """Get copy of whole data with a shape of (N, 1, H, W)."""
        return self._images.copy()

    @property
    def labels(self):
        """Get copy of whole label with a shape of (N, 1)."""
        return self._labels.copy()


def data_iterator_mnist(batch_size,
                        train=True,
                        rng=None,
                        shuffle=True,
                        with_memory_cache=False,
                        with_parallel=False,
                        with_file_cache=False):
    '''
    Provide DataIterator with :py:class:`MnistDataSource`
    with_memory_cache, with_parallel and with_file_cache option's default value is all False,
    because :py:class:`MnistDataSource` is able to store all data into memory.

    For example,

    .. code-block:: python

        with data_iterator_mnist(True, batch_size) as di:
            for data in di:
                SOME CODE TO USE data.

    '''
    return data_iterator(MnistDataSource(train=train, shuffle=shuffle, rng=rng),
                         batch_size,
                         with_memory_cache,
                         with_parallel,
                         with_file_cache)
