# Copyright 2023 Sony Group Corporation.
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


from nnabla.utils.data_iterator import data_iterator
from nnabla.utils.data_source import DataSource


class Data(object):
    '''
    Base class.
    Data preparation for model training, including data iterator's definition.
    May need training configurations as input to prepare data according to it.

    Args:
        /

    Example:

        .. code-block:: python

        class MnistData(Data):
            """
            Mnist data.

            Args:
                shuffle (bool): Whether the dataset is shuffled or not.
                rng (None or :obj:`numpy.random.RandomState`): Numpy random number generator.
                batch_size (int): Size of data unit.
            """

            class MnistDataSource(DataSource):
                """
                Get data directly from MNIST dataset from Internet(yann.lecun.com).

                Args:
                    train (bool): Whether the dataset is for training or validation.
                    shuffle (bool): Whether the dataset is shuffled or not.
                    rng (None or :obj:`numpy.random.RandomState`): Numpy random number generator.
                """

                import numpy

                def load_mnist(self, train=True):
                    """
                    Load MNIST dataset images and labels from the original page by Yan LeCun or the cache file.

                    Args:
                        train (bool): The testing dataset will be returned if False. Training data has 60000 images, while testing has 10000 images.

                    Returns:
                        numpy.ndarray: A shape of (#images, 1, 28, 28). Values in [0.0, 1.0].
                        numpy.ndarray: A shape of (#images, 1). Values in {0, 1, ..., 9}.
                    """

                    import struct
                    import zlib

                    from nnabla.logger import logger
                    from nnabla.utils.data_source_loader import download

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

                    r = download(label_uri)
                    data = zlib.decompress(r.read(), zlib.MAX_WBITS | 32)
                    _, size = struct.unpack('>II', data[0:8])
                    labels = self.numpy.frombuffer(
                        data[8:], self.numpy.uint8).reshape(-1, 1)
                    r.close()
                    logger.info('Getting label data done.')

                    logger.info('Getting image data from {}.'.format(image_uri))
                    r = download(image_uri)
                    data = zlib.decompress(r.read(), zlib.MAX_WBITS | 32)
                    _, size, height, width = struct.unpack('>IIII', data[0:16])
                    images = self.numpy.frombuffer(data[16:], self.numpy.uint8).reshape(
                        size, 1, height, width)
                    r.close()
                    logger.info('Getting image data done.')

                    return images, labels

                def _get_data(self, position):
                    image = self._images[self._indexes[position]]
                    label = self._labels[self._indexes[position]]
                    return (image, label)

                def __init__(self, train=True, shuffle=False, rng=None):
                    from numpy.random import RandomState

                    super().__init__(shuffle=shuffle)
                    self._train = train

                    self._images, self._labels = self.load_mnist(train)

                    self._size = self._labels.size
                    self._variables = ('x', 'y')
                    if rng is None:
                        rng = RandomState(313)
                    self.rng = rng
                    self.reset()

                def reset(self):
                    if self._shuffle:
                        self._indexes = self.rng.permutation(self._size)
                    else:
                        self._indexes = self.numpy.arange(self._size)
                    super().reset()

                @property
                def images(self):
                    # Get copy of whole data with a shape of (N, 1, H, W).
                    return self._images.copy()

                @property
                def labels(self):
                    # Get copy of whole label with a shape of (N, 1).
                    return self._labels.copy()

            def __data_source__(self, train, shuffle, rng):
                return self.MnistDataSource(train, shuffle, rng)

            def __data_iterator__(self, data_source, batch_size, rng=None, with_memory_cache=False, with_file_cache=False):
                return data_iterator(data_source, batch_size, rng, with_memory_cache, with_file_cache)

            def __init__(self, shuffle, rng, batch_size):
                self._data_source = self.__data_source__(True, shuffle, rng)
                self._data_iterator = self.__data_iterator__(
                    self._data_source, batch_size, rng)

                self._v_data_source = self.__data_source__(False, shuffle, rng)
                self._v_data_iterator = self.__data_iterator__(
                    self._v_data_source, batch_size)

            def reset(self, train=True):
                if train:
                    self._data_iterator._reset()
                else:
                    self._v_data_iterator._reset()

            def next(self, train=True):
                if train:
                    return self._data_iterator.next()
                else:
                    return self._v_data_iterator.next()

            def size(self, train=True):
                if train:
                    return self._data_iterator.size
                else:
                    return self._v_data_iterator.size
    '''

    class CustomDataSource(DataSource):
        '''
        The place for users to define customized data source.

        Args:
            /
        '''

        pass

    def __data_source__(self):
        '''
        Wrapper to create data source object.
        May put customized configurations for the data source here.

        Args:
            /
        '''

        return self.CustomDataSource()

    def __data_iterator__(self, data_source):
        '''
        Wrapper to create data iterator object.
        May put customized configurations for the data iterator here.

        Args:
            /
        '''

        return data_iterator(data_source)

    def __init__(self):
        self._data_source = self.__data_source__()
        self._data_iterator = self.__data_iterator__(self._data_source)

    def reset(self):
        '''
        Reset data iterator. Use before each epoch or some custom places.

        Args:
            /
        '''

        pass

    def next(self):
        '''
        Prepare data for the next iteration.

        Args:
            /
        '''

        pass

    def size(self):
        '''
        Get batch size.

        Args:
            /
        '''

        pass
