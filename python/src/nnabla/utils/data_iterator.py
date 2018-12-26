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
Data Iterator is module for getting data from user defined source with following features.

Detailed design document is :doc:`/doc/designs/data_iterator`.

'''

import atexit
import numpy
import six
import threading

from .data_source import DataSourceWithFileCache
from .data_source import DataSourceWithMemoryCache
from .data_source import SlicedDataSource

from .data_source_implements import SimpleDataSource
from .data_source_implements import CsvDataSource
from .data_source_implements import CacheDataSource
from .data_source_implements import ConcatDataSource

from nnabla.logger import logger


class DataIterator(object):
    '''DataIterator
    Collect data from :ref:`data_source_design` and yields bunch of data.

    Detailed documentation is available in :ref:`data_iterator_design`.

    Args:
        data_source (:py:class:`DataSource <nnabla.utils.data_source.DataSource>`):
             Instance of DataSource class witch provides data for this class.
        batch_size (int): Size of data unit.
        rng (None or :obj:`numpy.random.RandomState`): Numpy random number
            generator.
        epoch_begin_callbacks (list of functions): An item is a function
            which takes an epoch index as a argument. These are called
            at the beginning of an epoch.
        epoch_end_callbacks (list of functions): An item is a function
            which takes an epoch index as a argument. These are called
            at the end of an epoch.

    '''

    def __init__(self,
                 data_source,
                 batch_size,
                 rng=None,
                 use_thread=True,
                 epoch_begin_callbacks=[],
                 epoch_end_callbacks=[]):
        logger.info('Using DataIterator')
        if rng is None:
            rng = numpy.random.RandomState(313)
        self._rng = rng
        self._shape = None       # Only use with padding
        self._data_position = 0  # Only use with padding

        self._data_source = data_source
        # place holder for shuffle is enabled or not at starting time.
        self._shuffle = self._data_source.shuffle

        self._variables = data_source.variables
        self._num_of_variables = len(data_source.variables)
        self._batch_size = batch_size
        self._epoch = -1

        self._epoch_end_callbacks = list(epoch_end_callbacks)
        self._epoch_begin_callbacks = list(epoch_begin_callbacks)

        self._size = data_source.size

        self._reset()
        self._current_epoch = -1
        self._current_data = None

        self._use_thread = use_thread
        if self._use_thread:
            self._next_thread = threading.Thread(target=self._next)
            self._next_thread.start()

        self._closed = False
        atexit.register(self.close)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass

    def close(self):
        if not self._closed:
            if six.PY3:
                atexit.unregister(self.close)
            self._data_source.close()
            self._closed = True

    @property
    def epoch(self):
        '''epoch
        The number of times :py:meth:`position` returns to zero.

        Returns:
            int: epoch
        '''
        return self._current_epoch

    @property
    def position(self):
        '''position

        Data position in current epoch.

        Returns:
            int: Data position
        '''
        return self._data_source.position

    @property
    def size(self):
        '''size

        Data size that DataIterator will generate.
        This is the largest integer multiple of batch_size not exceeding
        :py:meth:`self._data_source.size`.

        Returns:
            int: Data size

        '''
        return self._size

    @property
    def variables(self):
        '''variables

        Variable names of the data.

        Returns:
           tuple: tuple of Variable names
        '''
        return self._variables

    @property
    def batch_size(self):
        '''batch_size

        Number of training samples that :py:meth:`next()` returns.

        Returns:
            int: Number of training samples.
        '''
        return self._batch_size

    def _reset(self):
        self._callback_epoch_end()
        self._epoch += 1
        self._callback_epoch_begin()
        self._data_source.reset()

    def _next(self):
        data = [[] for x in self._variables]

        for b in range(self._batch_size):

            d = self._data_source.next()
            if d is None:
                self._current_data = None
                return

            if self._data_source.position >= self._size:
                self._reset()

            for i, v in enumerate(self._variables):
                data[i].append(d[i])

        self._current_data = (self._epoch, tuple(
            [numpy.array(x) for x in data]))

    def next(self):
        '''next

        It generates tuple of data.

        For example,
        if :py:meth:`self._variables == ('x', 'y')`
        This method returns :py:meth:` ( [[X] * batch_size], [[Y] * batch_size] )`

        Returns:
            tuple: tuple of data for mini-batch in numpy.ndarray.
        '''
        if self._use_thread:
            # Wait for finish previous thread.
            self._next_thread.join()

            if self._current_data is None:
                logger.log(99, 'next() got None retrying.')
                self._next_thread = threading.Thread(target=self._next)
                self._next_thread.start()
                self._next_thread.join()
            self._current_epoch, data = self._current_data
            # Start next thread.
            self._next_thread = threading.Thread(target=self._next)
            self._next_thread.start()
        else:
            self._next()
            self._current_epoch, data = self._current_data

        return data

    def slice(self, rng, num_of_slices=None, slice_pos=None,
              slice_start=None, slice_end=None,
              cache_dir=None):
        '''
        Slices the data iterator so that newly generated data iterator has access to limited portion of the original data.

        Args:
            rng (numpy.random.RandomState): Random generator for Initializer.
            num_of_slices(int): Total number of slices to be made. Muts be used together with `slice_pos`. 
            slice_pos(int): Position of the slice to be assigned to the new data iterator. Must be used together with `num_of_slices`.
            slice_start(int): Starting position of the range to be sliced into new data iterator. Must be used together with `slice_end`.
            slice_end(int) : End position of the range to be sliced into new data iterator. Must be used together with `slice_start`.
            cache_dir(str) : Directory to save cache files

        Example:

        .. code-block:: python

            from nnabla.utils.data_iterator import data_iterator_simple
            import numpy as np

            def load_func1(index):
                d = np.ones((2, 2)) * index
                return d

            di = data_iterator_simple(load_func1, 1000, batch_size=3)

            di_s1 = di.slice(None, num_of_slices=10, slice_pos=0)
            di_s2 = di.slice(None, num_of_slices=10, slice_pos=1)

            di_s3 = di.slice(None, slice_start=100, slice_end=200)
            di_s4 = di.slice(None, slice_start=300, slice_end=400) 

        '''

        if num_of_slices is not None and slice_pos is not None and slice_start is None and slice_end is None:
            size = self._size // num_of_slices
            amount = self._size % num_of_slices
            slice_start = slice_pos * size
            if slice_pos < amount:
                slice_start += slice_pos
            else:
                slice_start += amount
            slice_end = slice_start + size
            if slice_end > self._size:
                slice_start -= (slice_end - self._size)
                slice_end = self._size

        elif num_of_slices is None and slice_pos is None and slice_start is not None and slice_end is not None:
            pass
        else:
            logger.critical(
                'You must specify position(num_of_slice and slice_pos) or range(slice_start and slice_end).')
            return None

        if cache_dir is None:
            ds = self._data_source
            while '_data_source' in dir(ds):
                if '_cache_dir' in dir(ds):
                    cache_dir = ds._cache_dir
                ds = ds._data_source

        if cache_dir is None:
            return DataIterator(
                DataSourceWithMemoryCache(
                    SlicedDataSource(
                        self._data_source,
                        self._data_source.shuffle,
                        slice_start=slice_start,
                        slice_end=slice_end),
                    shuffle=self._shuffle,
                    rng=rng),
                self._batch_size)
        else:
            return DataIterator(
                DataSourceWithMemoryCache(
                    DataSourceWithFileCache(
                        SlicedDataSource(
                            self._data_source,
                            self._data_source.shuffle,
                            slice_start=slice_start,
                            slice_end=slice_end),
                        cache_dir=cache_dir,
                        cache_file_name_prefix='cache_sliced_{:08d}_{:08d}'.format(
                            slice_start,
                            slice_end),
                        shuffle=self._shuffle,
                        rng=rng),
                    shuffle=self._shuffle,
                    rng=rng),
                self._batch_size)

    def _callback_epoch_end(self):
        for callback in self._epoch_end_callbacks:
            callback(self.epoch)

    def _callback_epoch_begin(self):
        for callback in self._epoch_begin_callbacks:
            callback(self.epoch)

    def register_epoch_end_callback(self, callback):
        """Register epoch end callback.

        Args:
            callback (function): A function takes an epoch index as an argument.
        """
        self._epoch_end_callbacks.append(callback)

    def register_epoch_begin_callback(self, callback):
        """Register epoch begin callback.

        Args:
            callback (function): A function takes an epoch index as an argument.
        """
        self._epoch_begin_callbacks.append(callback)


def data_iterator(data_source,
                  batch_size,
                  rng=None,
                  with_memory_cache=True,
                  with_file_cache=False,
                  cache_dir=None,
                  epoch_begin_callbacks=[],
                  epoch_end_callbacks=[]):
    '''data_iterator
    Helper method to use :py:class:`DataSource <nnabla.utils.data_source.DataSource>`.

    You can use :py:class:`DataIterator <nnabla.utils.data_iterator.DataIterator>` with your own :py:class:`DataSource <nnabla.utils.data_source.DataSource>`
    for easy implementation of data sources.

    For example,

    .. code-block:: python

        ds = YourOwnImplementationOfDataSource()
        batch = data_iterator(ds, batch_size)


    Args:
        data_source (:py:class:`DataSource <nnabla.utils.data_source.DataSource>`):
             Instance of DataSource class which provides data.
        batch_size (int): Batch size.
        rng (None or :obj:`numpy.random.RandomState`): Numpy random number
            generator.
        with_memory_cache (bool):
            If ``True``, use :py:class:`.data_source.DataSourceWithMemoryCache`
            to wrap ``data_source``. It is a good idea to set this as true unless
            data_source provides on-memory data.
            Default value is True.
        with_file_cache (bool):
            If ``True``, use :py:class:`.data_source.DataSourceWithFileCache`
            to wrap ``data_source``.
            If ``data_source`` is slow, enabling this option a is good idea.
            Default value is False.
        cache_dir (str):
            Location of file_cache.
            If this value is None, :py:class:`.data_source.DataSourceWithFileCache`
            creates file caches implicitly on temporary directory and erases them all
            when data_iterator is finished.
            Otherwise, :py:class:`.data_source.DataSourceWithFileCache` keeps created cache.
            Default is None.
        epoch_begin_callbacks (list of functions): An item is a function
            which takes an epoch index as an argument. These are called
            at the beginning of an epoch.
        epoch_end_callbacks (list of functions): An item is a function
            which takes an epoch index as an argument. These are called
            at the end of an epoch.

    Returns:
        :py:class:`DataIterator <nnabla.utils.data_iterator.DataIterator>`:
            Instance of DataIterator.
    '''
    if with_file_cache:
        ds = DataSourceWithFileCache(data_source=data_source,
                                     cache_dir=cache_dir,
                                     shuffle=data_source.shuffle,
                                     rng=rng)
        if with_memory_cache:
            ds = DataSourceWithMemoryCache(ds,
                                           shuffle=ds.shuffle,
                                           rng=rng)
        return DataIterator(ds,
                            batch_size,
                            epoch_begin_callbacks=epoch_begin_callbacks,
                            epoch_end_callbacks=epoch_end_callbacks)
    else:
        if with_memory_cache:
            data_source = DataSourceWithMemoryCache(data_source,
                                                    shuffle=data_source.shuffle,
                                                    rng=rng)
        return DataIterator(data_source, batch_size,
                            epoch_begin_callbacks=epoch_begin_callbacks,
                            epoch_end_callbacks=epoch_end_callbacks)


def data_iterator_simple(load_func,
                         num_examples,
                         batch_size,
                         shuffle=False,
                         rng=None,
                         with_memory_cache=True,
                         with_file_cache=True,
                         cache_dir=None,
                         epoch_begin_callbacks=[],
                         epoch_end_callbacks=[]):
    """A generator that ``yield`` s minibatch data as a tuple, as defined in ``load_func`` .
    It can unlimitedly yield minibatches at your request, queried from the provided data.

    Args:
        load_func (function): Takes a single argument `i`, an index of an
            example in your dataset to be loaded, and returns a tuple of data.
            Every call by any index `i` must return a tuple of arrays with
            the same shape.
        num_examples (int): Number of examples in your dataset. Random sequence
            of indexes is generated according to this number.
        batch_size (int): Size of data unit.
        shuffle (bool):
             Indicates whether the dataset is shuffled or not.
             Default value is False. 
        rng (None or :obj:`numpy.random.RandomState`): Numpy random number
            generator.
        with_memory_cache (bool):
            If ``True``, use :py:class:`.data_source.DataSourceWithMemoryCache`
            to wrap ``data_source``. It is a good idea to set this as true unless
            data_source provides on-memory data.
            Default value is True.
        with_file_cache (bool):
            If ``True``, use :py:class:`.data_source.DataSourceWithFileCache`
            to wrap ``data_source``.
            If ``data_source`` is slow, enabling this option a is good idea.
            Default value is False.
        cache_dir (str):
            Location of file_cache.
            If this value is None, :py:class:`.data_source.DataSourceWithFileCache`
            creates file caches implicitly on temporary directory and erases them all
            when data_iterator is finished.
            Otherwise, :py:class:`.data_source.DataSourceWithFileCache` keeps created cache.
            Default is None.
        epoch_begin_callbacks (list of functions): An item is a function
            which takes an epoch index as an argument. These are called
            at the beginning of an epoch.
        epoch_end_callbacks (list of functions): An item is a function
            which takes an epoch index as an argument. These are called
            at the end of an epoch.


    Returns:
        :py:class:`DataIterator <nnabla.utils.data_iterator.DataIterator>`:
            Instance of DataIterator.


    Here is an example of `load_func` which returns an image and a label of a
    classification dataset.

    .. code-block:: python

        import numpy as np
        from nnabla.utils.image_utils import imread
        image_paths = load_image_paths()
        labels = load_labels()
        def my_load_func(i):
            '''
            Returns:
                image: c x h x w array
                label: 0-shape array
            '''
            img = imread(image_paths[i]).astype('float32')
            return np.rollaxis(img, 2), np.array(labels[i])


    """
    return data_iterator(SimpleDataSource(load_func,
                                          num_examples,
                                          shuffle=shuffle,
                                          rng=rng),
                         batch_size=batch_size,
                         with_memory_cache=with_memory_cache,
                         with_file_cache=with_file_cache,
                         cache_dir=cache_dir,
                         epoch_begin_callbacks=epoch_begin_callbacks,
                         epoch_end_callbacks=epoch_end_callbacks)


def data_iterator_csv_dataset(uri,
                              batch_size,
                              shuffle=False,
                              rng=None,
                              normalize=True,
                              with_memory_cache=True,
                              with_file_cache=True,
                              cache_dir=None,
                              epoch_begin_callbacks=[],
                              epoch_end_callbacks=[]):
    '''data_iterator_csv_dataset
    Get data directly from a dataset provided as a CSV file.

    You can read files located on the local file system, http(s) servers or Amazon AWS S3 storage.

    For example,

    .. code-block:: python

        batch = data_iterator_csv_dataset('CSV_FILE.csv', batch_size, shuffle=True)

    Args:
        uri (str): Location of dataset CSV file.
        batch_size (int): Size of data unit.
        shuffle (bool):
             Indicates whether the dataset is shuffled or not.
             Default value is False. 
        rng (None or :obj:`numpy.random.RandomState`): Numpy random number
            generator.
        normalize (bool): If True, each sample in the data gets normalized by a factor of 255. 
            Default is True.
        with_memory_cache (bool):
            If ``True``, use :py:class:`.data_source.DataSourceWithMemoryCache`
            to wrap ``data_source``. It is a good idea to set this as true unless
            data_source provides on-memory data.
            Default value is True.
        with_file_cache (bool):
            If ``True``, use :py:class:`.data_source.DataSourceWithFileCache`
            to wrap ``data_source``.
            If ``data_source`` is slow, enabling this option a is good idea.
            Default value is False.
        cache_dir (str):
            Location of file_cache.
            If this value is None, :py:class:`.data_source.DataSourceWithFileCache`
            creates file caches implicitly on temporary directory and erases them all
            when data_iterator is finished.
            Otherwise, :py:class:`.data_source.DataSourceWithFileCache` keeps created cache.
            Default is None.
        epoch_begin_callbacks (list of functions): An item is a function
            which takes an epoch index as an argument. These are called
            at the beginning of an epoch.
        epoch_end_callbacks (list of functions): An item is a function
            which takes an epoch index as an argument. These are called
            at the end of an epoch.


    Returns:
        :py:class:`DataIterator <nnabla.utils.data_iterator.DataIterator>`:
            Instance of DataIterator
    '''
    ds = CsvDataSource(uri,
                       shuffle=shuffle,
                       rng=rng,
                       normalize=normalize)

    return data_iterator(ds,
                         batch_size=batch_size,
                         with_memory_cache=with_memory_cache,
                         with_file_cache=with_file_cache,
                         cache_dir=cache_dir,
                         epoch_begin_callbacks=epoch_begin_callbacks,
                         epoch_end_callbacks=epoch_end_callbacks)


def data_iterator_cache(uri,
                        batch_size,
                        shuffle=False,
                        rng=None,
                        normalize=True,
                        with_memory_cache=True,
                        epoch_begin_callbacks=[],
                        epoch_end_callbacks=[]):
    '''data_iterator_cache
    Get data from the cache directory.

    Cache files are read from the local file system.

    For example,

    .. code-block:: python

        batch = data_iterator_cache('CACHE_DIR', batch_size, shuffle=True)

    Args:
        uri (str): Location of directory with cache files.
        batch_size (int): Size of data unit.
        shuffle (bool):
             Indicates whether the dataset is shuffled or not.
             Default value is False. 
        rng (None or :obj:`numpy.random.RandomState`): Numpy random number
            generator.
        normalize (bool): If True, each sample in the data gets normalized by a factor of 255. 
            Default is True.
        with_memory_cache (bool):
            If ``True``, use :py:class:`.data_source.DataSourceWithMemoryCache`
            to wrap ``data_source``. It is a good idea to set this as true unless
            data_source provides on-memory data.
            Default value is True.
        epoch_begin_callbacks (list of functions): An item is a function
            which takes an epoch index as an argument. These are called
            at the beginning of an epoch.
        epoch_end_callbacks (list of functions): An item is a function
            which takes an epoch index as an argument. These are called
            at the end of an epoch.


    Returns:
        :py:class:`DataIterator <nnabla.utils.data_iterator.DataIterator>`:
            Instance of DataIterator
    '''
    ds = CacheDataSource(uri,
                         shuffle=shuffle,
                         rng=rng,
                         normalize=normalize)

    return data_iterator(ds,
                         batch_size=batch_size,
                         with_memory_cache=with_memory_cache,
                         epoch_begin_callbacks=epoch_begin_callbacks,
                         epoch_end_callbacks=epoch_end_callbacks)


def data_iterator_concat_datasets(data_source_list,
                                  batch_size,
                                  shuffle=False,
                                  rng=None,
                                  with_memory_cache=True,
                                  with_file_cache=False,
                                  cache_dir=None,
                                  epoch_begin_callbacks=[],
                                  epoch_end_callbacks=[]):
    '''data_iterator_concat_datasets
    Get data from multiple datasets.

    For example,

    .. code-block:: python

        batch = data_iterator_concat_datasets([DataSource0, DataSource1, ...], batch_size)

    Args:
        data_source_list (list of DataSource): list of datasets.
        batch_size (int): Size of data unit.
        shuffle (bool):
             Indicates whether the dataset is shuffled or not.
             Default value is False. 
        rng (None or :obj:`numpy.random.RandomState`): Numpy random number
            generator.
        with_memory_cache (bool):
            If ``True``, use :py:class:`.data_source.DataSourceWithMemoryCache`
            to wrap ``data_source``. It is a good idea to set this as true unless
            data_source provides on-memory data.
            Default value is True.
        with_file_cache (bool):
            If ``True``, use :py:class:`.data_source.DataSourceWithFileCache`
            to wrap ``data_source``.
            If ``data_source`` is slow, enabling this option a is good idea.
            Default value is False.
        cache_dir (str):
            Location of file_cache.
            If this value is None, :py:class:`.data_source.DataSourceWithFileCache`
            creates file caches implicitly on temporary directory and erases them all
            when data_iterator is finished.
            Otherwise, :py:class:`.data_source.DataSourceWithFileCache` keeps created cache.
            Default is None.
        epoch_begin_callbacks (list of functions): An item is a function
            which takes an epoch index as an argument. These are called
            at the beginning of an epoch.
        epoch_end_callbacks (list of functions): An item is a function
            which takes an epoch index as an argument. These are called
            at the end of an epoch.


    Returns:
        :py:class:`DataIterator <nnabla.utils.data_iterator.DataIterator>`:
            Instance of DataIterator
    '''
    ds = ConcatDataSource(data_source_list,
                          shuffle=shuffle,
                          rng=rng)
    return data_iterator(ds,
                         batch_size=batch_size,
                         with_memory_cache=with_memory_cache,
                         with_file_cache=with_file_cache,
                         epoch_begin_callbacks=epoch_begin_callbacks,
                         epoch_end_callbacks=epoch_end_callbacks)
