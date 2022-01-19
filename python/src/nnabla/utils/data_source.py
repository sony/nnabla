# Copyright 2017,2018,2019,2020,2021 Sony Corporation.
# Copyright 2021 Sony Group Corporation.
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


# TODO temporary work around to suppress FutureWarning message.
import warnings
warnings.simplefilter('ignore', category=FutureWarning)

from collections import OrderedDict
from contextlib import closing
from multiprocessing import Queue
from multiprocessing.pool import ThreadPool
from shutil import rmtree
import abc
import atexit

# TODO temporary work around to suppress FutureWarning message.
import warnings
warnings.simplefilter('ignore', category=FutureWarning)
import h5py

import csv
import numpy
import os
import six
import tempfile
import threading

from nnabla.config import nnabla_config
from nnabla.logger import logger
from nnabla.utils.progress import progress
from nnabla.utils.communicator_util import single_or_rankzero
from .data_source_loader import FileReader


class DataSource(object):
    '''
    This class contains various properties and methods for the data source, which are utilized by py:class:`DataIterator`.

    Args:
        shuffle (bool):
             Indicates whether the dataset is shuffled or not.
        rng (None or :obj:`numpy.random.RandomState`): Numpy random number
            generator.

    '''
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def _get_data(self, position):
        pass

    def __init__(self, shuffle=False, rng=None):
        '''
        Init method for DataSource
        '''
        logger.info('DataSource with shuffle({})'.format(shuffle))
        self._rng = rng
        if rng is None:
            self._rng = numpy.random.RandomState(313)
        self._variables = None
        self._generation = -1
        self._shuffle = shuffle
        self._position = 0
        self._size = 0
        self._closed = False
        self._order = None
        self._original_order = None
        self._original_source_uri = None
        atexit.register(self.close)

    def __next__(self):
        return self.next()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def close(self):
        if not self._closed:
            if six.PY3:
                atexit.unregister(self.close)
            self._closed = True

    @property
    def variables(self):
        '''variables

        Variable names of the data.

        Returns:
           tuple: tuple of Variable names
        '''
        return self._variables

    def next(self):
        data = self._get_data(self._position)
        self._position += 1
        return data

    @property
    def position(self):
        '''position

        Data position in current epoch.

        Returns:
            int: Data position
        '''
        return self._position

    @property
    def size(self):
        return self._size

    @property
    def shuffle(self):
        '''

        Whether dataset is shuffled or not.

        Returns:
            bool: whether dataset is shuffled.
        '''

        return self._shuffle

    @shuffle.setter
    def shuffle(self, value):
        self._shuffle = value

    @abc.abstractmethod
    def reset(self):
        self._position = 0


class DataSourceWithFileCacheError(Exception):
    pass


class DataSourceWithFileCache(DataSource):
    '''
    This class contains properties and methods for data source that can be read from cache files, which are utilized by data iterator.

    Args:
        data_source (:py:class:`DataSource <nnabla.utils.data_source.DataSource>`):
             Instance of DataSource class which provides data.
        cache_dir (str):
            Location of file_cache.
            If this value is None, :py:class:`.data_source.DataSourceWithFileCache`
            creates file caches implicitly on temporary directory and erases them all
            when data_iterator is finished.
            Otherwise, :py:class:`.data_source.DataSourceWithFileCache` keeps created cache.
            Default is None.
        cache_file_name_prefix (str):
            Beginning of the filenames of cache files.
            Default is 'cache'. 
        shuffle (bool):
             Indicates whether the dataset is shuffled or not.
        rng (None or :obj:`numpy.random.RandomState`): Numpy random number
            generator.
    '''

    def _save_cache_to_file(self):
        '''
        Store cache data into file.

        Data will be stored as hdf5 format, placed at config..
        Cache file name format is "cache_START_END.h5"
        '''
        if self._cache_dir is None:
            raise DataSourceWithFileCacheError(
                'Use this class with "with statement" if you don\'t specify cache dir.')
        cache_data = OrderedDict()

        def get_data(args):
            pos = args[0]
            q = args[1]
            retry = 1
            while True:
                if retry > 10:
                    logger.log(
                        99, '_get_current_data() retry count over give up.')
                    raise
                d = self._data_source._get_data(pos)
                if d is not None:
                    break
                logger.log(99, '_get_data() fails. retrying count {}/10.'.format(
                           retry))
                retry += 1

            q.put((pos, d))

        q = Queue()
        with closing(ThreadPool(processes=self._num_of_threads)) as pool:
            pool.map(get_data, [(pos, q) for pos in self._cache_positions])

        while len(cache_data) < len(self._cache_positions):
            index, data = q.get()
            cache_data[index] = data
        start_position = self.position - len(cache_data) + 1
        end_position = self.position
        cache_filename = os.path.join(
            self._cache_dir, '{}_{:08d}_{:08d}{}'.format(self._cache_file_name_prefix,
                                                         start_position,
                                                         end_position,
                                                         self._cache_file_format))

        data = OrderedDict([(n, []) for n in self._data_source.variables])
        for pos in sorted(cache_data):
            cd = cache_data[pos]
            for i, n in enumerate(self._data_source.variables):
                if isinstance(cd[i], numpy.ndarray):
                    d = cd[i]
                else:
                    d = numpy.array(cd[i]).astype(numpy.float32)
                data[n].append(d)

        logger.info('Creating cache file {}'.format(cache_filename))
        try:
            if self._cache_file_format == ".h5":
                h5 = h5py.File(cache_filename, 'w')
                for k, v in data.items():
                    h5.create_dataset(k, data=v)
                h5.close()
            else:
                retry_count = 1
                is_create_cache_incomplete = True
                while is_create_cache_incomplete:
                    try:
                        with open(cache_filename, 'wb') as f:
                            for v in data.values():
                                numpy.save(f, v)
                        is_create_cache_incomplete = False
                    except OSError:
                        retry_count += 1
                        if retry_count > 10:
                            raise
                        logger.info(
                            'Creating cache retry {}/10'.format(retry_count))
        except:
            logger.critical(
                'An error occurred while creating cache file from dataset.')
            for k, v in data.items():
                size = v[0].shape
                for d in v:
                    if size != d.shape:
                        logger.critical('The sizes of data "{}" are not the same. ({} != {})'.format(
                            k, size, d.shape))
            raise

        self._cache_file_names.append(cache_filename)
        self._cache_file_order.append(len(self._cache_file_order))
        self._cache_file_data_orders.append(list(range(len(cache_data))))
        self._cache_positions = []

    def _store_data_to_cache_buffer(self, position):
        self._cache_positions.append(position)
        if position == self._total_cached_size:
            self._total_cached_size += 1
            if len(self._cache_positions) >= self._cache_size or self._total_cached_size >= self.size:
                self._save_cache_to_file()

    def _get_data_from_cache_file(self, position):
        cache_file_index = self._cache_file_positions[position]
        cache_data_position = \
            self._cache_file_data_orders[cache_file_index][position -
                                                           self._cache_file_start_positions[cache_file_index]]

        if self._current_cache_file_index != cache_file_index:
            self._current_cache_file_index = cache_file_index

            if self._cache_file_format == '.npy':
                self._current_cache_data = {}
                if not os.path.exists(self._cache_file_names[cache_file_index]):
                    return None
                with open(self._cache_file_names[cache_file_index], 'rb') as f:
                    for v in self._variables:
                        self._current_cache_data[v] = numpy.load(
                            f, allow_pickle=True)
            else:
                h5 = h5py.File(self._cache_file_names[cache_file_index], 'r')
                self._current_cache_data = {}
                for k, v in h5.items():
                    self._current_cache_data[k] = v[()]
                h5.close()

        d = [self._current_cache_data[v][cache_data_position]
             for v in self.variables]
        return d

    def _get_data(self, position):
        with self._thread_lock:
            self._position = position
            return self._get_data_from_cache_file(position)

    def _create_cache(self):
        # Save all data into cache file(s).
        self._cache_positions = []
        self._position = 0

        percent = 0

        if single_or_rankzero():
            progress(None)

        while self._position < self._data_source._size:

            if single_or_rankzero():
                if self._position % int(self._data_source._size/20+1) == 0:
                    progress('Create cache', self._position *
                             1.0 / self._data_source._size)

            self._store_data_to_cache_buffer(self._position)
            self._position += 1
        if len(self._cache_positions) > 0:
            self._save_cache_to_file()

        if single_or_rankzero():
            progress(None)

        # Adjust data size into reseted position. In most case it means
        # multiple of bunch(mini-batch) size.
        num_of_cache_files = int(numpy.ceil(
            float(self._data_source._size) / self._cache_size))
        self._cache_file_order = self._cache_file_order[
            0:num_of_cache_files]
        self._cache_file_data_orders = self._cache_file_data_orders[
            0:num_of_cache_files]
        if self._data_source._size % self._cache_size != 0:
            self._cache_file_data_orders[num_of_cache_files - 1] = self._cache_file_data_orders[
                num_of_cache_files - 1][0:self._data_source._size % self._cache_size]

        # Create Index
        index_filename = os.path.join(self._cache_dir, "cache_index.csv")
        with open(index_filename, 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            for fn, orders in zip(self._cache_file_names, self._cache_file_data_orders):
                writer.writerow((os.path.basename(fn), len(orders)))
        # Create Info
        if self._cache_file_format == ".npy":
            info_filename = os.path.join(self._cache_dir, "cache_info.csv")
            with open(info_filename, 'w') as f:
                writer = csv.writer(f, lineterminator='\n')
                for variable in self._variables:
                    writer.writerow((variable, ))

        # Create original.csv
        if self._data_source._original_source_uri is not None:
            fr = FileReader(self._data_source._original_source_uri)
            with fr.open() as f:
                csv_lines = [x.decode('utf-8') for x in f.readlines()]
                with open(os.path.join(self._cache_dir, "original.csv"), 'w') as o:
                    for l in csv_lines:
                        o.write(l)

        # Create order.csv
        if self._data_source._order is not None and \
                self._data_source._original_order is not None:
            with open(os.path.join(self._cache_dir, "order.csv"), 'w') as o:
                writer = csv.writer(o, lineterminator='\n')
                for orders in zip(self._data_source._original_order, self._data_source._order):
                    writer.writerow(list(orders))

    def _create_cache_file_position_table(self):
        # Create cached data position table.
        pos = 0
        self._cache_file_start_positions = list(
            range(len(self._cache_file_order)))
        self._order = list(range(len(self._order)))

        self._cache_file_positions = list(range(len(self._order)))
        count = 0
        for i, cache_file_pos in enumerate(self._cache_file_order):
            self._cache_file_start_positions[cache_file_pos] = pos
            pos += len(self._cache_file_data_orders[cache_file_pos])
            for j in self._cache_file_data_orders[cache_file_pos]:
                p = j + (cache_file_pos * self._cache_size)
                self._order[count] = p
                self._cache_file_positions[count] = cache_file_pos
                count += 1

    def __init__(self,
                 data_source,
                 cache_dir=None,
                 cache_file_name_prefix='cache',
                 shuffle=False,
                 rng=None):
        self._tempdir_created = False
        logger.info('Using DataSourceWithFileCache')
        super(DataSourceWithFileCache, self).__init__(shuffle=shuffle, rng=rng)
        self._cache_file_name_prefix = cache_file_name_prefix
        self._cache_dir = cache_dir
        logger.info('Cache Directory is {}'.format(self._cache_dir))

        self._cache_size = int(nnabla_config.get(
            'DATA_ITERATOR', 'data_source_file_cache_size'))
        logger.info('Cache size is {}'.format(self._cache_size))

        self._num_of_threads = int(nnabla_config.get(
            'DATA_ITERATOR', 'data_source_file_cache_num_of_threads'))
        logger.info('Num of thread is {}'.format(self._num_of_threads))

        self._cache_file_format = nnabla_config.get(
            'DATA_ITERATOR', 'cache_file_format')
        logger.info('Cache file format is {}'.format(self._cache_file_format))

        self._thread_lock = threading.Lock()

        self._size = data_source._size
        self._variables = data_source.variables
        self._data_source = data_source
        self._generation = -1
        self._cache_positions = []
        self._total_cached_size = 0
        self._cache_file_names = []
        self._cache_file_order = []
        self._cache_file_start_positions = []
        self._cache_file_data_orders = []

        self._current_cache_file_index = -1
        self._current_cache_data = None

        self.shuffle = shuffle
        self._original_order = list(range(self._size))
        self._order = list(range(self._size))

        # __enter__
        if self._cache_dir is None:
            self._tempdir_created = True
            if nnabla_config.get('DATA_ITERATOR', 'data_source_file_cache_location') != '':
                self._cache_dir = tempfile.mkdtemp(dir=nnabla_config.get(
                    'DATA_ITERATOR', 'data_source_file_cache_location'))
            else:
                self._cache_dir = tempfile.mkdtemp()
            logger.info(
                'Tempdir for cache {} created.'.format(self._cache_dir))
        self._closed = False
        atexit.register(self.close)

        self._create_cache()
        self._create_cache_file_position_table()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def close(self):
        if not self._closed:
            if six.PY3:
                atexit.unregister(self.close)
            if self._tempdir_created:
                # logger.info('Remove created tempdir {}'.format(self._cache_dir))
                rmtree(self._cache_dir, ignore_errors=True)
            self._data_source.close()
            self._closed = True

    def reset(self):
        with self._thread_lock:
            if self._shuffle:
                self._cache_file_order = list(
                    self._rng.permutation(self._cache_file_order))
                for i in range(len(self._cache_file_data_orders)):
                    self._cache_file_data_orders[i] = list(
                        self._rng.permutation(self._cache_file_data_orders[i]))
                self._order = []
                for i in self._cache_file_order:
                    self._order += self._cache_file_data_orders[i]

            self._create_cache_file_position_table()
            self._data_source.reset()
            self._position = 0
            self._generation += 1


class DataSourceWithMemoryCache(DataSource):
    '''
    This class contains properties and methods for data source that can be read from memory cache, which is utilized by data iterator.

    Args:
        data_source (:py:class:`DataSource <nnabla.utils.data_source.DataSource>`):
             Instance of DataSource class which provides data.
        shuffle (bool):
             Indicates whether the dataset is shuffled or not.
        rng (None or :obj:`numpy.random.RandomState`): Numpy random number
            generator.

    '''

    def _get_data_func(self, position):
        return self._data_source._get_data(position)

    def _get_data(self, position):
        if self._on_memory:
            if self._order[position] < len(self._cache):
                data = self._cache[self._order[position]]
            else:
                data = self._get_data_func(position)
                self._cache.append(data)
        else:
            data = self._data_source._get_data(position)
        self._position = position
        return data

    def __init__(self, data_source, shuffle=False, rng=None):
        logger.info('Using DataSourceWithMemoryCache')
        super(DataSourceWithMemoryCache, self).__init__(
            shuffle=shuffle, rng=rng)
        self._buffer_max_size = int(nnabla_config.get(
            'DATA_ITERATOR', 'data_source_buffer_max_size'))
        self._size = data_source._size
        self._variables = data_source.variables
        self._data_source = data_source
        self._order = list(range(self._size))

        self._on_memory = False
        self._cache = []

        data = self._get_data_func(0)
        self._data_size = 0
        for d in data:
            if isinstance(d, list):
                d = numpy.array(d, dtype=numpy.float32)
            self._data_size += d.size * d.itemsize
        total_size = self._data_size * self._size
        if total_size < self._buffer_max_size:
            logger.info('On-memory')
            self._on_memory = True
        self._generation = -1
        self._closed = False
        atexit.register(self.close)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def close(self):
        if not self._closed:
            if six.PY3:
                atexit.unregister(self.close)
            self._data_source.close()
            self._closed = True

    def reset(self):
        if self._on_memory:
            self._generation += 1
            if self._shuffle and self._generation > 0:
                self._order = list(self._rng.permutation(self._size))

            else:
                self._order = list(range(self._size))

            if self._position == 0:
                self._generation = -1
            else:
                self._data_source._position = self._position
                self._data_source.reset()
        else:
            self._data_source.reset()
            self._generation = self._data_source._generation
            self._position = self._data_source._position

        super(DataSourceWithMemoryCache, self).reset()


class SlicedDataSource(DataSource):
    '''
    Provides sliced data source.

    Args:
        data_source (:py:class:`DataSource <nnabla.utils.data_source.DataSource>`):
             Instance of DataSource class which provides data.
    '''

    def __init__(self, data_source, shuffle=False, rng=None, slice_start=None, slice_end=None):
        logger.info('Using SlicedDataSource')
        super(SlicedDataSource, self).__init__(shuffle=shuffle, rng=rng)

        self._data_source = data_source
        self._variables = data_source._variables[:]
        self._slice_start = slice_start
        self._slice_end = slice_end
        self._size = self._slice_end - self._slice_start
        self._generation = -1
        self.reset()

    def reset(self):
        self._data_source.reset()
        self._data_source._position = self._slice_start
        self._generation += 1
        self._position = 0

    def _get_data(self, position):
        self._position = position
        data = self._data_source._get_data(self._slice_start + position)
        return data
