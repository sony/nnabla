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

'''data_source_implements
'''


import atexit
import csv
import os
import threading
from collections import OrderedDict, deque
from time import sleep

import numpy
from nnabla.config import nnabla_config
from nnabla.logger import logger
from nnabla.utils.communicator_util import current_communicator
from six.moves import queue

from .data_source import DataSource
from .data_source_loader import FileReader, load


class SimpleDataSource(DataSource):
    '''SimpleDataSource

    Get data from user defined function.

    '''

    def _get_data(self, position):
        return self._get_data_of_generation(position, 0)

    def _get_data_of_generation(self, position, generation):
        index = None
        for s in self._orders:
            if s.generation == generation:
                index = s.order
                break
        if index is None:
            raise ValueError("Its client is not synchronized!")
        return self._load_func(index[position])

    def apply_order(self):
        self._generation += 1
        self._generate_order(self._generation)

    def _generate_order(self, generation):
        for s in self._orders:
            if s.generation == generation:
                return

        class Snapshot:
            pass
        snapshot = Snapshot()
        snapshot.order = self._rng.permutation(
                self._size) if self._shuffle else numpy.arange(self._size)
        snapshot.generation = generation
        self._orders.append(snapshot)

    def reset(self):
        super(SimpleDataSource, self).reset()

    def __init__(self, load_func, num_examples, shuffle=False, rng=None):
        super(SimpleDataSource, self).__init__(shuffle=shuffle, rng=rng)
        self._load_func = load_func
        self._size = num_examples
        self._generation = 0
        self._variables = ['x' + str(x)
                           for x in range(len(self._load_func(0)))]
        # We set maximal unsynchnonized generations.
        self._orders = deque(maxlen=15)
        self.reset()
        self._generate_order(0)


class CachePrefetcher(object):
    def __init__(self, cachedir, variables):
        self._lock = threading.Lock()
        self._q = queue.Queue()
        self.file_name = None
        self._cachedir = cachedir
        self._variables = variables
        self._filereader = FileReader(self._cachedir)
        self._current_data = None
        self._thread = threading.Thread(target=self._worker)
        self._thread.setDaemon(True)
        self._thread.start()
        self._closed = False
        atexit.register(self.close)

    def read_cache(self, file_name, variables):
        retry = 1
        while True:
            if retry > 10:
                logger.log(99, 'read_cache() retry count over give up.')
                logger.log(
                    99, 'Cache file {} not found. pid={}'.format(file_name, os.getpid()))
                logger.log(99, 'Fatal Error! send SIGKILL to myself.')
                os.kill(os.getpid(), 9)

            result = {}
            try:
                with FileReader(file_name).open(textmode=False) as f:
                    for v in variables:
                        result[v] = numpy.load(f, allow_pickle=True)
                if set(result.keys()) == set(variables):
                    break
                else:
                    logger.log(
                        99, 'read_cache() fails retrying count {}/10.'.format(retry))
                    retry += 1
                    sleep(0.5)
            except:
                logger.log(
                    99, 'Cache file {} not found, retry count {}.'.format(file_name, retry))
                retry += 1
                sleep(0.5)

        return result

    def _worker(self):
        while True:
            sleep(0.001)
            cache_file_name = self._q.get()
            self._current_data = {}
            if cache_file_name is None:
                self._q.task_done()
                break
            self._current_data = self.read_cache(
                cache_file_name, self._variables)
            self._q.task_done()

    def request(self, cache_file_name):
        self.file_name = cache_file_name
        self._q.put(cache_file_name)

    def check_if_hit(self, fn):
        self._lock.acquire()
        if fn == self.file_name:
            self.file_name = None
            self._lock.release()
            return True
        self._lock.release()
        return False

    def read(self):
        self._q.join()
        result = self._current_data
        self.file_name = None
        self._current_data = None
        return result

    def close(self):
        if not self._closed:
            self._q.join()
            self._q.put(None)
            self._q.join()
            self._closed = True


class CacheReaderWithPrefetch(object):
    def __init__(self, cachedir, num_threads, variables):
        self._variables = variables
        self._cache_prefetchers = [CachePrefetcher(
            cachedir, variables) for _ in range(num_threads)]
        self._closed = False
        atexit.register(self.close)

    def open_and_prefetch_cache(self, file_name, file_names_to_prefetch):
        cp_file_names = [cf.file_name for cf in self._cache_prefetchers]
        # print('cp files', cp_file_names)
        result = None
        for cf in self._cache_prefetchers:
            if cf.check_if_hit(file_name):
                result = cf.read()
                break
        if not result:
            # print("no hit", file_name)
            result = cf.read_cache(file_name, self._variables)
        cp_file_names = [cf.file_name for cf in self._cache_prefetchers]
        for i, fn in enumerate(cp_file_names):
            if fn and fn not in file_names_to_prefetch:
                self._cache_prefetchers[i].read()  # waste prefetched cache
                # print("wasted", fn)
        for fn in file_names_to_prefetch:
            if fn not in cp_file_names:
                try:
                    index = cp_file_names.index(None)
                    cp_file_names[index] = fn
                    self._cache_prefetchers[index].request(
                        cp_file_names[index])
                except:
                    continue
        return result

    def close(self):
        if not self._closed:
            for cf in self._cache_prefetchers:
                cf.close()
            self._closed = True


class CacheDataSource(DataSource):
    '''
    Get data from file cache directly.
    '''

    def _get_next_data(self, filename, file_names_to_prefetch, retry=1):
        if retry > 10:
            logger.log(99, '_get_next_data() retry count over give up.')
            raise
        if self._cache_type == '.npy':
            next_data = self._cache_reader_with_prefetch.open_and_prefetch_cache(
                filename, file_names_to_prefetch)
        else:
            # h5 format
            next_data = {}
            with self._filereader.open_cache(filename) as cache:
                for k, v in cache.items():
                    next_data[k] = v[()]

        if current_communicator():
            if set(self._variables) != set(next_data.keys()):
                logger.log(99, '_get_next_data() fails at worker {} retrying count {}/10.'.format(
                    current_communicator().rank, retry))
                sleep(0.01)
                return self._get_next_data(filename, file_names_to_prefetch, retry+1)
        return next_data

    def _get_data(self, position):

        self._position = position
        if current_communicator():
            try:
                filename, index = self._order[position]
            except IndexError:
                logger.log(99, '_get_data() fails at worker {} retrying.'.format(
                    current_communicator().rank))
                sleep(0.01)
                return self._get_data(position)
        else:
            filename, index = self._order[position]

        if filename != self._current_filename:
            file_names_to_prefetch = None
            if self._cache_type == ".npy" and self._num_of_threads > 0:
                file_names_to_prefetch = [o[0] for o in self._order[position + self._max_length:position + self._max_length *
                                                                    self._num_of_threads:self._max_length]]

            self._current_data = self._get_next_data(
                filename, file_names_to_prefetch)
            self._current_filename = filename

        data = [self._current_data[v][index] for v in self.variables]

        if self._normalize:
            new_data = []
            for d in data:
                if d.dtype == numpy.uint8:
                    d = d.astype(numpy.float32) * (1.0 / 255.0)
                elif d.dtype == numpy.uint16:
                    d = d.astype(numpy.float32) * (1.0 / 65535.0)
                new_data.append(d)
            data = new_data
        return data

    def initialize_cache_files(self, filename):
        length = -1
        with self._filereader.open_cache(filename) as cache:

            # Check variables.
            if self._variables is None:
                self._variables = list(cache.keys())
            else:
                if current_communicator():
                    if not set(self._variables) == set(cache.keys()):
                        logger.log(99, 'Error at worker {} {} {}'.format(
                            current_communicator().rank, set(self._variables), set(cache.keys())))
                        raise

            for k, v in cache.items():
                if length < 0:
                    length = len(v)
                else:
                    assert (length == len(v))
            self._cache_files.append((filename, length))
            logger.info('{} {}'.format(filename, length))
            if length > self._max_length:
                self._max_length = length

    def initialize_cache_files_with_index(self, index_filename):
        self._filenames = []
        self._cache_files = []
        try:
            with FileReader(index_filename).open(textmode=True) as f:
                reader = csv.reader(f)
                for row in reader:
                    file_name = os.path.join(self._cachedir, row[0])
                    self._filenames.append(file_name)
                    length = int(row[1])
                    self._cache_files.append((file_name, length))
                    if length > self._max_length:
                        self._max_length = length
                    if self._variables is None:
                        with self._filereader.open_cache(file_name) as cache:
                            # Check variables.
                            self._variables = list(cache.keys())
        except:
            self._filenames = [f for f in self._filereader.listdir() if os.path.splitext(f)[
                1].lower() == ".h5"]
            for filename in self._filenames:
                self.initialize_cache_files(filename)

    def initialize_cache_info(self, info_filename):
        try:
            with FileReader(info_filename).open(textmode=True) as f:
                self._variables = []
                reader = csv.reader(f)
                for row in reader:
                    self._variables.append(row[0])
            self._cache_type = '.npy'
        except:
            self._cache_type = '.h5'

    def __init__(self, cachedir, shuffle=False, rng=None, normalize=False):
        super(CacheDataSource, self).__init__(shuffle=shuffle, rng=rng)

        self._current_data = {}
        self._current_filename = None

        self._cachedir = cachedir
        self._normalize = normalize
        self._filereader = FileReader(self._cachedir)
        self._num_of_threads = int(nnabla_config.get(
            'DATA_ITERATOR', 'cache_file_cache_num_of_threads'))
        self._variables = None

        self._generation = -1
        self._cache_files = []
        self._max_length = 1

        info_filename = os.path.join(self._cachedir, "cache_info.csv")
        self.initialize_cache_info(info_filename)

        index_filename = os.path.join(self._cachedir, "cache_index.csv")
        self.initialize_cache_files_with_index(index_filename)

        logger.info('{}'.format(len(self._cache_files)))

        self._cache_reader_with_prefetch = CacheReaderWithPrefetch(
            self._cachedir, self._num_of_threads, self._variables)
        self._thread_lock = threading.Lock()

        self._original_order = []
        for i in range(len(self._cache_files)):
            filename, length = self._cache_files[i]
            for j in range(length):
                self._original_order.append((filename, j))

        self.reset()

    def close(self):
        if hasattr(self, '_cache_reader_with_prefetch') and self._cache_reader_with_prefetch:
            self._cache_reader_with_prefetch.close()
            self._cache_reader_with_prefetch = None

    def reset(self):
        with self._thread_lock:
            super(CacheDataSource, self).reset()

            self._order = []

            if self._shuffle:
                for i in list(self._rng.permutation(list(range(len(self._cache_files))))):
                    filename, length = self._cache_files[i]
                    for j in list(self._rng.permutation(list(range(length)))):
                        self._order.append((filename, j))
            else:
                for i in range(len(self._cache_files)):
                    filename, length = self._cache_files[i]
                    for j in range(length):
                        self._order.append((filename, j))

            self._current_data = {}
            self._current_filename = None

            self._size = len(self._order)
            self._generation += 1


class CsvDataSource(DataSource):
    '''
    '''

    def _remove_comment_cols(self, header, rows):
        for col_index in reversed(range(len(header))):
            if header[col_index][0] == '#':
                del header[col_index]
                for row in rows:
                    del row[col_index]

    def _process_header(self, row):
        self._variables_dict = OrderedDict()
        self._columns = []
        for column, column_value in enumerate(row):

            # Analyze header "NAME[__INDEX][:LABELNAME]"
            # TODO: use regex instead of split....
            try:
                variable_with_index, label = column_value.split(':', 1)
            except:
                label = None
                variable_with_index = column_value
            try:
                variable, index = variable_with_index.split('__', 1)
            except:
                variable = variable_with_index
                index = None

            self._columns.append((variable, index, label))
            if index is None:
                self._variables_dict[variable] = {
                    'label': label, 'value': None}
            else:
                if variable not in self._variables_dict:
                    self._variables_dict[variable] = []
                self._variables_dict[variable].append(
                    {'label': label, 'value': None})

    def _process_row(self, row):
        values = OrderedDict()
        if len(row) == len(self._columns):
            for column, column_value in enumerate(row):
                variable, index, label = self._columns[column]
                if index is None:
                    values[variable] = self._get_value(
                        column_value, is_vector=True)
                else:
                    if variable not in values:
                        values[variable] = []
                    values[variable].append(self._get_value(column_value))
        return values.values()

    def _get_value(self, value, is_vector=False):
        try:
            if is_vector:
                value = [float(value)]
            else:
                value = float(value)
            return value
        except ValueError:
            pass
        ext = (os.path.splitext(value)[1]).lower()
        with self._filereader.open(value) as f:
            value = load(ext)(f, normalize=self._normalize)
        return value

    def _get_data(self, position):
        return tuple(self._process_row(self._rows[self._order[position]]))

    def __init__(self, filename, shuffle=False, rng=None, normalize=False):
        super(CsvDataSource, self).__init__(shuffle=shuffle, rng=rng)
        self._filename = filename
        self._normalize = normalize

        # Store contents of CSV file into the self._rows list.
        self._generation = -1
        self._rows = []
        self._filereader = FileReader(self._filename)
        with self._filereader.open(textmode=True, encoding='utf-8-sig') as f:
            csvreader = csv.reader(f)
            header = next(csvreader)
            self._rows = list(csvreader)
            self._size = len(self._rows)
            self._remove_comment_cols(header, self._rows)
            self._process_header(header)
        self._original_source_uri = self._filename
        self._original_order = list(range(self._size))
        self._order = list(range(self._size))
        self._variables = tuple(self._variables_dict.keys())
        self.reset()

    def reset(self):
        if self._shuffle:
            logger.debug('Shuffle start.')
            self._order = list(
                self._rng.permutation(list(range(self._size))))
            logger.debug('Shuffle end.')
        self._generation += 1
        super(CsvDataSource, self).reset()


class ConcatDataSource(DataSource):
    '''ConcatDataSource

    Wrapper DataSource for Multiple DataSources.

    '''

    def __init__(self, data_source_list, shuffle=True, rng=None):
        super(ConcatDataSource, self).__init__(shuffle=shuffle, rng=rng)
        self._data_sources = data_source_list

        self._sw_points = list(map(
            lambda s: sum(
                [x.size for x in data_source_list[:data_source_list.index(s) + 1]]),
            data_source_list))  # Switching DataSource index

        self._size = self._sw_points[-1]
        self._variables = data_source_list[0].variables
        self.reset()

    def _get_data(self, position):
        idx = self._indexes[position]
        for i, data_bound in enumerate(self._sw_points):
            if idx < data_bound:
                _idx = idx - self._sw_points[i - 1] if i > 0 else idx
                return self._data_sources[i]._get_data(_idx)
        return None

    def reset(self):
        # reset method initialize self._indexes
        if self._shuffle:
            self._indexes = self._rng.permutation(self._size)
        else:
            self._indexes = numpy.arange(self._size)
        super(ConcatDataSource, self).reset()
