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

'''data_source_implements
'''

from collections import OrderedDict
import csv
import numpy
import os
import struct
import zlib

from .data_source import DataSource
from .data_source_loader import FileReader, load
from nnabla.logger import logger


class SimpleDataSource(DataSource):
    '''SimpleDataSource

    Get data from user defined function.

    '''

    def _get_data(self, position):
        return self._load_func(self._order[position])

    def reset(self):
        pass

    def __init__(self, load_func, num_examples, shuffle=False, rng=None):
        super(SimpleDataSource, self).__init__(shuffle=shuffle, rng=rng)
        super(SimpleDataSource, self).reset()
        self._load_func = load_func
        self._size = num_examples
        self._variables = ['x' + str(x)
                           for x in range(len(self._load_func(0)))]
        if shuffle:
            self._order = list(
                numpy.random.permutation(list(range(self._size))))
        else:
            self._order = list(range(self._size))


class CacheDataSource(DataSource):
    '''
    '''

    def _get_data(self, position):
        self._position = position
        filename, index = self._order[position]
        if filename != self._current_filename:
            self._current_filename = filename
            self._current_data = {}
            with self._filereader.open_cache(self._current_filename) as cache:
                for k, v in cache.items():
                    self._current_data[k] = v.value
        data = [self._current_data[v][index] for v in self.variables]
        if self._normalize:
            data = [d.astype(numpy.float32) * (1.0 / 255.0)
                    if d.dtype == numpy.uint8 else d for d in data]
        return data

    def __init__(self, cachedir, shuffle=False, rng=None, normalize=False):
        super(CacheDataSource, self).__init__(shuffle=shuffle, rng=rng)
        self._cachedir = cachedir
        self._normalize = normalize
        self._filereader = FileReader(self._cachedir)
        self._filenames = self._filereader.listdir()

        self._generation = -1
        self._cache_files = []
        for filename in self._filenames:
            length = -1
            with self._filereader.open_cache(filename) as cache:
                if self._variables is None:
                    self._variables = list(cache.keys())
                for k, v in cache.items():
                    if length < 0:
                        length = len(v)
                    else:
                        assert(length == len(v))
                self._cache_files.append((filename, length))
                logger.info('{} {}'.format(filename, length))

        logger.info('{}'.format(len(self._cache_files)))
        self.reset()

    def reset(self):
        super(CacheDataSource, self).reset()
        self._order = []

        if self._shuffle:
            for i in list(numpy.random.permutation(list(range(len(self._cache_files))))):
                filename, length = self._cache_files[i]
                for j in list(numpy.random.permutation(list(range(length)))):
                    self._order.append((filename, j))
        else:
            for i in range(len(self._cache_files)):
                filename, length = self._cache_files[i]
                for j in range(length):
                    self._order.append((filename, j))

        self._current_filename = None
        self._current_data = {}
        self._size = len(self._order)
        self._generation += 1


class CsvDataSource(DataSource):
    '''
    '''

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
        except:
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
        with self._filereader.open() as f:
            csv_lines = [x.decode('utf-8') for x in f.readlines()]
            csvreader = csv.reader(csv_lines)
            first_line = True
            for row in csvreader:
                if first_line:
                    self._process_header(row)
                    first_line = False
                else:
                    self._rows.append(row)
                    self._size += 1
        self._order = list(range(self._size))
        self._variables = tuple(self._variables_dict.keys())
        self.reset()

    def reset(self):
        if self._shuffle:
            logger.debug('Shuffle start.')
            self._order = list(
                numpy.random.permutation(list(range(self._size))))
            logger.debug('Shuffle end.')
        self._generation += 1
        super(CsvDataSource, self).reset()
