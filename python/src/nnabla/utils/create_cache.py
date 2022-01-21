# Copyright 2019,2020,2021 Sony Corporation.
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
import collections
import csv
import os
import shutil
from contextlib import closing
from multiprocessing.pool import ThreadPool

import h5py
import numpy
from nnabla.config import nnabla_config
from nnabla.logger import logger
from nnabla.utils.data_source_implements import CsvDataSource
from nnabla.utils.data_source_loader import FileReader
from nnabla.utils.progress import progress
from nnabla.utils.communicator_util import single_or_rankzero


class CreateCache(CsvDataSource):
    '''Create dataset cache from local file.

    If you want to create cache data from remote resource, use data_iterator_csv_dataset instead.

    '''

    def _save_cache(self, args):
        position = args[0]
        cache_csv = args[1]
        # conv dataset
        cache_data = [tuple(self._process_row(row)) for row in cache_csv]

        start_position = position + 1 - len(cache_data)
        end_position = position
        cache_filename = os.path.join(
            self._cache_dir, '{}_{:08d}_{:08d}{}'.format(self._cache_file_name_prefix,
                                                         start_position,
                                                         end_position,
                                                         self._cache_file_format))

        logger.info('Creating cache file {}'.format(cache_filename))

        data = collections.OrderedDict(
            [(n, []) for n in self._variables])
        for _, cd in enumerate(cache_data):
            for i, n in enumerate(self._variables):
                if isinstance(cd[i], numpy.ndarray):
                    d = cd[i]
                else:
                    d = numpy.array(cd[i]).astype(numpy.float32)
                data[n].append(d)

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

        self.current_cache_position += 1
        if single_or_rankzero():
            if self.current_cache_position % int(self.num_of_cache_file/20+1) == 0:
                progress('Create cache', self.current_cache_position /
                         self.num_of_cache_file)
        return cache_filename, len(cache_data)

    def __init__(self, input_csv_filename, rng=None, shuffle=False, num_of_threads=None):
        self._cache_size = int(nnabla_config.get(
            'DATA_ITERATOR', 'data_source_file_cache_size'))
        logger.info('Cache size is {}'.format(self._cache_size))

        self._filereader = FileReader(input_csv_filename)
        self._original_source_uri = input_csv_filename
        if rng is None:
            self._rng = numpy.random.RandomState(313)
        else:
            self._rng = rng
        self._shuffle = shuffle

        # read index.csv
        self._file = open(input_csv_filename, 'r', encoding='utf-8-sig')
        csvreader = csv.reader(self._file)

        header = next(csvreader)

        # Store file positions of each data.
        self._csv_data = list(csvreader)
        self._size = len(self._csv_data)

        self._file.close()

        self._remove_comment_cols(header, self._csv_data)
        self._process_header(header)
        self._variables = tuple(self._variables_dict.keys())

        self._original_order = list(range(self._size))

        # Shuffle, the order is processing csv file order
        if self._shuffle:
            self._order = list(
                self._rng.permutation(list(range(self._size))))
        else:
            self._order = list(range(self._size))

        if num_of_threads:
            self._num_of_threads = num_of_threads
        else:
            self._num_of_threads = int(nnabla_config.get(
                'DATA_ITERATOR', 'data_source_file_cache_num_of_threads'))
        logger.info('Num of thread is {}'.format(self._num_of_threads))

    def create(self, output_cache_dirname, normalize=True, cache_file_name_prefix='cache'):

        self._normalize = normalize
        self._cache_file_name_prefix = cache_file_name_prefix
        self._cache_dir = output_cache_dirname

        self._cache_file_format = nnabla_config.get(
            'DATA_ITERATOR', 'cache_file_format')
        logger.info('Cache file format is {}'.format(self._cache_file_format))

        progress(None)

        csv_position_and_data = []
        csv_row = []
        for _position in range(self._size):
            csv_row.append(self._csv_data[self._order[_position]])
            if len(csv_row) == self._cache_size:
                csv_position_and_data.append((_position, csv_row))
                csv_row = []
        if len(csv_row):
            csv_position_and_data.append((self._size-1, csv_row))

        self.num_of_cache_file = len(csv_position_and_data)
        self.current_cache_position = 0
        if single_or_rankzero():
            progress('Create cache', 0)
        with closing(ThreadPool(processes=self._num_of_threads)) as pool:
            cache_index_rows = pool.map(
                self._save_cache, csv_position_and_data)
        if single_or_rankzero():
            progress('Create cache', 1.0)

        # Create Index
        index_filename = os.path.join(output_cache_dirname, "cache_index.csv")
        with open(index_filename, 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            for row in cache_index_rows:
                if row:
                    # row: (file_path, data_nums)
                    writer.writerow((os.path.basename(row[0]), row[1]))

        # Create Info
        if self._cache_file_format == ".npy":
            info_filename = os.path.join(
                output_cache_dirname, "cache_info.csv")
            with open(info_filename, 'w') as f:
                writer = csv.writer(f, lineterminator='\n')
                for variable in self._variables:
                    writer.writerow((variable, ))

        # Create original.csv
        if self._original_source_uri is not None:
            shutil.copy(self._original_source_uri, os.path.join(
                output_cache_dirname, "original.csv"))

        # Create order.csv
        if self._order is not None and \
                self._original_order is not None:
            with open(os.path.join(output_cache_dirname, "order.csv"), 'w') as o:
                writer = csv.writer(o, lineterminator='\n')
                for orders in zip(self._original_order, self._order):
                    writer.writerow(list(orders))
