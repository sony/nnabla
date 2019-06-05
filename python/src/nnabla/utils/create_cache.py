import collections
import csv
import numpy
import os
import shutil
import sys


from nnabla.utils.data_source_implements import CsvDataSource
from nnabla.utils.data_source_loader import FileReader

from nnabla.config import nnabla_config
from nnabla.logger import logger
from nnabla.utils.progress import progress


class CreateCache(CsvDataSource):
    '''Create dataset cache from local file.

    If you want to create cache data from remote resource, use data_iterator_csv_dataset instead.

    '''

    def _save_cache(self):
        if len(self._cache_data) > 0:
            start_position = self._position + 1 - len(self._cache_data)
            end_position = self._position
            cache_filename = os.path.join(
                self._cache_dir, '{}_{:08d}_{:08d}{}'.format(self._cache_file_name_prefix,
                                                             start_position,
                                                             end_position,
                                                             self._cache_file_format))

            logger.info('Creating cache file {}'.format(cache_filename))

            data = collections.OrderedDict(
                [(n, []) for n in self._variables])
            for pos, cd in enumerate(self._cache_data):
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
                    is_create_cache_imcomplete = True
                    while is_create_cache_imcomplete:
                        try:
                            with open(cache_filename, 'wb') as f:
                                for v in data.values():
                                    numpy.save(f, v)
                            is_create_cache_imcomplete = False
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
            self._cache_file_data_orders.append(
                list(range(len(self._cache_data))))
            self._cache_positions = []

    def __init__(self, input_csv_filename, rng=None, shuffle=False):
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

        # Binary mode is required to use seek and tell function.
        self._file = open(input_csv_filename, 'rb')

        self._line_positions = []
        line = self._file.readline().decode('utf-8')
        csvreader = csv.reader([line])
        self._process_header(next(csvreader))

        # Store file positions of each data.
        self._size = 0
        while True:
            self._line_positions.append(self._file.tell())
            line = self._file.readline()
            if line is None or len(line) == 0:
                break
            self._size += 1

        # rewind
        self._file.seek(0)

        self._original_order = list(range(self._size))
        self._order = list(range(self._size))
        self._variables = tuple(self._variables_dict.keys())

        # Shuffle
        if self._shuffle:
            self._order = list(
                self._rng.permutation(list(range(self._size))))
        else:
            self._order = list(range(self._size))

    def create(self, output_cache_dirname, normalize=True, cache_file_name_prefix='cache'):

        self._normalize = normalize
        self._cache_file_name_prefix = cache_file_name_prefix

        self._cache_file_format = nnabla_config.get(
            'DATA_ITERATOR', 'cache_file_format')
        logger.info('Cache file format is {}'.format(self._cache_file_format))

        self._cache_dir = output_cache_dirname

        progress(None)

        self._cache_file_order = []
        self._cache_file_data_orders = []
        self._cache_file_names = []

        self._cache_data = []
        percent = 0
        progress('Create cache', 0)
        for self._position in range(self._size):
            if int(self._position * 10.0 / self._size) > percent:
                percent = int(self._position * 10.0 / self._size)
                progress('Create cache', percent / 10.0)
            self._file.seek(self._line_positions[self._order[self._position]])
            line = self._file.readline().decode('utf-8')
            csvreader = csv.reader([line])
            row = next(csvreader)
            self._cache_data.append(tuple(self._process_row(row)))

            if len(self._cache_data) >= self._cache_size:
                self._save_cache()
                self._cache_data = []

        self._save_cache()
        progress('Create cache', 1.0)

        # Adjust data size into reseted position. In most case it means
        # multiple of bunch(mini-batch) size.
        num_of_cache_files = int(numpy.ceil(
            float(self._size) / self._cache_size))
        self._cache_file_order = self._cache_file_order[
            0:num_of_cache_files]
        self._cache_file_data_orders = self._cache_file_data_orders[
            0:num_of_cache_files]
        if self._size % self._cache_size != 0:
            self._cache_file_data_orders[num_of_cache_files - 1] = self._cache_file_data_orders[
                num_of_cache_files - 1][0:self._size % self._cache_size]

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
        if self._original_source_uri is not None:
            shutil.copy(self._original_source_uri, os.path.join(
                self._cache_dir, "original.csv"))

        # Create order.csv
        if self._order is not None and \
                self._original_order is not None:
            with open(os.path.join(self._cache_dir, "order.csv"), 'w') as o:
                writer = csv.writer(o, lineterminator='\n')
                for orders in zip(self._original_order, self._order):
                    writer.writerow(list(orders))
