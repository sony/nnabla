import os
import csv
import gc
import h5py
import numpy
import shutil
import collections
import multiprocessing
from contextlib import closing
from types import SimpleNamespace

from nnabla.utils.data_source_implements import CsvDataSource
from nnabla.utils.data_source_loader import FileReader, load

from nnabla.config import nnabla_config
from nnabla.logger import logger
from nnabla.utils.progress import progress


def multiprocess_save_cache(create_cache_args):

    def _process_row(row, args):
        def _get_value(value, is_vector=False):
            try:
                if is_vector:
                    value = [float(value)]
                else:
                    value = float(value)
                return value
            except ValueError:
                pass
            ext = (os.path.splitext(value)[1]).lower()
            with args._filereader.open(value) as f:
                value = load(ext)(f, normalize=args._normalize)
            return value

        values = collections.OrderedDict()
        if len(row) == len(args._columns):
            for column, column_value in enumerate(row):
                variable, index, label = args._columns[column]
                if index is None:
                    values[variable] = _get_value(
                        column_value, is_vector=True)
                else:
                    if variable not in values:
                        values[variable] = []
                    values[variable].append(_get_value(column_value))
        return values.values()

    (position, cache_csv), cc_args = create_cache_args
    cc_args = SimpleNamespace(**cc_args)
    cache_data = []
    for row in cache_csv:
        cache_data.append(tuple(_process_row(row, cc_args)))

    if len(cache_data) > 0:
        start_position = position + 1 - len(cache_data)
        end_position = position
        cache_filename = os.path.join(
            cc_args._output_cache_dirname,
            '{}_{:08d}_{:08d}{}'.format(cc_args._cache_file_name_prefix,
                                        start_position,
                                        end_position,
                                        cc_args._cache_file_format))

        logger.info('Creating cache file {}'.format(cache_filename))

        data = collections.OrderedDict(
            [(n, []) for n in cc_args._variables])
        for _, cd in enumerate(cache_data):
            for i, n in enumerate(cc_args._variables):
                if isinstance(cd[i], numpy.ndarray):
                    d = cd[i]
                else:
                    d = numpy.array(cd[i]).astype(numpy.float32)
                data[n].append(d)
        try:
            if cc_args._cache_file_format == ".h5":
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

        cc_args._cache_file_name_and_data_nums_list.append(
            (cache_filename, len(cache_data)))
        progress(
            'Create cache',
            len(cc_args._cache_file_name_and_data_nums_list) / cc_args._cache_file_count)


class CreateCache(CsvDataSource):
    '''Create dataset cache from local file.

    If you want to create cache data from remote resource, use data_iterator_csv_dataset instead.

    '''

    def __init__(self, input_csv_filename, rng=None, shuffle=False, process_num=None):
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
        self._file = open(input_csv_filename, 'r')
        csvreader = csv.reader(self._file)

        self._process_header(next(csvreader))
        self._variables = tuple(self._variables_dict.keys())

        # Store file positions of each data.
        self._csv_data = list(csvreader)
        self._size = len(self._csv_data)

        self._file.close()

        self._original_order = list(range(self._size))

        # Shuffle, the order is processing csv file order
        if self._shuffle:
            self._order = list(
                self._rng.permutation(list(range(self._size))))
        else:
            self._order = list(range(self._size))

        # multiprocess num
        if process_num:
            self._process_num = process_num
        else:
            self._process_num = multiprocessing.cpu_count()
        logger.info('Num of process is {}'.format(self._process_num))

    def create(self, output_cache_dirname, normalize=True, cache_file_name_prefix='cache'):

        cache_file_format = nnabla_config.get(
            'DATA_ITERATOR', 'cache_file_format')
        logger.info('Cache file format is {}'.format(cache_file_format))

        progress(None)

        cache_file_name_and_data_nums_list = multiprocessing.Manager().list()

        csv_position_and_data = []
        csv_row = []
        for _position in range(self._size):
            csv_row.append(self._csv_data[self._order[_position]])
            if len(csv_row) == self._cache_size:
                csv_position_and_data.append((_position, csv_row))
                csv_row = []
        if len(csv_row):
            csv_position_and_data.append((self._size-1, csv_row))

        self_args = {
            '_cache_file_name_prefix': cache_file_name_prefix,
            '_cache_file_format': cache_file_format,
            '_cache_file_name_and_data_nums_list': cache_file_name_and_data_nums_list,
            '_output_cache_dirname': output_cache_dirname,
            '_variables': self._variables,
            '_filereader': self._filereader,
            '_normalize': normalize,
            '_columns': self._columns,
            '_cache_file_count': len(csv_position_and_data)
        }

        # Notice:
        #   Here, we have to place a gc.collect(), since we found
        #   python might perform garbage collection operation in
        #   a child process, which tends to release some objects
        #   created by its parent process, thus, it might touch
        #   cuda APIs which has not initialized in child process.
        #   Place a gc.collect() here can avoid such cases.
        gc.collect()

        progress('Create cache', 0)
        with closing(multiprocessing.Pool(self._process_num)) as pool:
            pool.map(multiprocess_save_cache,
                     ((i, self_args) for i in csv_position_and_data))
        progress('Create cache', 1.0)

        logger.info('The total of cache files is {}'.format(
                    len(cache_file_name_and_data_nums_list)))

        # Create Index
        index_filename = os.path.join(output_cache_dirname, "cache_index.csv")
        cache_index_rows = sorted(
            cache_file_name_and_data_nums_list, key=lambda x: x[0])
        with open(index_filename, 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            for file_name, data_nums in cache_index_rows:
                writer.writerow((os.path.basename(file_name), data_nums))

        # Create Info
        if cache_file_format == ".npy":
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
