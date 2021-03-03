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

from __future__ import absolute_import

import os
import csv
import pytest
import tempfile
from shutil import rmtree
from contextlib import contextmanager, closing

# from nnabla
from nnabla.config import nnabla_config
from nnabla.utils.create_cache import CreateCache
from nnabla.utils.data_source_implements import CacheDataSource, CsvDataSource
from nnabla.testing import assert_allclose

from .conftest import test_data_csv_csv_20, test_data_csv_png_20


@contextmanager
def create_temp_with_dir():
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    rmtree(tmpdir, ignore_errors=True)


def associate_variables_and_data(source) -> dict:
    data_dict = {}
    for v, data in zip(source.variables, source.next()):
        data_dict[v] = data
    return data_dict


def check_relative_csv_file_result(cache_file_fmt, csvfilename, cachedir):
    # check cache_index.csv
    cache_info_csv_path = os.path.join(cachedir, 'cache_index.csv')
    assert os.path.exists(cache_info_csv_path)

    with open(cache_info_csv_path, 'r') as f:
        for row in csv.reader(f):
            assert os.path.exists(os.path.join(cachedir, row[0]))

    # check cache_info.csv
    if cache_file_fmt == '.npy':
        assert os.path.exists(os.path.join(cachedir, 'cache_info.csv'))

    # check order.csv
    assert os.path.exists(os.path.join(cachedir, 'order.csv'))

    # check original.csv
    original_csv_path = os.path.join(cachedir, 'original.csv')
    assert os.path.exists(original_csv_path)

    with open(original_csv_path, 'r') as of, open(csvfilename, 'r') as cf:
        for row in of:
            assert row == cf.readline()


@pytest.mark.parametrize('input_file_fmt', ['png', 'csv'])
@pytest.mark.parametrize('cache_file_fmt', ['.npy', '.h5'])
@pytest.mark.parametrize('shuffle', [False, True])
@pytest.mark.parametrize('normalize', [False, True])
@pytest.mark.parametrize('num_of_threads', [i for i in range(10)])
def test_create_cache(test_data_csv_csv_20,
                      test_data_csv_png_20,
                      input_file_fmt,
                      cache_file_fmt,
                      shuffle,
                      normalize,
                      num_of_threads):
    if input_file_fmt == 'csv':
        csvfilename = test_data_csv_csv_20
    else:
        csvfilename = test_data_csv_png_20

    nnabla_config.set('DATA_ITERATOR', 'cache_file_format', cache_file_fmt)

    with create_temp_with_dir() as tmpdir:
        cc = CreateCache(csvfilename, shuffle=shuffle,
                         num_of_threads=num_of_threads)
        cc.create(tmpdir, normalize=normalize)

        # get cache data source and csv file data source
        with closing(CacheDataSource(tmpdir)) as cache_source:
            csv_source = CsvDataSource(csvfilename, normalize=normalize)

            check_relative_csv_file_result(cache_file_fmt, csvfilename, tmpdir)

            assert cache_source.size == csv_source.size
            assert set(cache_source.variables) == set(csv_source.variables)

            if shuffle:
                with open(os.path.join(tmpdir, 'order.csv'), 'r') as f:
                    csv_source._order = [int(row[1]) for row in csv.reader(f)]

            for _ in range(cache_source.size):
                cache_data = associate_variables_and_data(cache_source)
                csv_data = associate_variables_and_data(csv_source)

                for v in cache_source.variables:
                    assert_allclose(cache_data[v], csv_data[v])
