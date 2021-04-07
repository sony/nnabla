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
import pytest

# from NNabla
from nnabla.logger import logger
from nnabla.config import nnabla_config
from nnabla.utils.data_source_loader import load_image
from nnabla.utils.data_iterator import data_iterator_csv_dataset
from nnabla.utils.data_iterator import data_iterator_simple
from nnabla.utils.data_iterator import data_iterator_concat_datasets
from nnabla.utils.data_source_implements import CsvDataSource
from .conftest import test_data_csv_png_10, test_data_csv_png_20


def check_data_iterator_concat_result(di, batch_size, normalize, ds1, ds2, stop_exhausted):
    datalist = []
    count = 0
    for data in di:
        for i in range(batch_size):
            count += 1
            if normalize:
                v1 = round(data[0][i].flatten()[0] * 256)
            else:
                v1 = round(data[0][i].flatten()[0])
            v2 = round(data[1][i][0])
            assert v1 == v2
            datalist.append(v1)
        if not stop_exhausted and di.epoch > 0:
            print("epoch=", di.epoch)
            break

    if not stop_exhausted:
        s = sorted(list(range(ds1)) + list(range(ds2)))
        assert len(set(datalist) - set(s)) == 0
        for i in s:
            assert i in datalist
            datalist.remove(i)
    else:
        assert count == di._data_source.size // batch_size * batch_size


def check_data_iterator_result(di, batch_size, shuffle, normalize, stop_exhausted):
    datalist = []
    count = 0
    for data in di:
        for i in range(batch_size):
            count += 1
            if normalize:
                v1 = round(data[0][i].flatten()[0] * 256)
            else:
                v1 = round(data[0][i].flatten()[0])
            v2 = round(data[1][i][0])
            assert v1 == v2
            datalist.append(v1)
        if not stop_exhausted and di.epoch > 0:
            print("epoch=", di.epoch)
            break

    if not stop_exhausted:
        s1 = set(datalist)
        s2 = set(range(di._data_source.size))
        assert len(s1 - s2) == 0
        assert len(s2 - s1) == di._data_source.size - di.size
    else:
        assert count == di._data_source.size // batch_size * batch_size


@pytest.mark.parametrize("batch_size", [5, 7])
@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize('stop_exhausted', [False, True])
def test_data_iterator_simple(test_data_csv_png_10, batch_size, shuffle, stop_exhausted):
    src_data = []
    with open(test_data_csv_png_10) as f:
        for l in f.readlines():
            values = [x.strip() for x in l.split(',')]
            img_file_name = os.path.join(
                os.path.dirname(test_data_csv_png_10), values[0])
            if os.path.exists(img_file_name):
                with open(img_file_name, 'rb') as img_file:
                    d = load_image(img_file)
                    src_data.append((d, [int(values[1])]))

    def test_load_func(position):
        return src_data[position]

    size = len(src_data)
    with data_iterator_simple(test_load_func, size, batch_size, shuffle=shuffle, stop_exhausted=stop_exhausted) as di:
        check_data_iterator_result(
            di, batch_size, shuffle, False, stop_exhausted)


@pytest.mark.parametrize("size", [10, 20])
@pytest.mark.parametrize("batch_size", [5, 7, 23, 73])
@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("use_thread", [True, False])
@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("with_memory_cache", [True, False])
@pytest.mark.parametrize("with_file_cache", [True, False])
@pytest.mark.parametrize("with_context", [True, False])
@pytest.mark.parametrize('stop_exhausted', [True, False])
def test_data_iterator_csv_dataset(test_data_csv_png_10,
                                   test_data_csv_png_20,
                                   size,
                                   batch_size,
                                   shuffle,
                                   use_thread,
                                   normalize,
                                   with_memory_cache,
                                   with_file_cache,
                                   with_context,
                                   stop_exhausted):

    nnabla_config.set('DATA_ITERATOR', 'data_source_file_cache_size', '3')
    nnabla_config.set(
        'DATA_ITERATOR', 'data_source_buffer_max_size', '10000')
    nnabla_config.set(
        'DATA_ITERATOR', 'data_source_buffer_num_of_data', '9')

    if size == 10:
        csvfilename = test_data_csv_png_10
    elif size == 20:
        csvfilename = test_data_csv_png_20

    logger.info(csvfilename)

    if with_context:
        with data_iterator_csv_dataset(uri=csvfilename,
                                       batch_size=batch_size,
                                       shuffle=shuffle,
                                       normalize=normalize,
                                       with_memory_cache=with_memory_cache,
                                       with_file_cache=with_file_cache,
                                       use_thread=use_thread,
                                       stop_exhausted=stop_exhausted) as di:
            check_data_iterator_result(
                di, batch_size, shuffle, normalize, stop_exhausted)
    else:
        di = data_iterator_csv_dataset(uri=csvfilename,
                                       batch_size=batch_size,
                                       shuffle=shuffle,
                                       normalize=normalize,
                                       with_memory_cache=with_memory_cache,
                                       with_file_cache=with_file_cache,
                                       use_thread=use_thread,
                                       stop_exhausted=stop_exhausted)
        check_data_iterator_result(
            di, batch_size, shuffle, normalize, stop_exhausted)
        di.close()


@pytest.mark.parametrize("batch_size", [5, 7, 23, 73])
@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("use_thread", [True, False])
@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("with_memory_cache", [True, False])
@pytest.mark.parametrize("with_file_cache", [True, False])
@pytest.mark.parametrize("with_context", [True, False])
@pytest.mark.parametrize('stop_exhausted', [True, False])
def test_data_iterator_concat_datasets(test_data_csv_png_10,
                                       test_data_csv_png_20,
                                       batch_size,
                                       shuffle,
                                       use_thread,
                                       normalize,
                                       with_memory_cache,
                                       with_file_cache,
                                       with_context,
                                       stop_exhausted):

    nnabla_config.set('DATA_ITERATOR', 'data_source_file_cache_size', '3')
    nnabla_config.set(
        'DATA_ITERATOR', 'data_source_buffer_max_size', '10000')
    nnabla_config.set(
        'DATA_ITERATOR', 'data_source_buffer_num_of_data', '9')

    csvfilename_1 = test_data_csv_png_10
    csvfilename_2 = test_data_csv_png_20

    ds1 = CsvDataSource(csvfilename_1,
                        shuffle=shuffle,
                        normalize=normalize)

    ds2 = CsvDataSource(csvfilename_2,
                        shuffle=shuffle,
                        normalize=normalize)

    if with_context:
        with data_iterator_concat_datasets([ds1, ds2],
                                           batch_size=batch_size,
                                           shuffle=shuffle,
                                           with_memory_cache=with_memory_cache,
                                           with_file_cache=with_file_cache,
                                           use_thread=use_thread,
                                           stop_exhausted=stop_exhausted) as di:
            check_data_iterator_concat_result(
                di, batch_size, normalize, ds1.size, ds2.size, stop_exhausted)
    else:
        di = data_iterator_concat_datasets([ds1, ds2],
                                           batch_size=batch_size,
                                           shuffle=shuffle,
                                           with_memory_cache=with_memory_cache,
                                           with_file_cache=with_file_cache,
                                           use_thread=use_thread,
                                           stop_exhausted=stop_exhausted)
        check_data_iterator_concat_result(
            di, batch_size, normalize, ds1.size, ds2.size, stop_exhausted)
        di.close()
