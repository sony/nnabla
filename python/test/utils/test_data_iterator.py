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

from .conftest import test_data_csv_png_10, test_data_csv_png_20


def check_data_iterator_result(di, batch_size, shuffle, normalize):
    datalist = []
    for data in di:
        for i in range(batch_size):
            if normalize:
                v1 = round(data[0][i].flatten()[0] * 256)
            else:
                v1 = round(data[0][i].flatten()[0])
            v2 = round(data[1][i][0])
            assert v1 == v2
            datalist.append(v1)
        if di.epoch > 0:
            break
    s1 = set(datalist)
    s2 = set(range(di._data_source.size))
    assert len(s1 - s2) == 0
    assert len(s2 - s1) == di._data_source.size - di.size


@pytest.mark.parametrize("batch_size", [5, 7])
@pytest.mark.parametrize("shuffle", [False, True])
def test_data_iterator_simple(test_data_csv_png_10, batch_size, shuffle):
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
    with data_iterator_simple(test_load_func, size, batch_size, shuffle=shuffle) as di:
        check_data_iterator_result(di, batch_size, shuffle, False)


@pytest.mark.parametrize("size", [10, 20])
@pytest.mark.parametrize("batch_size", [5, 7, 23, 73])
@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("with_memory_cache", [False, True])
@pytest.mark.parametrize("with_file_cache", [False, True])
@pytest.mark.parametrize("with_context", [False, True])
def test_data_iterator_csv_dataset(test_data_csv_png_10,
                                   test_data_csv_png_20,
                                   size,
                                   batch_size,
                                   shuffle,
                                   normalize,
                                   with_memory_cache,
                                   with_file_cache,
                                   with_context):

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
                                       with_file_cache=with_file_cache) as di:
            check_data_iterator_result(di, batch_size, shuffle, normalize)
    else:
        di = data_iterator_csv_dataset(uri=csvfilename,
                                       batch_size=batch_size,
                                       shuffle=shuffle,
                                       normalize=normalize,
                                       with_memory_cache=with_memory_cache,
                                       with_file_cache=with_file_cache)
        check_data_iterator_result(di, batch_size, shuffle, normalize)
        di.close()
