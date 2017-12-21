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

import os
import pytest

# from NNabla
from nnabla.utils.data_source_implements import SimpleDataSource, CsvDataSource, ConcatDataSource
from nnabla.utils.data_source_loader import load_image

from .conftest import test_data_csv_csv_10, test_data_csv_csv_20
from .conftest import test_data_csv_png_10, test_data_csv_png_20


@pytest.mark.parametrize("size", [10, 20])
@pytest.mark.parametrize("shuffle", [False, True])
def test_csv_data_source(test_data_csv_csv_10, test_data_csv_csv_20, size, shuffle):

    if size == 10:
        csvfilename = test_data_csv_csv_10
    elif size == 20:
        csvfilename = test_data_csv_csv_20

    cds = CsvDataSource(csvfilename, shuffle)
    cds.reset()
    order = []
    for n in range(0, cds.size):
        data, label = cds.next()
        assert data[0][0] == label[0]
        order.append(int(round(data[0][0])))
    if shuffle:
        assert not list(range(size)) == order
        assert list(range(size)) == sorted(order)
    else:
        assert list(range(size)) == order


@pytest.mark.parametrize("shuffle", [False, True])
def test_simple_data_source(test_data_csv_png_20, shuffle):
    src_data = []
    with open(test_data_csv_png_20) as f:
        for l in f.readlines():
            values = [x.strip() for x in l.split(',')]
            img_file_name = os.path.join(
                os.path.dirname(test_data_csv_png_20), values[0])
            if os.path.exists(img_file_name):
                with open(img_file_name, 'rb') as img_file:
                    d = load_image(img_file)
                    src_data.append((d, int(values[1])))

    def test_load_func(position):
        return src_data[position]

    size = len(src_data)
    ds = SimpleDataSource(test_load_func, size, shuffle=shuffle)
    order = []
    for i in range(ds.size):
        data, label = ds.next()
        assert data[0][0][0] == label
        order.append(label)
    if shuffle:
        assert not list(range(size)) == order
        assert list(range(size)) == sorted(order)
    else:
        assert list(range(size)) == order


@pytest.mark.parametrize("shuffle", [False, True])
def test_concat_data_source(test_data_csv_csv_10, test_data_csv_csv_20, shuffle):
    data_list = [test_data_csv_csv_10, test_data_csv_csv_20]
    ds_list = [CsvDataSource(csvfilename, shuffle)
               for csvfilename in data_list]

    cds = ConcatDataSource(ds_list, shuffle)
    cds.reset()
    order = []
    for n in range(0, cds.size):
        data, label = cds.next()
        assert data[0][0] == label[0]
        order.append(int(round(data[0][0])))
    original_order = list(range(10)) + list(range(20))
    if shuffle:
        assert not original_order == order
        assert sorted(original_order) == sorted(order)
    else:
        assert original_order == order
