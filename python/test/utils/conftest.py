# Copyright 2017,2018,2019,2020,2021 Sony Corporation.
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
from __future__ import print_function

import pytest

from contextlib import contextmanager
from shutil import rmtree
import numpy
import os
import tempfile
import warnings

from nnabla.utils.image_utils import imsave


@contextmanager
def create_temp_with_dir(filename):
    tmpdir = tempfile.mkdtemp()
    print('created {}'.format(tmpdir))
    csvfilename = os.path.join(tmpdir, filename)
    yield csvfilename
    rmtree(tmpdir, ignore_errors=True)
    print('deleted {}'.format(tmpdir))


@contextmanager
def generate_csv_csv(filename, num_of_data, data_size):
    with create_temp_with_dir(filename) as csvfilename:
        datadir = os.path.dirname(csvfilename)
        with open(csvfilename, 'w') as f:
            f.write('x:data, y\n')
            for n in range(0, num_of_data):
                x = numpy.ones(data_size).astype(numpy.uint8) * n
                data_name = 'data_{}.csv'.format(n)
                with open(os.path.join(datadir, data_name), 'w') as df:
                    for d in x:
                        df.write('{}\n'.format(d))
                f.write('{}, {}\n'.format(data_name, n))
        yield csvfilename


@contextmanager
def generate_csv_png(filename, num_of_data, img_size):
    with create_temp_with_dir(filename) as csvfilename:
        imgdir = os.path.dirname(csvfilename)
        with open(csvfilename, 'w') as f:
            f.write('x:image, y\n')
            for n in range(0, num_of_data):
                x = numpy.identity(img_size).astype(numpy.uint8) * n
                img_name = 'image_{}.png'.format(n)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    imsave(os.path.join(imgdir, img_name), x)
                f.write('{}, {}\n'.format(img_name, n))
        yield csvfilename


@pytest.fixture(scope="module", autouse=True)
def test_data_csv_png_10():
    with generate_csv_png('test.csv', 10, 14) as csvfilename:
        yield csvfilename


@pytest.fixture(scope="module", autouse=True)
def test_data_csv_png_20():
    with generate_csv_png('test.csv', 20, 14) as csvfilename:
        yield csvfilename


@pytest.fixture(scope="module", autouse=True)
def test_data_csv_csv_10():
    with generate_csv_csv('test.csv', 10, 14) as csvfilename:
        yield csvfilename


@pytest.fixture(scope="module", autouse=True)
def test_data_csv_csv_20():
    with generate_csv_csv('test.csv', 20, 14) as csvfilename:
        yield csvfilename


@contextmanager
def generate_cache_dir(num_of_data):
    cache_block_size = 100
    with create_temp_with_dir('dummy.csv') as csvfilename:
        imgdir = os.path.dirname(csvfilename)

        # generate data
        data_x = numpy.random.random((num_of_data, 2, 2))
        data_y = numpy.random.randint(0, 10, size=(num_of_data,))

        cache_info = []
        for i in range(0, num_of_data, cache_block_size):
            npy_fn = os.path.join(imgdir,
                                  "cache_{}_{}.npy".format(
                                      str(i).zfill(8), str(i + cache_block_size - 1).zfill(8)))
            with open(npy_fn, 'wb') as f:
                numpy.save(f, data_x[i: i + cache_block_size - 1])
                numpy.save(f, data_y[i: i + cache_block_size - 1])

            cache_info.append(
                (npy_fn, len(data_y[i: i + cache_block_size - 1])))

        # generate cache_index.csv
        with open(os.path.join(imgdir, 'cache_index.csv'), 'w') as f:
            for fn, num in cache_info:
                f.write("{},{}\n".format(fn, str(num)))

        # generate cache_info.csv
        with open(os.path.join(imgdir, 'cache_info.csv'), 'w') as f:
            f.write('x\n')
            f.write('y\n')

        # generate order.csv
        with open(os.path.join(imgdir, 'order.csv'), 'w') as f:
            for i in range(num_of_data):
                f.write('{},{}\n'.format(i, i))

        print("start testing...")
        yield imgdir
        print("finished testing")
