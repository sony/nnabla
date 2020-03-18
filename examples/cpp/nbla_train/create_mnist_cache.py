# Copyright (c) 2018 Sony Corporation. All Rights Reserved.
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

import argparse
import os
import sys

from nnabla.logger import logger
from nnabla.config import nnabla_config
from nnabla.utils.data_source import DataSourceWithFileCache

parser = argparse.ArgumentParser(description='Create mnist cache.')
parser.add_argument('-f', '--file_cache', action='store_true',
                    help='Use file cache')
parser.add_argument('-s', '--cache_size', type=int, default=100,
                    help='Cache size (num of data).')
parser.add_argument('-o', '--output', type=str, default='cache',
                    help='If specified, cache data will output to here.')
args = parser.parse_args()

logger.debug('file_cache: {}'.format(args.file_cache))
logger.debug('cache_size: {}'.format(args.cache_size))
logger.debug('output: {}'.format(args.output))

nnabla_config.set('DATA_ITERATOR', 'data_source_file_cache_size',
                  '{}'.format(args.cache_size))
nnabla_config.set('DATA_ITERATOR', 'cache_file_format', '.h5')

HERE = os.path.dirname(__file__)
nnabla_examples_root = os.path.join(HERE, '../../../../nnabla-examples')
mnist_examples_root = os.path.realpath(
    os.path.join(nnabla_examples_root, 'mnist-collection'))
sys.path.append(mnist_examples_root)

from mnist_data import MnistDataSource
mnist_training_cache = args.output + '/mnist_training.cache'
if not os.path.exists(mnist_training_cache):
    os.makedirs(mnist_training_cache)
DataSourceWithFileCache(data_source=MnistDataSource(train=True, shuffle=False, rng=None),
                        cache_dir=mnist_training_cache,
                        shuffle=False,
                        rng=None)
mnist_test_cache = args.output + '/mnist_test.cache'
if not os.path.exists(mnist_test_cache):
    os.makedirs(mnist_test_cache)
DataSourceWithFileCache(data_source=MnistDataSource(train=False, shuffle=False, rng=None),
                        cache_dir=mnist_test_cache,
                        shuffle=False,
                        rng=None)
