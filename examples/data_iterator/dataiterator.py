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

import argparse
import os
import sys
import time
from tqdm import tqdm


def test_data_iterator(di, args):
    current_epoch = -1
    logger.info('{}'.format(di.size))
    pbar = None
    for data in di:
        time.sleep(args.wait / 1000.0)
        if di.epoch >= args.max_epoch:
            break
        if current_epoch != di.epoch:
            current_epoch = di.epoch
            if pbar is not None:
                pbar.close()
            logger.info('Epoch {}'.format(current_epoch))
            pbar = tqdm(total=di.size)
        pbar.update(len(data[0]))


if __name__ == '__main__':
    from nnabla.logger import logger
    from nnabla.config import nnabla_config

    parser = argparse.ArgumentParser(description='Data iterator sample.')
    parser.add_argument('-m', '--memory_cache', action='store_true',
                        help='Use memory cache')
    parser.add_argument('-f', '--file_cache', action='store_true',
                        help='Use file cache')
    parser.add_argument('-S', '--shuffle', action='store_true',
                        help='Enable shuffling data')
    parser.add_argument('-b', '--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('-s', '--cache_size', type=int, default=100,
                        help='Cache size (num of data).')
    parser.add_argument('-M', '--memory_size', type=int, default=1048576,
                        help='Memory buffer size in byte.')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='If specified, cache data will output to here.')
    parser.add_argument('-n', '--normalize', action='store_true',
                        help='Enable data normalize')
    parser.add_argument('-e', '--max_epoch', type=int, default=3,
                        help='Max epoch to read.')
    parser.add_argument('-w', '--wait', type=float, default=0,
                        help='Wait time for dummy data processing.')
    parser.add_argument('uri', help='PATH to CSV_DATASET format file or '
                        '"MNIST_TRAIN", "MNIST_TEST", "TINY_IMAGENET_TRAIN",'
                        '"TINY_IMAGENET_VAL"')
    args = parser.parse_args()

    logger.debug('memory_cache: {}'.format(args.memory_cache))
    logger.debug('file_cache: {}'.format(args.file_cache))
    logger.debug('shuffle: {}'.format(args.shuffle))
    logger.debug('batch_size: {}'.format(args.batch_size))
    logger.debug('cache_size: {}'.format(args.cache_size))
    logger.debug('memory_size: {}'.format(args.memory_size))
    logger.debug('output: {}'.format(args.output))
    logger.debug('normalize: {}'.format(args.normalize))
    logger.debug('max_epoch: {}'.format(args.max_epoch))
    logger.debug('wait: {}'.format(args.wait))

    nnabla_config.set('DATA_ITERATOR', 'data_source_file_cache_size',
                      '{}'.format(args.cache_size))
    nnabla_config.set('DATA_ITERATOR', 'data_source_buffer_max_size',
                      '{}'.format(args.memory_size))

    if args.uri == 'MNIST_TRAIN':
        sys.path.append(os.path.join(os.path.dirname(
            os.path.abspath(__file__)), '..', 'vision', 'mnist'))
        from mnist_data import data_iterator_mnist
        with data_iterator_mnist(args.batch_size,
                                 True,
                                 None,
                                 args.shuffle,
                                 args.memory_cache,
                                 args.file_cache) as di:
            test_data_iterator(di, args)
    elif args.uri == 'MNIST_TEST':
        sys.path.append(os.path.join(os.path.dirname(
            os.path.abspath(__file__)), '..', 'vision', 'mnist'))
        from mnist_data import data_iterator_mnist
        with data_iterator_mnist(args.batch_size,
                                 False,
                                 None,
                                 args.shuffle,
                                 args.memory_cache,
                                 args.file_cache) as di:
            test_data_iterator(di, args)
    elif args.uri == 'TINY_IMAGENET_TRAIN':
        sys.path.append(os.path.join(os.path.dirname(
            os.path.abspath(__file__)), '..', 'vision', 'imagenet'))
        from tiny_imagenet_data import data_iterator_tiny_imagenet
        with data_iterator_tiny_imagenet(args.batch_size, 'train') as di:
            test_data_iterator(di, args)
    elif args.uri == 'TINY_IMAGENET_VAL':
        sys.path.append(os.path.join(os.path.dirname(
            os.path.abspath(__file__)), '..', 'vision', 'imagenet'))
        from tiny_imagenet_data import data_iterator_tiny_imagenet
        with data_iterator_tiny_imagenet(args.batch_size, 'val') as di:
            test_data_iterator(di, args)
    else:
        if os.path.splitext(args.uri)[1].lower() == '.cache':
            from nnabla.utils.data_iterator import data_iterator_cache
            with data_iterator_cache(uri=args.uri,
                                     batch_size=args.batch_size,
                                     shuffle=args.shuffle,
                                     with_memory_cache=args.memory_cache,
                                     normalize=args.normalize) as di:
                test_data_iterator(di, args)
        else:
            from nnabla.utils.data_iterator import data_iterator_csv_dataset
            with data_iterator_csv_dataset(uri=args.uri,
                                           batch_size=args.batch_size,
                                           shuffle=args.shuffle,
                                           normalize=args.normalize,
                                           with_memory_cache=args.memory_cache,
                                           with_file_cache=args.file_cache,
                                           cache_dir=args.output) as di:
                test_data_iterator(di, args)
