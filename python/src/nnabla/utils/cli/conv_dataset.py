# Copyright 2017,2018,2019,2020,2021 Sony Corporation.
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

import os
import shutil

from nnabla.config import nnabla_config
from nnabla.utils.create_cache import CreateCache
from nnabla.utils.data_source import DataSourceWithFileCache
from nnabla.utils.data_source_implements import CacheDataSource, CsvDataSource
from tqdm import tqdm


def _convert(args, source):
    _, ext = os.path.splitext(args.destination)
    if ext.lower() == '.cache':
        with DataSourceWithFileCache(source, cache_dir=args.destination, shuffle=args.shuffle) as ds:
            print('Number of Data: {}'.format(ds.size))
            print('Shuffle:        {}'.format(args.shuffle))
            print('Normalize:      {}'.format(args.normalize))
            pbar = None
            if nnabla_config.get('MISC', 'misc_show_progress') == 'True':
                pbar = tqdm(total=ds.size)
            for i in range(ds.size):
                ds._get_data(i)
                if pbar is not None:
                    pbar.update(1)
    else:
        print('Command `conv_dataset` only supports CACHE as destination.')


def conv_dataset_command(args):
    if type(args.num_of_threads) == int and args.num_of_threads <= 0:
        print(
            "The numbers of threads [{}] must be positive integer.".format(args.num_of_threads))
        return False

    if os.path.exists(args.destination):
        if not args.force:
            print(
                'File or directory [{}] is exists use `-F` option to overwrite it.'.format(args.destination))
            return False
        elif os.path.isdir(args.destination):
            print('Overwrite destination [{}].'.format(args.destination))
            shutil.rmtree(args.destination, ignore_errors=True)
            os.mkdir(args.destination)
        else:
            print('Cannot overwrite file [{}] please delete it.'.format(
                args.destination))
            return False
    else:
        os.mkdir(args.destination)

    _, ext = os.path.splitext(args.source)
    if ext.lower() == '.csv':

        if os.path.exists(args.source):
            cc = CreateCache(args.source, shuffle=args.shuffle,
                             num_of_threads=args.num_of_threads)
            print('Number of Data: {}'.format(cc._size))
            print('Shuffle:        {}'.format(cc._shuffle))
            print('Normalize:      {}'.format(args.normalize))
            cc.create(args.destination, normalize=args.normalize)
        else:
            with CsvDataSource(args.source, shuffle=args.shuffle, normalize=args.normalize) as source:
                _convert(args, source)

    elif ext.lower() == '.cache':
        with CacheDataSource(args.source, shuffle=args.shuffle, normalize=args.normalize) as source:
            _convert(args, source)
    else:
        print('Command `conv_dataset` only supports CSV or CACHE as source.')
    return True


def add_conv_dataset_command(subparsers):
    # Convert dataset
    subparser = subparsers.add_parser(
        'conv_dataset', help='Convert CSV dataset to cache.')
    subparser.add_argument('-F', '--force', action='store_true',
                           help='force overwrite destination', required=False)
    subparser.add_argument(
        '-S', '--shuffle', action='store_true', help='shuffle data', required=False)
    subparser.add_argument('-N', '--normalize', action='store_true',
                           help='normalize data range', required=False)
    subparser.add_argument('-t', "--num_of_threads", type=int, required=False,
                           help='use multithreading to convert cache, default to 10')
    subparser.add_argument('source')
    subparser.add_argument('destination')
    subparser.set_defaults(func=conv_dataset_command)
