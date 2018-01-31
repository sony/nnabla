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
import shutil
import sys
from tqdm import tqdm

from nnabla.utils.data_source import DataSourceWithFileCache
from nnabla.utils.data_source_implements import CacheDataSource, CsvDataSource


def _convert(args, source):
    _, ext = os.path.splitext(args.destination)
    if ext.lower() == '.cache':
        with DataSourceWithFileCache(source, cache_dir=args.destination, shuffle=args.shuffle) as ds:
            print('Number of Data: {}'.format(ds.size))
            print('Shuffle:        {}'.format(args.shuffle))
            print('Normalize:      {}'.format(args.normalize))
            pbar = tqdm(total=ds.size)
            for i in range(ds.size):
                ds._get_data(i)
                pbar.update(1)
    else:
        print('Command `conv_dataset` only supports CACHE as destination.')


def conv_dataset_command(args):
    if os.path.exists(args.destination):
        if not args.force:
            print(
                'File or directory [{}] is exists use `-F` option to overwrite it.'.format(args.destination))
            sys.exit(-1)
        elif os.path.isdir(args.destination):
            print('Overwrite destination [{}].'.format(args.destination))
            shutil.rmtree(args.destination, ignore_errors=True)
            os.makedirs(args.destination, exist_ok=True)
        else:
            print('Cannnot overwrite file [{}] please delete it.'.format(
                args.destination))
            sys.exit(-1)
    else:
        os.makedirs(args.destination, exist_ok=True)

    _, ext = os.path.splitext(args.source)
    if ext.lower() == '.csv':
        with CsvDataSource(args.source, shuffle=args.shuffle, normalize=args.normalize) as source:
            _convert(args, source)
    elif ext.lower() == '.cache':
        with CacheDataSource(args.source, shuffle=args.shuffle, normalize=args.normalize) as source:
            _convert(args, source)
    else:
        print('Command `conv_dataset` only supports CSV or CACHE as source.')
