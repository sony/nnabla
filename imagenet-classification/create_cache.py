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
import numpy as np
import os
import re
import scipy.io
import shutil
import tarfile
import tempfile
import tqdm
import csv

from nnabla.utils.image_utils import imresize, imread

VALIDATION_DATA_LABEL = "validation_data_label.txt"
LABEL_WORDNETID = "label_wordnetid.csv"

def _resize_image(im, width, height, padding):
    # resize
    h = im.shape[0]
    w = im.shape[1]
    if w != width or h != height:
        # resize image
        if not padding:
            # trimming mode
            if float(h) / w > float(height) / width:
                target_h = int(float(w) / width * height)
                im = im[(h - target_h) // 2:h -
                        (h - target_h) // 2, ::]
            else:
                target_w = int(float(h) / height * width)
                im = im[::, (w - target_w) // 2:w -
                        (w - target_w) // 2]
        else:
            # padding mode
            if float(h) / w < float(height) / width:
                target_h = int(float(height) / width * w)
                pad = (((target_h - h) // 2, target_h -
                        (target_h - h) // 2 - h), (0, 0))
            else:
                target_w = int(float(width) / height * h)
                pad = ((0, 0), ((target_w - w) // 2,
                                target_w - (target_w - w) // 2 - w))
            pad = pad + ((0, 0),)
            im = np.pad(im, pad, 'constant')
        im = imresize(im, (width, height))

    x = np.array(im, dtype=np.uint8).transpose((2, 0, 1))
    return x


def _create_train_cache(archive, output, names, args):
    # Read label and wordnet_id
    wid2ind = np.loadtxt(fname=LABEL_WORDNETID, dtype=object, delimiter=",")

    def _get_label(wordnet_id):
        for item in wid2ind:
            if item[1] == wordnet_id:
                return item[0]

    images0 = []
    print("Count image in TAR")
    pbar = tqdm.tqdm(total=len(names), unit='%')
    for name in names:
        category = os.path.splitext(name)[0]
        marchive = tarfile.open(fileobj=archive.extractfile(name))
        for mname in marchive.getnames():
            if re.match(r'{}_[0-9]+\.JPEG'.format(category), mname):
                images0.append((_get_label(name[:9]), name, marchive, mname))
            else:
                print('Invalid file {} includes in tar file'.format(mname))
                exit(-1)
        pbar.update(1)
    pbar.close()

    # Thinning
    images = []
    for i, image in enumerate(images0):
        if i % args.thinning == 0:
            images.append(image)

    def _load_func(index):
        y, name, marchive, mname = images[index]
        im = imread(marchive.extractfile(mname), num_channels=3)
        x = _resize_image(im, args.width, args.height, args.mode == 'padding')
        return x, np.array([y]).astype(np.int32)

    from nnabla.utils.data_source import DataSourceWithFileCache
    from nnabla.utils.data_source_implements import SimpleDataSource
    from nnabla.logger import logger

    logger.info('Num of data : {}'.format(len(images)))
    shuffle = True
    if args.shuffle == 'False':
        shuffle = False
    source = SimpleDataSource(_load_func, len(images), shuffle, rng=None)
    DataSourceWithFileCache(
        source, cache_dir=output, shuffle=args.shuffle)


def _create_validation_cache(archive, output, names, ground_truth, args):
    images0 = sorted(names)

    # Thinning
    images = []
    for i, image in enumerate(images0):
        if i % args.thinning == 0:
            images.append(image)

    def _load_func(index):
        y, name = ground_truth[index], images[index]
        im = imread(archive.extractfile(name), num_channels=3)
        x = _resize_image(im, args.width, args.height, args.mode == 'padding')
        return x, np.array([y]).astype(np.int32)

    from nnabla.utils.data_source import DataSourceWithFileCache
    from nnabla.utils.data_source_implements import SimpleDataSource
    from nnabla.logger import logger

    logger.info('Num of data : {}'.format(len(images)))
    shuffle = False
    if args.shuffle == 'True':
        shuffle = True
    source = SimpleDataSource(_load_func, len(images), shuffle, rng=None)
    DataSourceWithFileCache(
        source, cache_dir=output, shuffle=args.shuffle)


_pbar = None
_prev_progress = None


def _progress(state, progress=0.0):
    global _pbar
    global _prev_progress

    if state is None:
        if _pbar is not None:
            _pbar.close()
        _pbar = None
        _prev_progress = None
    else:
        if _pbar is None:
            _pbar = tqdm.tqdm(desc=state, total=100, unit='%')
        else:
            if _prev_progress is None:
                _prev_progress = 0
            update = int((progress - _prev_progress) * 100)
            if update > 0:
                _pbar.update(update)
                _prev_progress = progress


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, nargs='+',
                        help='Source file or directory.')
    parser.add_argument('output', type=str,
                        help='Destination directory.')
    parser.add_argument('-W', '--width', type=int, default=320,
                        help='width of output image (default:320)')
    parser.add_argument('-H', '--height', type=int, default=320,
                        help='height of output image (default:320)')
    parser.add_argument('-m', '--mode', default='trimming',
                        choices=['trimming', 'padding'],
                        help='shaping mode (trimming or padding)  (default:trimming)')
    parser.add_argument('-S', '--shuffle', choices=['True', 'False'],
                        help='shuffle mode if not specified, train:True, val:False.' +
                        ' Otherwise specified value will be used for both.')
    parser.add_argument('-N', '--file-cache-size', type=int, default=100,
                        help='num of data in cache file (default:100)')
    parser.add_argument('-C', '--cache-type', default='npy',
                        choices=['h5', 'npy'],
                        help='cache format (h5 or npy) (default:npy)')
    parser.add_argument('--thinning', type=int, default=1,
                        help='Thinning rate')

    args = parser.parse_args()
    ############################################################################
    # Analyze tar
    # If it consists only of members corresponding to regular expression
    # 'n[0-9]{8}\.tar', it is judged as train data archive.
    # If it consists only of members corresponding to regular expression
    # 'ILSVRC2012_val_[0-9]{8}\.JPEG', it is judged as validation data archive.

    archives = {'train': None, 'val': None}
    for inputarg in args.input:
        print('Checking input file [{}]'.format(inputarg))
        archive = tarfile.open(inputarg)
        is_train = False
        is_val = False
        names = []
        for name in archive.getnames():
            if re.match(r'n[0-9]{8}\.tar', name):
                if is_val:
                    print('Train data {} includes in validation tar'.format(name))
                    exit(-1)
                is_train = True
            elif re.match(r'ILSVRC2012_val_[0-9]{8}\.JPEG', name):
                if is_train:
                    print('Validation data {} includes in train tar'.format(name))
                    exit(-1)
                is_val = True
            else:
                print('Invalid member {} includes in tar file'.format(name))
                exit(-1)
            names.append(name)
        if is_train:
            if archives['train'] is None:
                archives['train'] = (archive, names)
            else:
                print('Please specify only 1 training tar archive.')
                exit(-1)
        if is_val:
            if archives['val'] is None:
                archives['val'] = (archive, names)
            else:
                print('Please specify only 1 validation tar archive.')
                exit(-1)

    # Read label of validation data, (Use ascending label of wordnet_id)
    validation_ground_truth = []
    g_file = VALIDATION_DATA_LABEL
    with open(g_file, 'r') as f:
        for l in f.readlines():
            validation_ground_truth.append(int(l.rstrip()))

    ############################################################################
    # Prepare logging
    tmpdir = tempfile.mkdtemp()
    logfilename = os.path.join(tmpdir, 'nnabla.log')

    # Temporarily chdir to tmpdir just before importing nnabla to reflect nnabla.conf.
    cwd = os.getcwd()
    os.chdir(tmpdir)
    with open('nnabla.conf', 'w') as f:
        f.write('[LOG]\n')
        f.write('log_file_name = {}\n'.format(logfilename))
        f.write('log_file_format = %(funcName)s : %(message)s\n')
        f.write('log_console_level = CRITICAL\n')

    from nnabla.config import nnabla_config
    os.chdir(cwd)

    ############################################################################
    # Data iterator setting
    nnabla_config.set('DATA_ITERATOR',
                      'cache_file_format', '.' + args.cache_type)
    nnabla_config.set('DATA_ITERATOR',
                      'data_source_file_cache_size', str(args.file_cache_size))
    nnabla_config.set('DATA_ITERATOR',
                      'data_source_file_cache_num_of_threads', '1')

    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    ############################################################################
    # Prepare status monitor
    from nnabla.utils.progress import configure_progress
    configure_progress(None, _progress)

    ############################################################################
    # Converter

    try:
        if archives['train'] is not None:
            from nnabla.logger import logger
            logger.info('StartCreatingCache')
            archive, names = archives['train']
            output = os.path.join(args.output, 'train')
            if not os.path.isdir(output):
                os.makedirs(output)
            _create_train_cache(archive, output, names, args)
        if archives['val'] is not None:
            from nnabla.logger import logger
            logger.info('StartCreatingCache')
            archive, names = archives['val']
            output = os.path.join(args.output, 'val')
            if not os.path.isdir(output):
                os.makedirs(output)
            _create_validation_cache(
                archive, output, names, validation_ground_truth, args)
    except KeyboardInterrupt:
        shutil.rmtree(tmpdir, ignore_errors=True)

        # Even if CTRL-C is pressed, it does not stop if there is a running
        # thread, so it sending a signal to itself.
        os.kill(os.getpid(), 9)

    ############################################################################
    # Finish
    _finish = True
    shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == '__main__':
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        main()
