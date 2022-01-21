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

import csv
import multiprocessing as mp
import os
import re

import nnabla.logger as logger
import numpy as np
import tqdm
from nnabla.utils.image_utils import imsave, imread, imresize


def convert_image(args):
    file_name = args[0]
    source_dir = args[1]
    dest_dir = args[2]
    width = args[3]
    height = args[4]
    mode = args[5]
    ch = args[6]

    src_file_name = os.path.join(source_dir, file_name)
    file_name = os.path.splitext(file_name)[0] + ".png"
    dest_file_name = os.path.join(dest_dir, file_name)
    dest_path = os.path.dirname(dest_file_name)
    # print(src_file_name, dest_file_name)

    # open source image
    try:
        im = imread(src_file_name)
        if len(im.shape) < 2 or len(im.shape) > 3:
            logger.warning(
                "Illegal image file format %s.".format(src_file_name))
            raise
        elif len(im.shape) == 3:
            # RGB image
            if im.shape[2] != 3:
                logger.warning(
                    "The image must be RGB or monochrome.")
                csv_data.remove(data)
                raise

        # resize
        h = im.shape[0]
        w = im.shape[1]
        # print(h, w)
        if w != width or h != height:
            # resize image
            if mode == 'trimming':
                # trimming mode
                if float(h) / w > float(height) / width:
                    target_h = int(float(w) / width * height)
                    # print('crop_target_h', target_h)
                    im = im[(h - target_h) // 2:h - (h - target_h) // 2, ::]
                else:
                    target_w = int(float(h) / height * width)
                    # print('crop_target_w', target_w)
                    im = im[::, (w - target_w) // 2:w - (w - target_w) // 2]
                # print('before', im.shape)
            elif mode == 'padding':
                # padding mode
                if float(h) / w < float(height) / width:
                    target_h = int(float(height) / width * w)
                    # print('padding_target_h', target_h)
                    pad = (((target_h - h) // 2, target_h -
                            (target_h - h) // 2 - h), (0, 0))
                else:
                    target_w = int(float(width) / height * h)
                    # print('padding_target_w', target_w)
                    pad = ((0, 0), ((target_w - w) // 2,
                                    target_w - (target_w - w) // 2 - w))
                if len(im.shape) == 3:
                    pad = pad + ((0, 0),)
                im = np.pad(im, pad, 'constant')
                # print('before', im.shape)
            im = imresize(im, size=(width, height))
            # print('after', im.shape)

        # change color ch
        if len(im.shape) == 2 and ch == 3:
            # Monochrome to RGB
            im = np.array([im, im, im]).transpose((1, 2, 0))
        elif len(im.shape) == 3 and ch == 1:
            # RGB to monochrome
            im = np.dot(im[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

        # output
        try:
            os.makedirs(dest_path)
        except OSError:
            pass  # python2 does not support exists_ok arg

        imsave(dest_file_name, im)
    except:
        logger.warning(
            "Failed to convert %s." % (src_file_name))


def create_image_classification_dataset_command(args):
    # settings
    source_dir = args.sourcedir
    dest_dir = args.outdir
    width = int(args.width)
    height = int(args.height)
    mode = args.mode
    ch = int(args.channel)
    shuffle = args.shuffle == 'true'

    dest_csv_file_name = [os.path.join(args.outdir, args.file1)]
    if args.file2:
        dest_csv_file_name.append(os.path.join(args.outdir, args.file2))
    test_data_ratio = int(args.ratio2) if args.ratio2 else 0

    if args.sourcedir == args.outdir:
        logger.critical("Input directory and output directory are same.")
        return False

    # create file list
    logger.log(99, "Creating file list...")
    dirs = os.listdir(args.sourcedir)
    dirs = [d for d in dirs if os.path.isdir(os.path.join(args.sourcedir, d))]
    dirs.sort()
    if len(dirs) == 0:
        logger.critical("Subdirectory not found in the input directory.")
        return False
    # print(dirs)

    labels = []
    label_index = -1
    csv_data = []
    pbar = tqdm.tqdm(total=100, unit='%')
    last = 0
    for i, dir in enumerate(dirs):
        # print(dir)
        full_path = os.path.join(args.sourcedir, dir)
        files = os.listdir(full_path)
        files = [f for f in files if os.path.isfile(
            os.path.join(full_path, f))]
        files.sort()
        found = False
        for i2, file in enumerate(files):
            file_name = os.path.join(full_path, file)
            if re.search('\.(bmp|jpg|jpeg|png|gif|tif|tiff)', os.path.splitext(file_name)[1], re.IGNORECASE):
                if not found:
                    labels.append(dir)
                    label_index += 1
                    found = True
                csv_data.append([os.path.join('.', dir, file), label_index])
            current = round(100 * (float(i) / len(dirs) +
                                   float(i2) / (len(dirs) * len(files))))
            if last < current:
                pbar.update(current - last)
                last = current
    pbar.close()
    if len(csv_data) == 0:
        logger.critical(
            "No image file found in the subdirectory of the input directory.")
        return False

    # create output data
    logger.log(99, "Creating output images...")
    process_args = [(data[0], source_dir, dest_dir, width,
                     height, mode, ch) for data in csv_data]
    p = mp.Pool(mp.cpu_count())
    pbar = tqdm.tqdm(total=len(process_args))
    for _ in p.imap_unordered(convert_image, process_args):
        pbar.update()
    pbar.close()

    for data in csv_data:
        file_name = os.path.splitext(data[0])[0] + ".png"
        data[0] = file_name if os.path.exists(
            os.path.join(dest_dir, file_name)) else None
    for data in csv_data[:]:
        if not data[0]:
            csv_data.remove(data)

    logger.log(99, "Creating CSV files...")
    if shuffle:
        import random
        random.shuffle(csv_data)

    csv_data_num = [(len(csv_data) * (100 - test_data_ratio)) // 100]
    csv_data_num.append(len(csv_data) - csv_data_num[0])
    data_head = 0
    for csv_file_name, data_num in zip(dest_csv_file_name, csv_data_num):
        if data_num:
            csv_data_2 = csv_data[data_head:data_head + data_num]
            data_head += data_num

            csv_data_2.insert(0, ['x:image', 'y:label'])
            with open(csv_file_name, 'w') as f:
                writer = csv.writer(f, lineterminator='\n')
                writer.writerows(csv_data_2)

    logger.log(99, "Dataset was successfully created.")
    return True


def add_create_image_classification_dataset_command(subparsers):
    # Create image classification dataset
    subparser = subparsers.add_parser('create_image_classification_dataset',
                                      help='Create dataset from image files.')
    subparser.add_argument(
        '-i', '--sourcedir', help='source directory with directories for each class', required=True)
    subparser.add_argument(
        '-o', '--outdir', help='output directory', required=True)
    subparser.add_argument(
        '-c', '--channel', help='number of output color channels', required=True)
    subparser.add_argument(
        '-w', '--width', help='width of output image', required=True)
    subparser.add_argument(
        '-g', '--height', help='height of output image', required=True)
    subparser.add_argument(
        '-m', '--mode', help='shaping mode (trimming or padding)', required=True)
    subparser.add_argument(
        '-s', '--shuffle', help='shuffle mode (true or false)', required=True)
    subparser.add_argument(
        '-f1', '--file1', help='output file name 1', required=True)
    subparser.add_argument(
        '-r1', '--ratio1', help='output file ratio(%%) 1')
    subparser.add_argument(
        '-f2', '--file2', help='output file name 2')
    subparser.add_argument(
        '-r2', '--ratio2', help='output file ratio(%%) 2')
    subparser.set_defaults(func=create_image_classification_dataset_command)
