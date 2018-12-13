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
import sys
import imghdr
import tqdm
import numpy as np
import scipy.misc
import argparse
import h5py
import nnabla.logger as logger


def create_cache_file(args):
    # settings
    source_dir = args.sourcedir
    dest_dir = args.outdir
    width = int(args.width)
    height = int(args.height)
    padding = args.mode == 'padding'
    shuffle = args.shuffle == 'true'

    if source_dir == dest_dir:
        logger.critical("Input directory and output directory are same.")
        return

    # create file list
    logger.log(99, "Creating file list...")
    dirs = os.listdir(source_dir)
    dirs = [d for d in dirs if os.path.isdir(os.path.join(source_dir, d))]
    dirs.sort()
    # print(dirs)

    labels = []
    label_index = -1
    csv_data = []
    pbar = tqdm.tqdm(total=100, unit='%')
    last = 0
    for i, dir in enumerate(dirs):
        # print(dir)
        full_path = os.path.join(source_dir, dir)
        files = os.listdir(full_path)
        files = [f for f in files if os.path.isfile(
            os.path.join(full_path, f))]
        files.sort()
        found = False
        for i2, file in enumerate(files):
            file_name = os.path.join(full_path, file)
            if imghdr.what(file_name) is not None:
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

    logger.log(99, "Creating cache files...")
    if shuffle:
        import random
        random.shuffle(csv_data)

    data_size = 100
    num_data_files = int((len(csv_data)-1)/data_size + 1)
    for i in tqdm.tqdm(range(num_data_files)):
        num_image = data_size if (
            i+1) * data_size < len(csv_data) else len(csv_data) - i * data_size
        data = {}
        data['x'] = []
        data['y'] = []
        for i2 in range(num_image):
            image_file_name = csv_data[i2 + i * data_size][0]
            class_index_str = csv_data[i2 + i * data_size][1]
            image_file_name = source_dir + csv_data[i2 + i * data_size][0][1:]
            class_index = int(class_index_str)
            if os.path.exists(image_file_name):
                im = scipy.misc.imread(image_file_name, mode='RGB')
                # resize
                h = im.shape[0]
                w = im.shape[1]
                # print(h, w)
                if w != width or h != height:
                    # resize image
                    if not padding:
                        # trimming mode
                        if float(h) / w > float(height) / width:
                            target_h = int(float(w) / width * height)
                            # print('crop_target_h', target_h)
                            im = im[(h - target_h) // 2:h -
                                    (h - target_h) // 2, ::]
                        else:
                            target_w = int(float(h) / height * width)
                            # print('crop_target_w', target_w)
                            im = im[::, (w - target_w) // 2:w -
                                    (w - target_w) // 2]
                        # print('before', im.shape)
                        im = scipy.misc.imresize(arr=im, size=(
                            height, width), interp='lanczos')
                        # print('after', im.shape)
                    else:
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
                        pad = pad + ((0, 0),)
                        im = np.pad(im, pad, 'constant')
                        # print('before', im.shape)
                        im = scipy.misc.imresize(arr=im, size=(
                            height, width), interp='lanczos')
                        # print('after', im.shape)
                    x = np.array(im, dtype=np.uint8).transpose((2, 0, 1))
                # print x.shape, x.dtype
                data['x'].append(x)
                data['y'].append(np.array([class_index], dtype=np.int16))
            else:
                print(image_file_name, ' is not found.')
        out_file_name = dest_dir + '/data{:04d}_{}.h5'.format(i, num_image)
        h5 = h5py.File(out_file_name, 'w')
        h5.create_dataset('y', data=data['y'])
        h5.create_dataset('x', data=data['x'])
        h5.close


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--sourcedir', help='source directory with directories for each class', required=True)
    parser.add_argument(
        '-o', '--outdir', help='output directory', required=True)
    parser.add_argument(
        '-w', '--width', help='width of output image', required=True)
    parser.add_argument(
        '-g', '--height', help='height of output image', required=True)
    parser.add_argument(
        '-m', '--mode', help='shaping mode (trimming or padding)', required=True)
    parser.add_argument(
        '-s', '--shuffle', help='shuffle mode (true or false)', required=True)
    args = parser.parse_args()

    create_cache_file(args)


if __name__ == '__main__':
    main()
