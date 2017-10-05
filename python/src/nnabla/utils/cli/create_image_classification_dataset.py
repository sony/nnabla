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
import imghdr
import numpy as np
import scipy.misc
import nnabla.logger as logger
import csv
import tqdm


def create_image_classification_dataset_command(args):
    # settings
    source_dir = args.sourcedir
    dest_csv_file_name = [os.path.join(args.outdir, args.file1)]
    if args.file2:
        dest_csv_file_name.append(os.path.join(args.outdir, args.file2))
    dest_dir = args.outdir
    width = int(args.width)
    height= int(args.height)
    padding = args.mode == 'padding'
    ch = int(args.channel)
    shuffle = args.shuffle == 'true'
    test_data_ratio = int(args.ratio2) if args.ratio2 else 0
    
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
        files = [f for f in files if os.path.isfile(os.path.join(full_path, f))]
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
            current = round(100 * (float(i) / len(dirs) + float(i2) / (len(dirs) * len(files))))
            if last < current:
                pbar.update(current - last)
                last = current
    pbar.close()

    # create output data
    logger.log(99, "Creating output images...")
    for data in tqdm.tqdm(csv_data, unit='images'):
        src_file_name = os.path.join(source_dir, data[0])
        data[0] = os.path.splitext(data[0])[0] + ".png"
        dest_file_name = os.path.join(dest_dir, data[0])
        dest_path = os.path.dirname(dest_file_name)
        # print(src_file_name, dest_file_name)
        
        # open source image
        im = scipy.misc.imread(src_file_name)
        if len(im.shape) < 2 or len(im.shape) > 3:
            logger.warning("Illigal image file format %s.".format(src_file_name))
            csv_data.remove(data)
            continue
        elif len(im.shape) == 3:
            # RGB image
            if im.shape[2] != 3:
                logger.warning("The image must be RGB or monochrome %s.".format(src_file_name))
                csv_data.remove(data)
                continue
        
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
                    im = im[(h - target_h) // 2:h - (h - target_h) // 2, ::]
                else:
                    target_w = int(float(h) / height * width)
                    # print('crop_target_w', target_w)
                    im = im[::, (w - target_w) // 2:w - (w - target_w) // 2]
                # print('before', im.shape)
                im = scipy.misc.imresize(arr=im, size=(height, width), interp='lanczos')
                # print('after', im.shape)
            else:
                # padding mode
                if float(h) / w < float(height) / width:
                    target_h = int(float(height) / width * w)
                    # print('padding_target_h', target_h)
                    pad = (((target_h - h) // 2, target_h - (target_h - h) // 2 - h), (0, 0))
                else:
                    target_w = int(float(width) / height * h)
                    # print('padding_target_w', target_w)
                    pad = ((0, 0), ((target_w - w) // 2, target_w - (target_w - w) // 2 - w))
                if len(im.shape) == 3:
                    pad = pad + ((0, 0),)
                im = np.pad(im, pad, 'constant')
                # print('before', im.shape)
                im = scipy.misc.imresize(arr=im, size=(height, width), interp='lanczos')
                # print('after', im.shape)

        # change color ch
        if len(im.shape) == 2 and ch == 3:
            # Monochrome to RGB
            im = np.array([im, im, im]).transpose((1,2,0))
        elif len(im.shape) == 3 and ch == 1:
            # RGB to monochrome
            im = np.dot(im[...,:3], [0.299, 0.587, 0.114])
        
        # output
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
        scipy.misc.imsave(dest_file_name, im)
    
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
