# Copyright 2019,2020,2021 Sony Corporation.
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


class ObjectRect:
    def __init__(self, LRTB=None, XYWH=None):
        if LRTB is not None:
            self.rect = np.array(LRTB)
        elif XYWH is not None:
            self.rect = np.array([XYWH[0] - XYWH[2] * 0.5, XYWH[1] - XYWH[3]
                                  * 0.5, XYWH[0] + XYWH[2] * 0.5, XYWH[1] + XYWH[3] * 0.5])
        else:
            self.rect = np.full((4,), 0.0, dtype=float)

    def clip(self):
        return ObjectRect(LRTB=self.rect.clip(0.0, 1.0))

    def left(self):
        return self.rect[0]

    def top(self):
        return self.rect[1]

    def right(self):
        return self.rect[2]

    def bottom(self):
        return self.rect[3]

    def width(self):
        return np.max(self.rect[2] - self.rect[0], 0)

    def height(self):
        return np.max(self.rect[3] - self.rect[1], 0)

    def centerx(self):
        return (self.rect[0] + self.rect[2]) * 0.5

    def centery(self):
        return (self.rect[1] + self.rect[3]) * 0.5

    def center(self):
        return self.centerx(), self.centery()

    def area(self):
        return self.width() * self.height()

    def overlap(self, rect2):
        w = np.max([np.min([self.right(), rect2.right()]) -
                    np.max([self.left(), rect2.left()])], 0)
        h = np.max([np.min([self.bottom(), rect2.bottom()]) -
                    np.max([self.top(), rect2.top()])], 0)
        return w * h

    def iou(self, rect2):
        overlap = self.overlap(rect2)
        return overlap / (self.area() + rect2.area() - overlap)


def load_label(file_name):
    labels = []
    if os.path.exists(file_name):
        with open(file_name, "rt") as f:
            lines = f.readlines()
        for line in lines:
            values = [float(s) for s in line.split(' ')]
            if len(values) == 5:
                labels.append(values)
    else:
        logger.warning(
            "Label txt file is not found %s." % (file_name))
    return labels


def convert_image(args):
    file_name = args[0]
    source_dir = args[1]
    dest_dir = args[2]
    width = args[3]
    height = args[4]
    mode = args[5]
    ch = args[6]
    num_class = args[7]
    grid_size = args[8]
    anchors = args[9]

    src_file_name = os.path.join(source_dir, file_name)
    src_label_file_name = os.path.join(
        source_dir, os.path.splitext(file_name)[0] + ".txt")
    image_file_name = os.path.join(
        dest_dir, 'data', os.path.splitext(file_name)[0] + ".png")
    label_file_name = os.path.join(
        dest_dir, 'data', os.path.splitext(file_name)[0] + "_label.csv")
    region_file_name = os.path.join(
        dest_dir, 'data', os.path.splitext(file_name)[0] + "_region.csv")
    try:
        os.makedirs(os.path.dirname(image_file_name))
    except OSError:
        pass  # python2 does not support exists_ok arg
    # print(src_file_name, dest_file_name)

    # open source image
    labels = load_label(src_label_file_name)

    warp_func = None
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
        input_size = (w, h)
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

                def trim_warp(label, input_size, output_size):
                    w_scale = input_size[0] * 1.0 / output_size[0]
                    h_scale = input_size[1] * 1.0 / output_size[1]
                    label[0] = (label[0] - (1.0 - 1.0 / w_scale)
                                * 0.5) * w_scale
                    label[1] = (label[1] - (1.0 - 1.0 / h_scale)
                                * 0.5) * h_scale
                    label[3] *= w_scale
                    label[4] *= h_scale
                    return label
                warp_func = trim_warp
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

                def pad_warp(label, input_size, output_size):
                    w_scale = input_size[0] * 1.0 / output_size[0]
                    h_scale = input_size[1] * 1.0 / output_size[1]
                    label[0] = (label[0] * w_scale + (1.0 - w_scale) * 0.5)
                    label[1] = (label[1] * h_scale + (1.0 - h_scale) * 0.5)
                    label[3] *= w_scale
                    label[4] *= h_scale
                    return label
                warp_func = pad_warp
            im = imresize(im, size=(width, height))
            output_size = (width, height)
            # print('after', im.shape)

        # change color ch
        if len(im.shape) == 2 and ch == 3:
            # Monochrome to RGB
            im = np.array([im, im, im]).transpose((1, 2, 0))
        elif len(im.shape) == 3 and ch == 1:
            # RGB to monochrome
            im = np.dot(im[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

        # output image
        imsave(image_file_name, im)

    except:
        logger.warning(
            "Failed to convert %s." % (src_file_name))
        raise

    # create label and region file
    if warp_func is not None:
        labels = [warp_func(label, input_size, output_size)
                  for label in labels]
    grid_w = width // grid_size
    grid_h = height // grid_size
    label_array = np.full((len(anchors), grid_h, grid_w), -1, dtype=int)
    region_array = np.full(
        (len(anchors), grid_h, grid_w, 4), 0.0, dtype=float)

    for label in labels:
        label_rect = ObjectRect(XYWH=label[1:]).clip()

        if label_rect.width() > 0.0 and label_rect.height() > 0.0:
            gx, gy = int(label_rect.centerx() *
                         grid_w), int(label_rect.centery() * grid_h)
            max_iou = 0
            anchor_index = 0
            for i, anchor in enumerate(anchors):
                anchor_rect = ObjectRect(
                    XYWH=[(gx + 0.5) / grid_w, (gy + 0.5) / grid_h, anchor[0], anchor[1]])
                iou = label_rect.iou(anchor_rect)
                if iou > max_iou:
                    anchor_index = i
                    max_iou = iou
            label_array[anchor_index][gy][gx] = int(label[0])
            region_array[anchor_index][gy][gx] = [(label_rect.centerx() - anchor_rect.centerx()) * grid_w + 0.5, (label_rect.centery(
            ) - anchor_rect.centery()) * grid_h + 0.5, np.log(label_rect.width() * grid_w), np.log(label_rect.height() * grid_h)]
    np.savetxt(label_file_name, label_array.reshape(
        (label_array.shape[0] * label_array.shape[1], -1)), fmt='%i', delimiter=',')
    np.savetxt(region_file_name, region_array.reshape(
        (region_array.shape[0] * region_array.shape[1], -1)), fmt='%f', delimiter=',')


def get_anchors(source_dir, file_list, num_anchor):
    # List label width and height
    labels = []
    for file_name in tqdm.tqdm(file_list):
        src_label_file_name = os.path.join(
            source_dir, os.path.splitext(file_name)[0] + ".txt")
        labels.extend(np.array(load_label(src_label_file_name))[:, 3:5])
    labels = np.array(labels)
    logger.log(99, '{} labels are found in {} images ({:.2f} labels/image on average).'.format(
        len(labels), len(file_list), len(labels) * 1.0 / len(file_list)))

    # k-means
    np.random.seed(0)
    classes = np.random.randint(num_anchor, size=len(labels))
    loop = 0
    while loop < 1000:
        means = []  # anchor * wh
        distance = []  # anchor * data
        for i in range(num_anchor):
            mean = labels[classes == i].mean(axis=0)
            means.append(mean)
            distance.append(np.sum((labels - mean) ** 2, axis=1))

        new_classes = np.array(distance).argmin(axis=0)
        # print(loop, means)
        loop += 1
        if np.sum(classes == new_classes) == len(classes):
            break
        classes = new_classes

    # sort anchors by area
    means = np.array(means)
    area = means[:, 0] * means[:, 1]
    return means[area.argsort()]


def create_object_detection_dataset_command(args):
    # settings
    source_dir = args.sourcedir
    dest_dir = args.outdir
    width = int(args.width)
    height = int(args.height)
    mode = args.mode
    ch = int(args.channel)
    num_class = int(args.num_class)
    grid_size = int(args.grid_size)
    shuffle = args.shuffle == 'true'
    num_anchor = int(args.num_anchor)

    if width % grid_size != 0:
        logger.log(99, 'width" must be divisible by grid_size.')
        return
    if height % grid_size != 0:
        logger.log(99, 'height must be divisible by grid_size.')
        return

    dest_csv_file_name = [os.path.join(args.outdir, args.file1)]
    if args.file2:
        dest_csv_file_name.append(os.path.join(args.outdir, args.file2))
    test_data_ratio = int(args.ratio2) if args.ratio2 else 0

    if args.sourcedir == args.outdir:
        logger.critical("Input directory and output directory are same.")
        return False

    # create file list
    logger.log(99, "Creating file list...")

    def create_file_list(dir=""):
        result = []
        items = os.listdir(os.path.join(source_dir, dir))
        for item in items:
            if os.path.isdir(os.path.join(source_dir, dir, item)):
                result.extend(create_file_list(os.path.join(dir, item)))
            elif re.search('\.(bmp|jpg|jpeg|png|gif|tif|tiff)', os.path.splitext(item)[1], re.IGNORECASE):
                result.append(os.path.join(dir, item))
        return result

    file_list = create_file_list()

    if len(file_list) == 0:
        logger.critical(
            "No image file found in the subdirectory of the input directory.")
        return False

    # calc anchor
    logger.log(99, "Calculating anchors...")
    anchors = get_anchors(source_dir, file_list, num_anchor)

    # create output data
    logger.log(99, "Creating output images...")
    process_args = [(data, source_dir, dest_dir, width,
                     height, mode, ch, num_class, grid_size, anchors) for data in file_list]
    p = mp.Pool(mp.cpu_count())
    pbar = tqdm.tqdm(total=len(process_args))
    for _ in p.imap_unordered(convert_image, process_args):
        pbar.update()
    pbar.close()

    file_list = [os.path.join('.', 'data', file) for file in file_list]
    file_list = [file for file in file_list if os.path.exists(
        os.path.join(dest_dir, os.path.splitext(file)[0] + '.png'))]
    if len(file_list) == 0:
        logger.critical("No image and label file created correctly.")
        return False

    logger.log(99, "Creating CSV files...")
    if shuffle:
        import random
        random.shuffle(file_list)

    csv_data_num = [(len(file_list) * (100 - test_data_ratio)) // 100]
    csv_data_num.append(len(file_list) - csv_data_num[0])
    data_head = 0
    for csv_file_name, data_num in zip(dest_csv_file_name, csv_data_num):
        if data_num:
            file_list_2 = file_list[data_head:data_head + data_num]
            data_head += data_num

            with open(csv_file_name, 'w') as f:
                writer = csv.writer(f, lineterminator='\n')
                writer.writerow(['x:image', 'y:label', 'r:region'])
                for file in file_list_2:
                    base_file_name = os.path.splitext(file)[0]
                    writer.writerow([file, os.path.splitext(
                        file)[0] + '_label.csv', os.path.splitext(file)[0] + '_region.csv'])

    logger.log(99, "Dataset was successfully created.")
    return True


def add_create_object_detection_dataset_command(subparsers):
    # Create object detection dataset
    subparser = subparsers.add_parser('create_object_detection_dataset',
                                      help='Create dataset from image and label files.')
    subparser.add_argument(
        '-i', '--sourcedir', help='source directory with souce image and label files', required=True)
    subparser.add_argument(
        '-o', '--outdir', help='output directory', required=True)
    subparser.add_argument(
        '-n', '--num_class', help='number of object classes', required=True)
    subparser.add_argument(
        '-c', '--channel', help='number of output color channels', required=True)
    subparser.add_argument(
        '-w', '--width', help='width of output image', required=True)
    subparser.add_argument(
        '-g', '--height', help='height of output image', required=True)
    subparser.add_argument(
        '-a', '--num_anchor', help='number of anchor', required=True)
    subparser.add_argument(
        '-d', '--grid_size', help='width and height of detection grid', required=True)
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
    subparser.set_defaults(func=create_object_detection_dataset_command)
