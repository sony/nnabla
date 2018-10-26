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


# This file was forked from https://github.com/marvis/pytorch-yolo2 ,
# licensed under the MIT License (see LICENSE.external for more details).


import os
import random
import time
import numpy as np
from PIL import Image
import image
try:
    import image2
    import cv2
    allow_on_memory_data = True
except:
    print('!!cv2 seems not to be available. Using PIL for preprocessing instead, which slows down the training.!!')
    allow_on_memory_data = False


def read_truths(lab_path):
    if not os.path.exists(lab_path):
        return np.array([])
    if os.path.getsize(lab_path):
        truths = np.loadtxt(lab_path)
        # to avoid single truth problem
        truths = truths.reshape(truths.size/5, 5)
        return truths
    else:
        return np.array([])


def read_truths_args(lab_path, min_box_scale):
    truths = read_truths(lab_path)
    new_truths = []
    for i in range(truths.shape[0]):
        if truths[i][3] < min_box_scale:
            continue
        new_truths.append([truths[i][0], truths[i][1],
                           truths[i][2], truths[i][3], truths[i][4]])
    return np.array(new_truths)


class listDataset(object):
    '''
    TODO.

    During training, the image size augmentation is applied to the image. The image size is randomly sampled from the tuple argument ``image_size`` at the specified rate by ``image_size_change_freq``.

    Arg:
        imaeg_sizes(tuple): A tuple or list of image sizes for image size
            augmentation. The default value is the following multiples
            of 32: (320, 352, ..., 608).
            If you use the YOLOv2 network architecture, the numbers specified
            must be multiples of 32 since the size of the network final
            output is reduced by a factor of 32.
        image_size_change_freq (int): At every number of samples specified, the image size is randomly sapmled from the tuple image_size
        on_memory_data (tuple):
            path, image (uint8), label (float)
    '''

    def __init__(self, root, args, shuffle=True, train=False, seen=0, image_sizes=None, image_size_change_freq=6400, on_memory_data=None, use_cv2=True, shape=None):
        if not allow_on_memory_data or not use_cv2:
            print('On-memory data cannot be used if cv2 package is not available.')
            on_memory_data = None
        self.on_memory_data = on_memory_data
        if on_memory_data is None:
            with open(root, 'r') as file:
                self.lines = file.readlines()
            if shuffle:
                random.shuffle(self.lines)
            self.nSamples = len(self.lines)
        else:
            assert use_cv2
            if shuffle:
                self.indexes = np.random.permutation(len(on_memory_data[0]))
            self.nSamples = len(on_memory_data[0])
        if train:
            self.jitter = args.jitter
            self.hue = args.hue
            self.saturation = args.saturation
            self.exposure = args.exposure

        self.train = train
        self.seen = seen
        if image_sizes is None:
            image_sizes = tuple(range(320, 608 + 1, 32))
        self.image_sizes = image_sizes
        self.image_size_change_freq = image_size_change_freq
        self.shape = shape
        if shape is None:
            self.shape = (self.image_sizes[0], self.image_sizes[0])

        self.image_module = image
        if allow_on_memory_data and use_cv2:
            self.image_module = image2

    def __len__(self):
        return self.nSamples

    def __add__(self, other):
        print("Not implemented!!!!")
        raise NotImplementedError
        # return ConcatDataset([self, other])

    def __getitem__(self, index):
        # random.seed(index)
        assert index <= len(self), 'index range error'
        if self.train and index % self.image_size_change_freq == 0:
            # TODO: Use the max image size at the end of training.
            # The original darknet implementation does, but not sure how much it
            # affects the performance in practical use-cases.
            width = random.choice(self.image_sizes)
            self.shape = (width, width)

        if self.on_memory_data is None:
            imgpath = self.lines[index].rstrip()
            labpath = imgpath.replace('images', 'labels').replace(
                'JPEGImages', 'labels').replace('.jpg', '.txt').replace('.png', '.txt')
        else:
            imgpath = self.on_memory_data[1][self.indexes[index]]
            labpath = self.on_memory_data[2][self.indexes[index]]

        if self.train:
            img, label = self.image_module.load_data_detection(
                imgpath, labpath, self.shape, self.jitter, self.hue, self.saturation, self.exposure)
            img = np.asarray(img)
        else:
            assert isinstance(
                imgpath, str), 'valid does not handle on_memory_data'
            assert isinstance(
                labpath, str), 'valid does not handle on_memory_data'
            img = Image.open(imgpath).convert('RGB')
            if self.shape:
                img = img.resize(self.shape)
            label = np.zeros(50*5)
            try:
                tmp = self.image_module.read_truths_args(
                    labpath, 8.0/img.width).astype('float32')
            except Exception:
                tmp = np.zeros((1, 5))
            tmp = tmp.reshape(-1)
            tsz = tmp.size
            if tsz > 50*5:
                label = tmp[0:50*5]
            elif tsz > 0:
                label[0:tsz] = tmp
            img = np.asarray(img)

        img = img.transpose((2, 0, 1))
        img = img / np.float32(255.0)

        self.seen = self.seen + 1
        return (img, label)


def list2np(t, batch_array):
    ret = tuple()
    for j, items in enumerate(zip(*t)):
        shape = (len(items),) + items[0].shape
        if j >= len(batch_array):
            batch_array.append(np.empty(shape, dtype=np.float32))
        elif batch_array[j].shape != shape:
            batch_array[j] = np.empty(shape, dtype=np.float32)
        arr = batch_array[j]
        for i, item in enumerate(items):
            arr[i] = item
        ret += (arr,)
    return ret


def next_batch(it, batch_size, batch_array):
    i = 0
    retlist = []
    tic = time.time()
    for i in range(batch_size):
        item = next(it, None)
        if item is None:
            break
        retlist.append(item)

    # If no data, returns None
    if len(retlist) == 0:
        return
    ret = list2np(retlist, batch_array)

    # Don't train for batches that contain no labels
    if not (np.sum(ret[1]) == 0):
        return ret, (time.time() - tic) * 1000


def create_batch_iter(it, batch_size):
    batch_array = []
    while True:
        ret = next_batch(it, batch_size, batch_array)
        if ret is None:
            break
        yield ret


def load_on_memory_data(root):
    import tqdm
    if not allow_on_memory_data:
        print('On-memory data is not allowed because cv2 is not installed.')
        return None
    with open(root, 'r') as file:
        lines = [x.strip() for x in file.readlines()]
    images = []
    labels = []
    print('Loading dataset to on-memory...')
    width = np.zeros(len(lines), dtype=np.uint16)
    height = np.zeros_like(width)
    labelcnt = np.zeros_like(width)
    for i, imgpath in enumerate(tqdm.tqdm(lines, unit='images')):
        # Load image
        img = cv2.imread(imgpath)
        width[i] = img.shape[1]
        height[i] = img.shape[0]

        # Load label
        labpath = imgpath.replace('images', 'labels').replace(
            'JPEGImages', 'labels').replace('.jpg', '.txt').replace('.png', '.txt')
        label = None
        if os.path.getsize(labpath):
            label = np.loadtxt(labpath)
            labelcnt[i] = label.size / 5
        images.append(img)
        labels.append(label)
    print('''----------------------------------------------------------------------
Images: {} images
Avg. size: {:.1f} x {:.1f}
Max size: {} x {}
Avg. label: {:.1f}
Max label: {}
Nolabel count: {}
----------------------------------------------------------------------
    '''.format(len(lines), width.mean(), height.mean(), width.max(), height.max(), labelcnt.mean(), labelcnt.max(), np.sum(labelcnt == 0)))
    return lines, images, labels
