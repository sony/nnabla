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
from nnabla.utils.data_iterator import data_iterator
from nnabla.utils.data_source import DataSource

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


class YoloDataSource(DataSource):
    '''
    data iterator for yolo 
    '''

    def __init__(self, root, args, shuffle=False, train=False, image_sizes=None, image_size_change_freq=6400, on_memory_data=None, use_cv2=True, shape=None):
        super(YoloDataSource, self).__init__(shuffle=shuffle)
        if not allow_on_memory_data or not use_cv2:
            print('On-memory data cannot be used if cv2 package is not available.')
            on_memory_data = None
        self.on_memory_data = on_memory_data
        if on_memory_data is None:
            with open(root, 'r') as file:
                self.lines = file.readlines()
            self._size = len(self.lines)
        else:
            assert use_cv2
            self._size = len(on_memory_data[0])
        if shuffle:
            self.indexes = np.random.permutation(self._size)
        else:
            self.indexes = np.arange(self._size)
        if train:
            self.jitter = args.jitter
            self.hue = args.hue
            self.saturation = args.saturation
            self.exposure = args.exposure

        self.train = train

        # The attribute self.seen maintains how many examples have been
        # created after creating this data source. This will be used
        # to determine the timing of size change for image size
        # augmentation.
        #
        # WARNING: This initialization with -1 relies on an undocumented
        # behavior of super classs DataSource where `_get_data(self, position)`
        # is called to obtain the input size and attributes before the first
        # call of next() method at `__init__(self, ...)`. This seen counter
        # will fail if this behavior changes in the super class.
        self.seen = -1

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

        self._variables = ("img", "labels")

    def _get_data(self, index):
        # random.seed(index)
        assert index <= len(self), 'index range error'
        if self.train and self.seen % self.image_size_change_freq == 0:
            # TODO: Use the max image size at the end of training.
            # The original darknet implementation does, but not sure how much it
            # affects the performance in practical use-cases.
            width = random.choice(self.image_sizes)
            self.shape = (width, width)

        if self.on_memory_data is None:
            imgpath = self.lines[self.indexes[index]].rstrip()
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

    def reset(self):
        if self._shuffle:
            self._indexes = self._rng.permutation(self._size)
        else:
            self._indexes = np.arange(self._size)
        super(YoloDataSource, self).reset()

    def __len__(self):
        return self._size


def data_iterator_yolo(root, args, batch_size, shuffle=True, train=False, image_sizes=None, image_size_change_freq=640, on_memory_data=None, use_cv2=True, shape=None):

    # "dataItertor for YoloDataSource"
    assert image_size_change_freq % batch_size == 0, 'image_size_change_freq should be divisible by batch_size'

    return data_iterator(YoloDataSource(root, args, shuffle=shuffle, train=train, image_sizes=image_sizes, image_size_change_freq=image_size_change_freq, on_memory_data=on_memory_data, use_cv2=use_cv2, shape=shape), batch_size=batch_size)


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
