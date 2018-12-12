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


"""
Provide data iterator for horse2zebra examples.
"""

import os
import scipy.misc
import zipfile
from contextlib import contextmanager
import numpy as np
import nnabla as nn
from nnabla.logger import logger
from nnabla.utils.data_iterator import data_iterator, data_iterator_simple
from nnabla.utils.data_source import DataSource
from nnabla.utils.data_source_loader import download, get_data_home

#import cv2
import tarfile
import glob

def load_pix2pix_dataset(dataset="edges2shoes", train=True, num_samples=-1):
    image_uri = 'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/{}.tar.gz'\
                .format(dataset)
    logger.info('Getting {} data from {}.'.format(dataset, image_uri))
    r = download(image_uri)

    # Load concatenated images, then save separately
    # TODO: how to do for test
    img_A_list = []
    img_B_list = []
    img_names = []
    with tarfile.open(fileobj=r, mode="r") as tar:
        cnt = 0
        for tinfo in tar.getmembers():
            if not ".jpg" in tinfo.name:
                continue
            if not ((train == True and "train" in tinfo.name)
                    or (train == False and "val" in tinfo.name)):
                continue

            logger.info("Loading {} ...".format(tinfo.name))
            f = tar.extractfile(tinfo)
            img = scipy.misc.imread(f, mode="RGB")
            h, w, c = img.shape
            img_A = img[:, 0:w // 2, :].transpose((2, 0, 1))
            img_B = img[:, w // 2:, :].transpose((2, 0, 1))
            img_A_list.append(img_A)
            img_B_list.append(img_B)
            img_names.append(tinfo.name.split("/")[-1])
            cnt += 1

            if num_samples != -1 and cnt >= num_samples:
                break

    r.close()
    logger.info('Getting image data done.')
    img_A, img_B = np.asarray(img_A_list), np.asarray(img_B_list)
    return img_A, img_B, img_names


class Edges2ShoesDataSource(DataSource):

    def _get_data(self, position):
        if self._paired:
            images_A = self._images_A[self._indices[position]]
            images_B = self._images_B[self._indices[position]]
        else:
            images_A = self._images_A[self._indices_A[position]]
            images_B = self._images_B[self._indices_B[position]]
        return self._normalize_method(images_A), self._normalize_method(images_B)

    def __init__(self, dataset="edges2shoes", train=True, paired=True,
                 shuffle=False, rng=None,
                 normalize_method=lambda x: (x - 127.5) / 127.5,
                 num_samples=-1):
        super(Edges2ShoesDataSource, self).__init__(shuffle=shuffle)

        if rng is None:
            rng = np.random.RandomState(313)

        images_A, images_B, names = load_pix2pix_dataset(dataset=dataset, train=train,
                                                         normalize_method=normalize_method,
                                                         num_samples=num_samples)
        self._images_A = images_A
        self._images_B = images_B
        # since A and B is the paired image, lengths are the same
        self._size = len(self._images_A)
        self._size_A = len(self._images_A)
        self._size_B = len(self._images_B)
        self._variables = ('A', 'B')
        self._paired = paired
        self._normalize_method = normalize_method
        self.rng = rng
        self.reset()

    def reset(self):
        if self._paired:
            self._indices = self.rng.permutation(self._size) \
                if self._shuffle else np.arange(self._size)

        else:
            self._indices_A = self.rng.permutation(self._size_A) \
                if self._shuffle else np.arange(self._size_A)
            self._indices_B = self.rng.permutation(self._size_B) \
                if self._shuffle else np.arange(self._size_B)
        return super(Edges2ShoesDataSource, self).reset()


def pix2pix_data_source(dataset, train=True, paired=True, shuffle=False, rng=None, num_samples=-1):
    return Edges2ShoesDataSource(dataset=dataset,
                                 train=train, paired=paired, shuffle=shuffle, rng=rng,
                                 num_samples=num_samples)


def pix2pix_data_iterator(data_source, batch_size):
    return data_iterator(data_source,
                         batch_size=batch_size,
                         with_memory_cache=False,
                         with_file_cache=False)


def edges2shoes_data_iterator(img_path, batch_size=1, 
                              normalize_method=lambda x: (x - 127.5) / 127.5, 
                              num_samples=-1):
    imgs = glob.glob("{}/*.jpg".format(img_path))

    if num_samples == -1:
        num_samples = len(imgs)
    else:
        logger.info(
            "Num. of data ({}) is used for debugging".format(num_samples))

    def load_func(i):
        img = scipy.misc.imread(imgs[i], mode="RGB")
        img = normalize_method(img)
        h, w, c = img.shape
        img_A = img[:, 0:w // 2, :].transpose((2, 0, 1))
        img_B = img[:, w // 2:, :].transpose((2, 0, 1))
        return img_A, img_B

    return data_iterator_simple(load_func, num_samples, batch_size, 
                                shuffle=shuffle, rng=rng, with_file_cache=False)


def prepare_pix2pix_dataset(dataset="edges2shoes", train=True):
    imgs_A, imgs_B, names = load_pix2pix_dataset(dataset=dataset, train=train)
    dname = "train" if train else "val"
    dpath_A = os.path.join(get_data_home(), "{}_A".format(dataset), dname)
    dpath_B = os.path.join(get_data_home(), "{}_B".format(dataset), dname)

    def save_img(dpath, names, imgs):
        if os.path.exists(dpath):
            os.rmdir(dpath)
            os.makedirs(dpath)
        else:
            os.makedirs(dpath)
        for name, img in zip(names, imgs):
            fpath = os.path.join(dpath, name)
            img = img.transpose((1, 2, 0))
            logger.info("Save img to {}".format(fpath))
            scipy.misc.imsave(fpath, img)
    save_img(dpath_A, names, imgs_A)
    save_img(dpath_B, names, imgs_B)


def munit_data_iterator(img_path, batch_size=1, image_size=256, num_samples=-1,
                        normalize_method=lambda x: (x - 127.5) / 127.5, 
                        shuffle=True,
                        rng=None):
    imgs = []
    if type(img_path) == list:
        for p in img_path:
            imgs.append(p)
    elif os.path.isdir(img_path):
        imgs += glob.glob("{}/*.jpg".format(img_path))
        imgs += glob.glob("{}/*.JPG".format(img_path))
        imgs += glob.glob("{}/*.jpeg".format(img_path))
        imgs += glob.glob("{}/*.JPEG".format(img_path))
        imgs += glob.glob("{}/*.png".format(img_path))
        imgs += glob.glob("{}/*.PNG".format(img_path))
    elif img_path.endswith(".jpg") or img_path.endswith(".JPG") \
         or img_path.endswith(".jpeg") or img_path.endswith(".JPEG") \
         or img_path.endswith(".png") or img_path.endswith(".PNG"):
        imgs.append(img_path)
    else:
        raise ValueError("Path specified is not `directory path` or `list of files`.")

    if num_samples == -1:
        num_samples = len(imgs)
    else:
        logger.info(
            "Num. of data ({}) is used for debugging".format(num_samples))

    def load_func(i):
        img = scipy.misc.imread(imgs[i], mode="RGB")
        img = scipy.misc.imresize(img, (image_size, image_size))
        img = normalize_method(img)
        img = img.transpose((2, 0, 1))
        return img, None

    return data_iterator_simple(load_func, num_samples, batch_size, 
                                shuffle=shuffle, rng=rng, with_file_cache=False)
    

if __name__ == '__main__':
    # Hand-made test
    dataset = "edges2shoes"
    num_samples = 100
    ds = Edges2ShoesDataSource(dataset=dataset, num_samples=num_samples)
    di = pix2pix_data_iterator(ds, batch_size=1)
