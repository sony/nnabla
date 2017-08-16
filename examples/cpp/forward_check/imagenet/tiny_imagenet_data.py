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

from six.moves import map

from tqdm import tqdm
import glob
import numpy
import os
import zipfile

from nnabla.logger import logger
from nnabla.utils.data_iterator import data_iterator
from nnabla.utils.data_source import DataSource
from nnabla.utils.data_source_loader import get_data_home, download

from nnabla.utils.data_source_loader import load_image


def download_tiny_imagenet():
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    dir_data = os.path.join(get_data_home(), 'tiny-imagenet-200')
    if not os.path.isdir(dir_data):
        f = download(url)
        logger.info('Extracting {} ...'.format(f.name))
        z = zipfile.ZipFile(f)
        d = get_data_home()
        l = z.namelist()
        for i in tqdm(range(len(l))):
            z.extract(l[i], d)
        z.close()
        f.close()
    return dir_data


class TinyImageNetDataSource(DataSource):

    def _get_data(self, _position):
        position = self._indexes[_position]
        img = load_image(self.paths[position])
        if img.shape[0] == 1:
            img = img.repeat(3, 0)
        return (img, self.labels[position])

    def __init__(self, kind, shuffle=True):
        super(TinyImageNetDataSource, self).__init__(shuffle=shuffle)
        assert kind in ['train', 'val', 'test']
        self.dir_data = download_tiny_imagenet()
        classes = numpy.loadtxt(os.path.join(
            self.dir_data, 'wnids.txt'), dtype=str)
        wnid2label = dict(zip(classes, range(len(classes))))
        paths = []
        labels = []
        sort_key = (lambda x: int(x.split('_')[-1].split('.')[0]))
        if kind == 'train':
            for i, k in enumerate(classes):
                kpaths = sorted(glob.glob(os.path.join(
                    self.dir_data, 'train', k, 'images', '*.JPEG')), key=sort_key)
                paths.extend(kpaths)
                labels.append(numpy.ones((len(kpaths), 1), numpy.int16) * i)
            labels = numpy.concatenate(labels, 0)
        elif kind == 'val':
            val_set = numpy.loadtxt(os.path.join(
                self.dir_data, 'val', 'val_annotations.txt'), dtype=str)
            paths = list(map(lambda x: os.path.join(
                self.dir_data, 'val', 'images', x), val_set[:, 0]))
            labels = numpy.array(list(map(lambda x: wnid2label[x], val_set[
                :, 1])), dtype=numpy.int16)[:, numpy.newaxis]
        else:
            assert False, '{} is not implemented.'.format(kind)
        self.paths = paths
        self.labels = labels
        self._size = self.labels.size
        self._variables = ('x', 'y')
        self.rng = numpy.random.RandomState(313)
        self.reset()

    def reset(self):
        if self._shuffle:
            self._indexes = self.rng.permutation(self._size)
        else:
            self._indexes = numpy.arange(self._size)
        super(TinyImageNetDataSource, self).reset()


def data_iterator_tiny_imagenet(batch_size, kind):
    data_source = TinyImageNetDataSource(kind, shuffle=True)
    return data_iterator(data_source, batch_size, False, False, False)
