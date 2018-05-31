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
import scipy
import zipfile
from contextlib import contextmanager
import numpy as np
import nnabla as nn
from nnabla.logger import logger
from nnabla.utils.data_iterator import data_iterator
from nnabla.utils.data_source import DataSource
from nnabla.utils.data_source_loader import download


def load_cyclegan_dataset(dataset="horse2zebra", train=True, domain="A",
                          normalize_method=lambda x: (x - 127.5) / 127.5):
    '''
    Load CycleGAN dataset from `here <https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/>`_ 

    This function assumes that there are two domains in the dataset.

    Args:
        dataset (str): Dataset name excluding ".zip" extension, which you can find that `here <https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/>`_.
        train (bool): The testing dataset will be returned if False. Training data has 60000 images, while testing has 10000 images.
        domain (str): Domain name. It must be "A" or "B".
        normalize_method: Function of how to normalize an image.
    Returns:
        (np.ndarray, list): Images and filenames.

    '''
    assert domain in ["A", "B"]

    image_uri = 'https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/{}.zip'.format(
        dataset)
    logger.info('Getting {} data from {}.'.format(dataset, image_uri))
    r = download(image_uri)

    # Load unpaired images from zipfile.
    with zipfile.ZipFile(r, "r") as zf:
        images = []
        filename_list = []
        dirname = "{}{}".format("train" if train else "test", domain)

        # filter images by name
        zipinfos = filter(
            lambda zinfo: dirname in zinfo.filename and ".jpg" in zinfo.filename,
            zf.infolist())
        for zipinfo in zipinfos:
            with zf.open(zipinfo.filename, "r") as fp:
                # filename
                filename = zipinfo.filename
                logger.info('loading {}'.format(filename))

                # load image
                image = scipy.misc.imread(fp, mode="RGB")
                #image = scipy.misc.imread(fp)
                image = np.transpose(image, (2, 0, 1))
                image = normalize_method(image)
                image_name, ext = os.path.splitext(filename.split("/")[-1])
                images.append(image)
                filename_list.append(image_name)
    r.close()
    logger.info('Getting image data done.')
    return np.asarray(images), filename_list


class CycleGANDataSource(DataSource):
    """Cycle GAN DataSource
    """

    def _get_data(self, position):
        image = self._images[self._indices[position]]
        return image, None

    def __init__(self, dataset="horse2zebra", train=True, domain="A", shuffle=False, rng=None):
        super(CycleGANDataSource, self).__init__(shuffle=shuffle)

        if rng is None:
            rng = np.random.RandomState(313)

        images, filename_list = load_cyclegan_dataset(
            dataset=dataset, train=train, domain=domain)
        self._images = images
        self._size = len(self._images)
        self._variables = ('x', 'y')  # y is dummy
        self.rng = rng
        self.filename_list = filename_list
        self.reset()

    def reset(self):
        self._indices = self.rng.permutation(self._size) \
            if self._shuffle else np.arange(self._size)
        return super(CycleGANDataSource, self).reset()


def cycle_gan_data_source(dataset, train=True, domain="A", shuffle=False, rng=None):
    return CycleGANDataSource(dataset=dataset,
                              train=train, domain=domain, shuffle=shuffle, rng=rng)


def cycle_gan_data_iterator(data_source, batch_size):
    return data_iterator(data_source,
                         batch_size=batch_size,
                         with_memory_cache=False,
                         with_file_cache=False)
