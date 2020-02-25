# Copyright (c) 2019 Sony Corporation. All Rights Reserved.
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
Provide data iterator for instaGAN.
"""

import os
import numpy as np
import nnabla as nn
from nnabla.logger import logger
from nnabla.utils.data_iterator import data_iterator
from nnabla.utils.data_source import DataSource
from nnabla.utils.image_utils import imread
import glob


def load_instagan_dataset(args, train=True, domain="A"):
    ''' 
    Load InstaGAN dataset.
    This function assumes that there are two domains in the dataset.
    Args:
        dataset (str): Directory which contains the Dataset.
        train (bool): The testing dataset will be returned if False.
        domain (str): Domain name. It must be "A" or "B".
    Returns:
        (np.ndarray, np.ndarray): images and masks.
    '''

    assert domain in ["A", "B"]

    mode = "train" if train else "val"

    logger.info('Getting data from {}...'.format(args.dataroot))

    img_paths = glob.glob(args.dataroot + "/" + mode + domain + "/*")
    seg_paths = glob.glob(args.dataroot + "/" + mode + domain + "_seg/*")
    img_paths.sort()
    seg_paths.sort()

    max_segs = max([int(_.split("_")[-1].split(".")[0])
                    for _ in seg_paths]) + 1  # get the maximum segmentation masks
    seg_dict = dict()  # {..., img_path: num of masks, ...}"
    for i in range(max_segs):
        query = "_{}.png".format(i)
        extracted = [_.replace("_seg/", "/").split(query)
                     [0] + ".png" for _ in seg_paths if query in _]
        for k in extracted:
            seg_dict[k] = i

    # extract only one seg path per image
    seg_paths = [_ for _ in seg_paths if "_0.png" in _]

    # can be applied only on training dataset.
    assert all([img.split("/")[-1].split(".png")[0] == seg.split("/")
                [-1].split("_")[0] for img, seg in zip(img_paths, seg_paths)])

    images = []
    masks = []
    filename_list = []
    maskname_list = []

    # filter images by name
    for img_path, seg_path in zip(img_paths, seg_paths):

        # load (and resize) image and mask
        image = imread(img_path, size=(args.loadSizeW, args.loadSizeH),
                       interpolate="bicubic", num_channels=3, channel_first=True)
        # clip image's value
        if image.dtype == np.uint8:
            image = image / 255.0  # [0, 255] ->  [0.0, 1.0]
        image = (image - 0.5) / 0.5  # normalize

        combined = image
        num_of_masks = seg_dict[img_path] + 1
        for j in range(num_of_masks):
            curr_seg_path = seg_path.replace("_0.", "_{}.".format(j))
            mask = imread(curr_seg_path, size=(args.loadSizeW, args.loadSizeH),
                          interpolate="bicubic", num_channels=1, channel_first=True, grayscale=True)
            if mask.dtype == np.uint8:
                mask = mask / 255.0  # [0, 255] ->  [0.0, 1.0]
            mask = (mask - 0.5) / 0.5  # normalize
            combined = np.concatenate((combined, mask), axis=0)

        for k in range(max_segs - num_of_masks):
            mask = (-1) * np.ones((1, args.loadSizeH, args.loadSizeW))
            combined = np.concatenate((combined, mask), axis=0)

        image = combined[:3, :, :]
        mask = combined[3:, :, :]

        image_name, ext = os.path.splitext(img_path.split("/")[-1])
        images.append(image)
        filename_list.append(image_name)

        mask_name, ext = os.path.splitext(seg_path.split("/")[-1])
        masks.append(mask)
        maskname_list.append(mask_name)

    return np.asarray(images), np.asarray(masks)


class InstaGANDataSource(DataSource):
    """InstaGAN DataSource
    """

    def _get_data(self, position):
        image = self._images[self._indices[position]]
        mask = self._masks[self._indices[position]]
        return image, mask

    def __init__(self, args, train=True, domain="A", shuffle=False, rng=None):
        super(InstaGANDataSource, self).__init__(shuffle=shuffle)

        if rng is None:
            rng = np.random.RandomState(313)

        images, masks = load_instagan_dataset(args, train=train, domain=domain)
        self._images = images
        self._masks = masks
        self._size = len(self._images)
        self._variables = ('x', 'y')  # y is dummy
        self.rng = rng
        #self.filename_list = filename_list
        self.reset()

    def reset(self):
        self._indices = self.rng.permutation(self._size) \
            if self._shuffle else np.arange(self._size)
        return super(InstaGANDataSource, self).reset()


def insta_gan_data_source(args, train=True, domain="A", shuffle=False, rng=None):
    return InstaGANDataSource(args=args,
                              train=train, domain=domain, shuffle=shuffle, rng=rng)


def insta_gan_data_iterator(data_source, batch_size):
    return data_iterator(data_source,
                         batch_size=batch_size,
                         with_memory_cache=False,
                         with_file_cache=False)
