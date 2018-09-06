# Copyright (c) 2018 Sony Corporation. All Rights Reserved.
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
import numpy as np
from nnabla.utils.data_source_loader import download
import zipfile
import imageio

OMNIGLOT_URL = "https://raw.githubusercontent.com/brendenlake/omniglot/master/python/"

if not os.path.exists('omniglot/data'):
    os.makedirs('omniglot/data')


def load_omniglot(test=False):

    fname = "images_background.zip"
    if test:
        fname = "images_evaluation.zip"

    dlname = OMNIGLOT_URL + fname
    r = download(dlname)
    f = zipfile.ZipFile(r, mode="r")

    x = []
    y = []
    imgs = []
    lang_dict = {}
    n_letter = 0
    for path in f.namelist():

        # Four types of "path" is possible
        #  "image_xxx/"
        #  "image_xxx/Alphabet"
        #  "image_xxx/Alphabet/Letter"
        #  "image_xxx/Alphabet/Letter/img.png"

        names = path.split('/')
        if len(names) == 3:  # i.e. [images_xxx, Alphabet, None]
            alphabet = names[1]
            print("loading alphabet: " + alphabet)
            lang_dict[alphabet] = [n_letter, None]
        if len(names) == 4:
            if names[3] is not '':  # i.e. [images_xxx, Alphabet, Letter, Image]
                imgs.append(imageio.imread(f.read(path)))
        if len(imgs) == 20:  # Number of images are limited to 20
            x.append(np.stack(imgs))
            y.append(np.ones(20, ) * n_letter)
            n_letter += 1
            lang_dict[alphabet][1] = n_letter - 1
            imgs = []

    x = np.stack(x)
    y = np.stack(y)
    return x, y, lang_dict


x, y, c = load_omniglot(test=False)
with open("omniglot/data/train.npy", "wb") as f:
    np.save(f, (x, c))

x, y, c = load_omniglot(test=True)
with open("omniglot/data/val.npy", "wb") as f:
    np.save(f, (x, c))
