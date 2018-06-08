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
import numpy as np
from PIL import Image

# nnabla imports
import nnabla as nn
from nnabla.contrib.context import extension_context

# import my u-net definition
import unet


def normalize_image(img):
    return np.clip(img * 128 + 128, 0, 255)


def label_to_image(label):
    # HSV visualization
    ret = np.zeros(label.shape[-2:]+(3,), dtype=np.uint8)
    ret[:, :, 1] = 255
    ret[:, :, 2] = 255
    for c in range(0, label.shape[1]):
        ret[:, :, 0] += np.uint8(15 * (12 - c - .5) * label[0, c, :, :])
    ret = np.asarray(Image.fromarray(ret, mode='HSV').convert('RGB'))
    ret = np.expand_dims(ret.transpose(2, 0, 1), 0)
    return ret


def save_as_image(arr, save_path):
    arr = arr.transpose(1, 2, 0).astype(np.uint8)
    Image.fromarray(arr).save(save_path)


def generate(generator, model_path, data_iterator, out_root):
    # Output Directory Setup
    out_dir = os.path.join(out_root, 'generated')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    _, la = data_iterator.next()  # for checking image shape
    x = nn.Variable(la.shape)  # x
    fake = generator(x, test=False)  # pix2pix infers just like training mode.
    with nn.parameter_scope('generator'):
        nn.load_parameters(model_path)

    for i in range(data_iterator.size):
        _, x.d = data_iterator.next()
        fake.forward()
        save_as_image(label_to_image(
            x.d)[0], '{}/{:03d}_input.png'.format(out_dir, i))
        save_as_image(normalize_image(fake.d)[
                      0], '{}/{:03d}_output.png'.format(out_dir, i))
