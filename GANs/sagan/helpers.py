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


import argparse
from nnabla.contrib.context import extension_context
from nnabla.monitor import Monitor, MonitorImage, MonitorImageTile, MonitorSeries, tile_images
from nnabla.utils.data_iterator import data_iterator
import os

import nnabla as nn
import nnabla.functions as F
import nnabla.logger as logger
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import numpy as np
from scipy import linalg

def generate_random_class(n_classes, batch_size):
    return np.random.choice(np.arange(n_classes),
                            batch_size,
                            replace=False)


def generate_one_class(class_id, batch_size):
    return np.repeat(class_id, batch_size)
    

def get_input_and_output(nnp, batch_size, name=""):
    network_name = nnp.get_network_names()[0]
    net = nnp.get_network(network_name, batch_size=batch_size)
    x = list(net.inputs.values())[0]
    y = list(net.outputs.values())[0]
    if name != "":
        h = net.variables[name]
        return x, h
    return x, y


def denormalize(x):
    x = (x + 1.0) / 2.0 * 255.0
    return x


def normalize_method(x):
    x = ((x + 1.0) / 2.0 * 255.0).astype(np.uint8)
    return x


def nnp_preprocess(x, a=0.01735, b=-1.99):
    x = a * x + b
    return x


def resize_images(images, oshape=(320, 320)):
    import cv2
    images = images.transpose((0, 2, 3, 1))
    images_ = []
    for img in images:
        # others than bilinear get pretty better
        img = cv2.resize(img, oshape, interpolation=cv2.INTER_CUBIC)
        images_.append(img)
    images_ = np.asarray(images_).transpose(0, 3, 1, 2)
    return images_


def preprocess(x_d, oshape, use_nnp_preprocess):
    x_d = denormalize(x_d)
    x_d = nnp_preprocess(x_d) if use_nnp_preprocess else x_d
    x_d = resize_images(x_d, oshape=oshape)
    return x_d


def resample(batch_size, latent, threshold=np.inf, count=100):
    i = 0
    z_fixded = np.random.randn(batch_size * latent)
    idx_fixed = set()
    while i < count:
        z_data = np.random.randn(batch_size * latent)
        idx_candidate = np.where(np.abs(z_data) < threshold)[0]
        idx_update = np.asarray(list(set(idx_candidate.tolist()) - idx_fixed))
        if len(idx_update) == 0:
            i += 1
            continue
        z_fixded[idx_update] = z_data[idx_update]
        idx_fixed = idx_fixed | set(idx_update.tolist())
        i += 1
        if len(idx_fixed) == batch_size * latent:
            break
    return z_fixded.reshape((batch_size, latent))
