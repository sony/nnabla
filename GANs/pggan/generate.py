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
from nnabla import Variable
from nnabla import logger
from nnabla.ext_utils import get_extension_context
from nnabla.monitor import Monitor
import os

from args import get_args
from functions import pixel_wise_feature_vector_normalization
from helpers import MonitorImageTileWithName
from helpers import load_gen
from networks import Generator, Discriminator
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import numpy as np


def generate_images(model_load_path,
                    batch_size=16, n_latent=512, use_bn=False,
                    hyper_sphere=True, last_act='tanh',
                    use_wscale=True, use_he_backward=False,
                    resolution_list=[4, 8, 16, 32, 64, 128],
                    channel_list=[512, 512, 256, 128, 64, 32]):
    # Generate
    gen = load_gen(model_load_path, use_bn=use_bn, last_act=last_act,
                   use_wscale=use_wscale, use_he_backward=use_he_backward)
    z_data = np.random.randn(batch_size, n_latent, 1, 1)
    z = nn.Variable.from_numpy_array(z_data)
    z = pixel_wise_feature_vector_normalization(z) if hyper_sphere else z
    y = gen(z, test=True)
    return y.d


def generate_interpolated_images(model_load_path,
                                 batch_size=16, n_latent=512,
                                 use_bn=False, hyper_sphere=True, last_act='tanh',
                                 use_wscale=True, use_he_backward=False,
                                 resolution_list=[4, 8, 16, 32, 64, 128],
                                 channel_list=[512, 512, 256, 128, 64, 32]):
    # Generate
    gen = load_gen(model_load_path, use_bn=use_bn, last_act=last_act,
                   use_wscale=use_wscale, use_he_backward=use_he_backward)
    z_data0 = np.random.randn(1, n_latent, 1, 1)
    z_data1 = np.random.randn(1, n_latent, 1, 1)
    imgs = []
    for i in range(batch_size):
        alpha = 1. * i / (batch_size - 1)
        z_data = (1 - alpha) * z_data0 + alpha * z_data1
        z = nn.Variable.from_numpy_array(z_data)
        z = pixel_wise_feature_vector_normalization(z) if hyper_sphere else z
        y = gen(z, test=True)
        imgs.append(y.d)
    imgs = np.concatenate(imgs, axis=0)
    return imgs


def main():
    # Args
    args = get_args()

    # Context
    ctx = get_extension_context(
        args.context, device_id=args.device_id, type_config=args.type_config)
    nn.set_default_context(ctx)
    nn.set_auto_forward(True)

    # Config
    resolution_list = [4, 8, 16, 32, 64, 128]
    channel_list = [512, 512, 256, 128, 64, 32]
    side = 8

    # Monitor
    monitor = Monitor(args.monitor_path)
    monitor_image_tile = MonitorImageTileWithName("Image Tile", monitor,
                                                  num_images=side**2)

    # Generate
    # generate tile images
    imgs = []
    for _ in range(side):
        img = generate_images(args.model_load_path,
                              batch_size=side, use_bn=args.use_bn,
                              n_latent=args.latent, hyper_sphere=args.hyper_sphere,
                              last_act=args.last_act,
                              use_wscale=args.not_use_wscale,
                              use_he_backward=args.use_he_backward,
                              resolution_list=resolution_list, channel_list=channel_list)
        imgs.append(img)
    imgs = np.concatenate(imgs, axis=0)
    monitor_image_tile.add("GeneratedImage", imgs)

    # generate interpolated tile images
    imgs = []
    for _ in range(side):
        img = generate_interpolated_images(args.model_load_path,
                                           batch_size=side, use_bn=args.use_bn,
                                           n_latent=args.latent, hyper_sphere=args.hyper_sphere,
                                           last_act=args.last_act,
                                           use_wscale=args.not_use_wscale,
                                           use_he_backward=args.use_he_backward,
                                           resolution_list=resolution_list, channel_list=channel_list)
        imgs.append(img)
    imgs = np.concatenate(imgs, axis=0)
    monitor_image_tile.add("GeneratedInterpolatedImage", imgs)


if __name__ == "__main__":
    main()
