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
import argparse
import nnabla as nn
import nnabla.logger as logger
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
from nnabla.monitor import Monitor, MonitorImage, MonitorImageTile, MonitorSeries, MonitorTimeElapsed
from nnabla.ext_utils import get_extension_context
from functools import reduce
from collections import OrderedDict

from args import get_args, save_args
from helpers import change_need_grad
from models import style_encoder, content_encoder, decoder, discriminators, recon_loss, lsgan_loss
from datasets import munit_data_iterator


def interpolate(args):
    # Load model
    nn.load_parameters(args.model_load_path)

    # Context
    extension_module = "cudnn"
    ctx = get_extension_context(extension_module, type_config=args.type_config)
    nn.set_default_context(ctx)

    # Input
    b, c, h, w = 1, 3, args.image_size, args.image_size
    x_real_a = nn.Variable([b, c, h, w])
    x_real_b = nn.Variable([b, c, h, w])
    one = nn.Variable.from_numpy_array(np.ones((1, 1, 1, 1)) * 0.5)

    # Model
    maps = args.maps
    # content/style (domain A)
    x_content_a = content_encoder(x_real_a, maps, name="content-encoder-a")
    x_style_a = style_encoder(x_real_a, maps, name="style-encoder-a")
    # content/style (domain B)
    x_content_b = content_encoder(x_real_b, maps, name="content-encoder-b")
    x_style_b = style_encoder(x_real_b, maps, name="style-encoder-b")
    # generate over domains and reconstruction of content and style (domain A)
    z_style_a = nn.Variable(
        x_style_a.shape) if not args.example_guided else x_style_a
    z_style_a = z_style_a.apply(persistent=True)
    x_fake_a = decoder(x_content_b, z_style_a, name="decoder-a")
    # generate over domains and reconstruction of content and style (domain B)
    z_style_b = nn.Variable(
        x_style_b.shape) if not args.example_guided else x_style_b
    z_style_b = z_style_b.apply(persistent=True)
    x_fake_b = decoder(x_content_a, z_style_b, name="decoder-b")

    # Monitor
    def file_names(path): return path.split("/")[-1].rstrip("_AB.jpg")
    suffix = "Stochastic" if not args.example_guided else "Example-guided"
    monitor = Monitor(args.monitor_path)
    monitor_image_tile_a = MonitorImageTile("Fake Image Tile {} B to A {} Interpolation".format(
        "-".join([file_names(path) for path in args.img_files_b]), suffix), monitor,
                                            interval=1, num_images=len(args.img_files_b))
    monitor_image_tile_b = MonitorImageTile("Fake Image Tile {} A to B {} Interpolation".format(
        "-".join([file_names(path) for path in args.img_files_a]), suffix), monitor,
                                            interval=1, num_images=len(args.img_files_a))

    # DataIterator
    di_a = munit_data_iterator(args.img_files_a, b, shuffle=False)
    di_b = munit_data_iterator(args.img_files_b, b, shuffle=False)
    rng = np.random.RandomState(args.seed)

    # Interpolate (A -> B)
    z_data_0 = [rng.randn(*z_style_a.shape) for j in range(di_a.size)]
    z_data_1 = [rng.randn(*z_style_a.shape) for j in range(di_a.size)]
    for i in range(args.num_repeats):
        r = 1.0 * i / args.num_repeats
        images = []
        for j in range(di_a.size):
            x_data_a = di_a.next()[0]
            x_real_a.d = x_data_a
            z_style_b.d = z_data_0[j] * (1.0 - r) + z_data_1[j] * r
            x_fake_b.forward(clear_buffer=True)
            cmp_image = np.concatenate([x_data_a, x_fake_b.d.copy()], axis=3)
            images.append(cmp_image)
        images = np.concatenate(images)
        monitor_image_tile_b.add(i, images)

    # Interpolate (B -> A)
    z_data_0 = [rng.randn(*z_style_b.shape) for j in range(di_b.size)]
    z_data_1 = [rng.randn(*z_style_b.shape) for j in range(di_b.size)]
    for i in range(args.num_repeats):
        r = 1.0 * i / args.num_repeats
        images = []
        for j in range(di_b.size):
            x_data_b = di_b.next()[0]
            x_real_b.d = x_data_b
            z_style_a.d = z_data_0[j] * (1.0 - r) + z_data_1[j] * r
            x_fake_a.forward(clear_buffer=True)
            cmp_image = np.concatenate([x_data_b, x_fake_a.d.copy()], axis=3)
            images.append(cmp_image)
        images = np.concatenate(images)
        monitor_image_tile_a.add(i, images)


def main():
    args = get_args()
    save_args(args, "generate")
    interpolate(args)


if __name__ == '__main__':
    main()
