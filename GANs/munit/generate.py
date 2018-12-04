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
from collections import OrderedDict

from args import get_args, save_args
from helpers import change_need_grad
from models import style_encoder, content_encoder, decoder, discriminators, recon_loss, lsgan_loss
from datasets import munit_data_iterator


def generate(args):
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
    ## content/style (domain A)
    x_content_a = content_encoder(x_real_a, maps, name="content-encoder-a")
    x_style_a = style_encoder(x_real_a, maps, name="style-encoder-a")
    ## content/style (domain B)
    x_content_b = content_encoder(x_real_b, maps, name="content-encoder-b")
    x_style_b = style_encoder(x_real_b, maps, name="style-encoder-b")
    ## generate over domains and reconstruction of content and style (domain A)
    z_style_a = F.randn(shape=x_style_a.shape) if not args.example_guided else x_style_a
    z_style_a = z_style_a.apply(persistent=True)
    x_fake_a = decoder(x_content_b, z_style_a, name="decoder-a")
    ## generate over domains and reconstruction of content and style (domain B)
    z_style_b = F.randn(shape=x_style_b.shape) if not args.example_guided else x_style_b
    z_style_b = z_style_b.apply(persistent=True)
    x_fake_b = decoder(x_content_a, z_style_b, name="decoder-b")

    # Monitor
    suffix = "Stochastic" if not args.example_guided else "Example-guided"
    monitor = Monitor(args.monitor_path)
    monitor_image_a = MonitorImage("Fake Image B to A {} Valid".format(suffix), monitor, interval=1)
    monitor_image_b = MonitorImage("Fake Image A to B {} Valid".format(suffix), monitor, interval=1)

    # DataIterator
    di_a = munit_data_iterator(args.img_path_a, args.batch_size)
    di_b = munit_data_iterator(args.img_path_b, args.batch_size)

    # Generate all
    ## generate (A -> B)
    if args.example_guided:
        x_real_b.d = di_b.next()[0]
    for i in range(di_a.size):
        x_real_a.d = di_a.next()[0]
        images = []
        images.append(x_data_a)
        for _ in range(args.num_repeats):
            x_fake_b.forward(clear_buffer=True)
            images.append(x_fake_b.d.copy())
        monitor_image_b.add(i, np.concatenate(images, axis=3))

    ## generate (B -> A)
    if args.example_guided:
        x_real_a.d = di_a.next()[0]
    for i in range(di_b.size):
        x_real_b.d = di_b.next()[0]
        x_fake_a.forward(clear_buffer=True)
        images = []
        images.append(x_data_b)
        for _ in range(args.num_repeats):
            x_fake_a.forward(clear_buffer=True)
            images.append(x_fake_a.d.copy())
        monitor_image_a.add(i, np.concatenate(images, axis=3))


def main():
    args = get_args()
    save_args(args, "generate")
    generate(args)


if __name__ == '__main__':
    main()

