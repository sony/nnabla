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


import os
import numpy as np
import nnabla as nn
import nnabla.logger as logger
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
from nnabla.monitor import Monitor, MonitorSeries, MonitorTimeElapsed, MonitorImageTile
import nnabla.utils.save as save
from nnabla.ext_utils import get_extension_context
from args import get_args, save_args

from helpers import denormalize
from models import generator, discriminator, gan_loss
from cifar10_data import data_iterator_cifar10


def generate(args):
    # Context
    ctx = get_extension_context(
        args.context, device_id=args.device_id, type_config=args.type_config)
    nn.set_default_context(ctx)

    # Args
    latent = args.latent
    maps = args.maps
    batch_size = args.batch_size

    # Generator
    nn.load_parameters(args.model_load_path)
    z_test = nn.Variable([batch_size, latent])
    x_test = generator(z_test, maps=maps, test=True, up=args.up)

    # Monitor
    monitor = Monitor(args.monitor_path)
    monitor_image_tile_test = MonitorImageTile("Image Tile Generated", monitor,
                                               num_images=batch_size,
                                               interval=1,
                                               normalize_method=denormalize)

    # Generation iteration
    for i in range(args.num_generation):
        z_test.d = np.random.randn(batch_size, latent)
        x_test.forward(clear_buffer=True)
        monitor_image_tile_test.add(i, x_test)


def main():
    args = get_args()
    save_args(args, "generate")

    generate(args)


if __name__ == '__main__':
    main()
