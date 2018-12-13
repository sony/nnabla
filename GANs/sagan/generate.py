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
import nnabla as nn
import nnabla.logger as logger
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
from nnabla.monitor import Monitor, MonitorImage, MonitorImageTile
import nnabla.utils.save as save
from nnabla.ext_utils import get_extension_context
from args import get_args, save_args

from helpers import generate_random_class, generate_one_class, resample, normalize_method
from models import generator, discriminator, gan_loss
from imagenet_data import data_iterator_imagenet


def generate(args):
    # Communicator and Context
    extension_module = "cudnn"
    ctx = get_extension_context(extension_module, type_config=args.type_config)
    nn.set_default_context(ctx)

    # Args
    latent = args.latent
    maps = args.maps
    batch_size = args.batch_size
    image_size = args.image_size
    n_classes = args.n_classes
    not_sn = args.not_sn
    threshold = args.truncation_threshold

    # Model
    nn.load_parameters(args.model_load_path)
    z = nn.Variable([batch_size, latent])
    y_fake = nn.Variable([batch_size])
    x_fake = generator(z, y_fake, maps=maps, n_classes=n_classes, test=True, sn=not_sn)\
             .apply(persistent=True)

    # Generate All
    if args.generate_all:
        # Monitor
        monitor = Monitor(args.monitor_path)
        name = "Generated Image Tile All"
        monitor_image = MonitorImageTile(name, monitor, interval=1,
                                         num_images=args.batch_size,
                                         normalize_method=normalize_method)
    
        # Generate images for all classes
        for class_id in range(args.n_classes):
            # Generate
            z_data = resample(batch_size, latent, threshold)
            y_data = generate_one_class(class_id, batch_size)
                     
            z.d = z_data
            y_fake.d = y_data
            x_fake.forward(clear_buffer=True)
            monitor_image.add(class_id, x_fake.d)
        return

    # Generate Indivisually
    monitor = Monitor(args.monitor_path)
    name = "Generated Image Tile {}".format(args.class_id) if args.class_id != -1 else "Generated Image Tile"
    monitor_image_tile = MonitorImageTile(name, monitor, interval=1,
                                     num_images=args.batch_size,
                                     normalize_method=normalize_method)
    name = "Generated Image {}".format(args.class_id) if args.class_id != -1 else "Generated Image"
    monitor_image = MonitorImage(name, monitor, interval=1,
                                 num_images=args.batch_size,
                                 normalize_method=normalize_method)
    z_data = resample(batch_size, latent, threshold)
    y_data = generate_random_class(n_classes, batch_size) if args.class_id == -1 else \
             generate_one_class(args.class_id, batch_size)
    z.d = z_data
    y_fake.d = y_data
    x_fake.forward(clear_buffer=True)
    monitor_image.add(0, x_fake.d)
    monitor_image_tile.add(0, x_fake.d)
    

def main():
    args = get_args()
    save_args(args, "generate")

    generate(args)

if __name__ == '__main__':
    main() 

                        
