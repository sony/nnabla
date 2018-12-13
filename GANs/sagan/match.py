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
from nnabla.utils.nnp_graph import NnpLoader
import nnabla.utils.save as save
from nnabla.ext_utils import get_extension_context
from args import get_args, save_args

from helpers import (generate_random_class, generate_one_class, 
                     get_input_and_output, preprocess, resample)
from models import generator, discriminator, gan_loss
from imagenet_data import data_iterator_imagenet


def match(args):
    # Context
    extension_module = "cudnn"
    ctx = get_extension_context(extension_module, device_id=args.device_id,
                                type_config=args.type_config)
    nn.set_default_context(ctx)

    # Args
    latent = args.latent
    maps = args.maps
    batch_size = 1
    image_size = args.image_size
    n_classes = args.n_classes
    not_sn = args.not_sn
    threshold = args.truncation_threshold

    # Model (SAGAN)
    nn.load_parameters(args.model_load_path)
    z = nn.Variable([batch_size, latent])
    y_fake = nn.Variable([batch_size])
    x_fake = generator(z, y_fake, maps=maps, n_classes=n_classes, test=True, sn=not_sn)\
             .apply(persistent=True)

    # Model (Inception model) from nnp file
    nnp = NnpLoader(args.nnp_inception_model_load_path)
    x, h = get_input_and_output(nnp, batch_size, args.variable_name)

    # DataIterator for a given class_id
    di = data_iterator_imagenet(args.train_dir, args.dirname_to_label_path,
                                batch_size=batch_size, n_classes=args.n_classes, 
                                noise=False,
                                class_id=args.class_id)

    # Monitor
    monitor = Monitor(args.monitor_path)
    name = "Matched Image {}".format(args.class_id)
    monitor_image = MonitorImage(name, monitor, interval=1,
                                 num_images=batch_size, 
                                 normalize_method=lambda x: (x + 1.) / 2. * 255.)
    name = "Matched Image Tile {}".format(args.class_id)
    monitor_image_tile = MonitorImageTile(name, monitor, interval=1,
                                          num_images=batch_size + args.top_n, 
                                          normalize_method=lambda x: (x + 1.) / 2. * 255.)

    # Generate and p(h|x).forward
    # generate
    z_data = resample(batch_size, latent, threshold)
    y_data = generate_one_class(args.class_id, batch_size)
    z.d = z_data
    y_fake.d = y_data
    x_fake.forward(clear_buffer=True)
    # p(h|x).forward
    x_fake_d = x_fake.d.copy()
    x_fake_d = preprocess(x_fake_d, (args.image_size, args.image_size), args.nnp_preprocess)
    x.d = x_fake_d
    h.forward(clear_buffer=True)
    h_fake_d = h.d.copy()

    # Feature matching
    norm2_list = []
    x_data_list = []
    x_data_list.append(x_fake.d)
    for i in range(di.size):
        # forward for real data
        x_d, _ = di.next()
        x_data_list.append(x_d)
        x_d = preprocess(x_d, (args.image_size, args.image_size), args.nnp_preprocess)
        x.d = x_d
        h.forward(clear_buffer=True)
        h_real_d = h.d.copy()
        # norm computation
        axis = tuple(np.arange(1, len(h.shape)).tolist())
        norm2 = np.sum((h_real_d - h_fake_d) ** 2.0, axis=axis)
        norm2_list.append(norm2)

    # Save top-n images
    argmins = np.argsort(norm2_list)
    for i in range(args.top_n):
        monitor_image.add(i, x_data_list[i])
    matched_images = np.concatenate(x_data_list)
    monitor_image_tile.add(0, matched_images)


def main():
    args = get_args()
    save_args(args, "match")

    match(args)

if __name__ == '__main__':
    main() 

                        
