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
from nnabla.monitor import Monitor, MonitorImage
import nnabla.utils.save as save
from nnabla.ext_utils import get_extension_context
from args import get_args, save_args

from helpers import generate_random_class, generate_one_class, resample, normalize_method
from models import generator, discriminator, gan_loss
from imagenet_data import data_iterator_imagenet


def morph(args):
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
    alpha = nn.Variable.from_numpy_array(np.zeros([1, 1]))
    beta = (nn.Variable.from_numpy_array(np.ones([1, 1])) - alpha)
    y_fake_a = nn.Variable([batch_size])
    
    y_fake_b = nn.Variable([batch_size])
    x_fake = generator(z, [y_fake_a, y_fake_b], maps=maps, n_classes=n_classes, 
                       test=True, sn=not_sn, coefs=[alpha, beta]).apply(persistent=True)
    b, c, h, w = x_fake.shape

    # Monitor
    monitor = Monitor(args.monitor_path)
    name = "Morphed Image {} {}".format(args.from_class_id, args.to_class_id)
    monitor_image = MonitorImage(name, monitor, interval=1,
                                 num_images=1,
                                 normalize_method=normalize_method)
    
    # Morph
    images = []
    z_data = resample(batch_size, latent, threshold)
    z.d = z_data
    for i in range(args.n_morphs):
        alpha.d = 1.0 * i / args.n_morphs
        y_fake_a.d = generate_one_class(args.from_class_id, batch_size)
        y_fake_b.d = generate_one_class(args.to_class_id, batch_size)
        x_fake.forward(clear_buffer=True)
        monitor_image.add(i, x_fake.d)
            

def main():
    args = get_args()
    save_args(args, "morph")

    morph(args)


if __name__ == '__main__':
    main() 

                        
