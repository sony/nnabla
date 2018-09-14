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


from nnabla import Variable
from nnabla import logger
from nnabla.ext_utils import get_extension_context
from nnabla.monitor import Monitor, MonitorSeries, MonitorTimeElapsed

from args import get_args, save_args
from datasets import data_iterator
from helpers import MonitorImageTileWithName
from networks import Generator, Discriminator
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import numpy as np
from trainer import Trainer


def main():
    # Args
    args = get_args()
    save_args(args)

    # Context
    ctx = get_extension_context(
        args.context, device_id=args.device_id, type_config=args.type_config)
    nn.set_default_context(ctx)
    nn.set_auto_forward(True)

    # Data Iterator
    di = data_iterator(args.img_path, args.batch_size,
                       imsize=(args.imsize, args.imsize),
                       num_samples=args.train_samples,
                       dataset_name=args.dataset_name)
    # Model
    generator = Generator(use_bn=args.use_bn, last_act=args.last_act,
                          use_wscale=args.not_use_wscale, use_he_backward=args.use_he_backward)
    discriminator = Discriminator(use_ln=args.use_ln, alpha=args.leaky_alpha,
                                  use_wscale=args.not_use_wscale, use_he_backward=args.use_he_backward)

    # Solver
    solver_gen = S.Adam(alpha=args.learning_rate,
                        beta1=args.beta1, beta2=args.beta2)
    solver_dis = S.Adam(alpha=args.learning_rate,
                        beta1=args.beta1, beta2=args.beta2)

    # Monitor
    monitor = Monitor(args.monitor_path)
    monitor_loss_gen = MonitorSeries("Generator Loss", monitor, interval=10)
    monitor_loss_dis = MonitorSeries(
        "Discriminator Loss", monitor, interval=10)
    monitor_p_fake = MonitorSeries("Fake Probability", monitor, interval=10)
    monitor_p_real = MonitorSeries("Real Probability", monitor, interval=10)
    monitor_time = MonitorTimeElapsed(
        "Training Time per Resolution", monitor, interval=1)
    monitor_image_tile = MonitorImageTileWithName("Image Tile", monitor,
                                                  num_images=4,
                                                  normalize_method=lambda x: (x + 1.) / 2.)

    # TODO: use argument
    resolution_list = [4, 8, 16, 32, 64, 128]
    channel_list = [512, 512, 256, 128, 64, 32]

    trainer = Trainer(di,
                      generator, discriminator,
                      solver_gen, solver_dis,
                      args.monitor_path,
                      monitor_loss_gen, monitor_loss_dis,
                      monitor_p_fake, monitor_p_real,
                      monitor_time,
                      monitor_image_tile,
                      resolution_list, channel_list,
                      n_latent=args.latent, n_critic=args.critic,
                      save_image_interval=args.save_image_interval,
                      hyper_sphere=args.hyper_sphere,
                      l2_fake_weight=args.l2_fake_weight)

    # TODO: use images per resolution?
    trainer.train(args.epoch_per_resolution)


if __name__ == "__main__":
    main()
