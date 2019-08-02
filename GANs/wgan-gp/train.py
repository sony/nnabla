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


def train(args):
    # Context
    ctx = get_extension_context(
        args.context, device_id=args.device_id, type_config=args.type_config)
    nn.set_default_context(ctx)

    # Args
    latent = args.latent
    maps = args.maps
    batch_size = args.batch_size
    image_size = args.image_size
    lambda_ = args.lambda_

    # Model
    # generator loss
    z = nn.Variable([batch_size, latent])
    x_fake = generator(z, maps=maps, up=args.up).apply(persistent=True)
    p_fake = discriminator(x_fake, maps=maps)
    loss_gen = gan_loss(p_fake).apply(persistent=True)
    # discriminator loss
    p_fake = discriminator(x_fake, maps=maps)
    x_real = nn.Variable([batch_size, 3, image_size, image_size])
    p_real = discriminator(x_real, maps=maps)
    loss_dis = gan_loss(p_fake, p_real).apply(persistent=True)
    # gradient penalty
    eps = F.rand(shape=[batch_size, 1, 1, 1])
    x_rmix = eps * x_real + (1.0 - eps) * x_fake
    p_rmix = discriminator(x_rmix, maps=maps)
    x_rmix.need_grad = True  # Enabling gradient computation for double backward
    grads = nn.grad([p_rmix], [x_rmix])
    l2norms = [F.sum(g ** 2.0, [1, 2, 3]) ** 0.5 for g in grads]
    gp = sum([F.mean((l - 1.0) ** 2.0) for l in l2norms])
    loss_dis += lambda_ * gp
    # generator with fixed value for test
    z_test = nn.Variable.from_numpy_array(np.random.randn(batch_size, latent))
    x_test = generator(z_test, maps=maps, test=True,
                       up=args.up).apply(persistent=True)

    # Solver
    solver_gen = S.Adam(args.lrg, args.beta1, args.beta2)
    solver_dis = S.Adam(args.lrd, args.beta1, args.beta2)

    with nn.parameter_scope("generator"):
        params_gen = nn.get_parameters()
        solver_gen.set_parameters(params_gen)
    with nn.parameter_scope("discriminator"):
        params_dis = nn.get_parameters()
        solver_dis.set_parameters(params_dis)

    # Monitor
    monitor = Monitor(args.monitor_path)
    monitor_loss_gen = MonitorSeries(
        "Generator Loss", monitor, interval=10)
    monitor_loss_cri = MonitorSeries(
        "Negative Critic Loss", monitor, interval=10)
    monitor_time = MonitorTimeElapsed(
        "Training Time", monitor, interval=10)
    monitor_image_tile_train = MonitorImageTile("Image Tile Train", monitor,
                                                num_images=batch_size,
                                                interval=1,
                                                normalize_method=denormalize)
    monitor_image_tile_test = MonitorImageTile("Image Tile Test", monitor,
                                               num_images=batch_size,
                                               interval=1,
                                               normalize_method=denormalize)

    # Data Iterator
    di = data_iterator_cifar10(batch_size, True)

    # Train loop
    for i in range(args.max_iter):
        # Train discriminator
        x_fake.need_grad = False  # no need backward to generator
        for _ in range(args.n_critic):
            solver_dis.zero_grad()
            x_real.d = di.next()[0] / 127.5 - 1.0
            z.d = np.random.randn(batch_size, latent)
            loss_dis.forward(clear_no_need_grad=True)
            loss_dis.backward(clear_buffer=True)
            solver_dis.update()

        # Train generator
        x_fake.need_grad = True  # need backward to generator
        solver_gen.zero_grad()
        z.d = np.random.randn(batch_size, latent)
        loss_gen.forward(clear_no_need_grad=True)
        loss_gen.backward(clear_buffer=True)
        solver_gen.update()
        # Monitor
        monitor_loss_gen.add(i, loss_gen.d)
        monitor_loss_cri.add(i, -loss_dis.d)
        monitor_time.add(i)

        # Save
        if i % args.save_interval == 0:
            monitor_image_tile_train.add(i, x_fake)
            monitor_image_tile_test.add(i, x_test)
            nn.save_parameters(os.path.join(
                args.monitor_path, "params_{}.h5".format(i)))

    # Last
    x_test.forward(clear_buffer=True)
    nn.save_parameters(os.path.join(
        args.monitor_path, "params_{}.h5".format(i)))
    monitor_image_tile_train.add(i, x_fake)
    monitor_image_tile_test.add(i, x_test)


def main():
    args = get_args()
    save_args(args, "train")

    train(args)


if __name__ == '__main__':
    main()
