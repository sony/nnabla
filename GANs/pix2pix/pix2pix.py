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

from __future__ import print_function
import os
import argparse
import numpy as np
from PIL import Image

# nnabla imports
import nnabla as nn
import nnabla.logger as logger
import nnabla.monitor as nm
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S

# import my u-net definition
import unet
# import my data iterator
import facade
# import generation functions
from generator import generate, normalize_image, label_to_image

from args import get_args


def train(generator, discriminator, patch_gan, solver_gen, solver_dis,
          weight_l1, train_iterator, val_iterator, epoch, monitor, interval):
    # Create Network Graph
    # for training
    im, la = train_iterator.next()  # for checking image shape
    real = nn.Variable(im.shape)  # real
    x = nn.Variable(la.shape)  # x
    # for validation
    real_val = nn.Variable(im.shape)  # real
    x_val = nn.Variable(la.shape)  # x

    # Generator
    fake = generator(x, test=False)
    # pix2pix infers just like training mode.
    fake_val = generator(x_val, test=False)
    fake_val.persistent = True  # Keep to visualize
    # Discriminator
    fake_y = discriminator(x, fake, patch_gan=patch_gan, test=False)
    real_y = discriminator(x, real, patch_gan=patch_gan, test=False)
    real_target = nn.Variable(fake_y.shape)
    real_target.data.fill(1)
    fake_target = nn.Variable(real_y.shape)
    fake_target.data.zero()

    loss_gen = F.mean(weight_l1 * F.abs(real - fake)) + \
        F.mean(F.sigmoid_cross_entropy(fake_y, real_target))
    loss_dis = F.mean(
        F.sigmoid_cross_entropy(real_y, real_target) + F.sigmoid_cross_entropy(fake_y, fake_target))

    # Setting Solvers
    with nn.parameter_scope('generator'):
        solver_gen.set_parameters(nn.get_parameters())

    with nn.parameter_scope('discriminator'):
        solver_dis.set_parameters(nn.get_parameters())

    # Create Monitors
    monitors = {
        'loss_gen': nm.MonitorSeries("Generator loss", monitor, interval=interval),
        'loss_dis': nm.MonitorSeries("Discriminator loss", monitor, interval=interval),
        'time': nm.MonitorTimeElapsed("Training time", monitor, interval=interval),
        'fake': nm.MonitorImageTile("Fake images", monitor, interval=interval,
                                    num_images=2, normalize_method=lambda x: x),
    }

    i = 0
    for e in range(epoch):
        logger.info('Epoch = {}'.format(e))
        # Training
        while e == train_iterator.epoch:
            # forward / backward process
            real.d, x.d = train_iterator.next()
            solver_dis.zero_grad()
            solver_gen.zero_grad()
            # Discriminator
            loss_dis.forward(clear_no_need_grad=True)
            loss_dis.backward(clear_buffer=True)
            solver_dis.update()
            # Generator
            loss_gen.forward(clear_no_need_grad=True)
            loss_gen.backward(clear_buffer=True)
            solver_gen.update()
            monitors['time'].add(i)
            monitors['loss_gen'].add(i, loss_gen.d.copy())
            monitors['loss_dis'].add(i, loss_dis.d.copy())
            # Validation
            real_val.d, x_val.d = val_iterator.next()
            fake_val.forward()
            pix2pix_vis = np.stack([
                label_to_image(x_val.d),
                normalize_image(fake_val.d)], axis=1).reshape((-1, ) + fake.shape[1:])
            monitors['fake'].add(i, pix2pix_vis)
            i += 1
    # save parameters of generator
    save_path = os.path.join(
        monitor._save_path, 'generator_model_{}.h5'.format(i))
    with nn.parameter_scope('generator'):
        nn.save_parameters(save_path)

    return save_path


def main():
    # argparse
    args = get_args()

    # Context Setting
    # Get context.
    from nnabla.ext_utils import get_extension_context
    logger.info("Running in %s" % args.context)
    ctx = get_extension_context(
        args.context, device_id=args.device_id)
    nn.set_default_context(ctx)

    model_path = args.model

    if args.train:
        # Data Loading
        logger.info("Initialing DataSource.")
        train_iterator = facade.facade_data_iterator(
            args.traindir,
            args.batchsize,
            shuffle=True,
            with_memory_cache=False)
        val_iterator = facade.facade_data_iterator(
            args.valdir,
            args.batchsize,
            random_crop=False,
            shuffle=False,
            with_memory_cache=False)

        monitor = nm.Monitor(args.logdir)
        solver_gen = S.Adam(alpha=args.lrate, beta1=args.beta1)
        solver_dis = S.Adam(alpha=args.lrate, beta1=args.beta1)

        generator = unet.generator
        discriminator = unet.discriminator

        model_path = train(generator, discriminator, args.patch_gan,
                           solver_gen, solver_dis,
                           args.weight_l1, train_iterator, val_iterator,
                           args.epoch, monitor, args.monitor_interval)

    if args.generate:
        if model_path is not None:
            # Data Loading
            logger.info("Generating from DataSource.")
            test_iterator = facade.facade_data_iterator(
                args.testdir,
                args.batchsize,
                shuffle=False,
                with_memory_cache=False)
            generator = unet.generator
            generate(generator, model_path, test_iterator, args.logdir)
        else:
            logger.error("Trained model was NOT given.")


if __name__ == '__main__':
    main()
