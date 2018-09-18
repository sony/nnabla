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
from nnabla.monitor import Monitor, MonitorImage, MonitorSeries
from nnabla.ext_utils import get_extension_context

from args import get_args, save_args
from cycle_gan_data import cycle_gan_data_source, cycle_gan_data_iterator
import models


class ImagePool(object):
    """ImagePool for the history of generated images"""

    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.pool = []
        self.size = 0

    def insert_then_get(self, image):
        if self.size < self.pool_size:
            image_ = image.copy()
            self.pool.append(image_)
            self.size += 1
        else:
            idx = np.random.choice(self.pool_size)
            image_ = self.pool[idx]
            self.pool[idx] = image.copy()
        return image_


def linear_decay(solver, base_lr, epoch, max_epoch):
    """Decay towards zero after 100 epoch"""
    numerator = max(0, epoch - 98)
    lr = base_lr * (1. - numerator / float(max_epoch - numerator))
    logger.info("Msg:Learning rate decayed,epoch:{},lr:{}".format(epoch, lr))
    solver.set_learning_rate(lr)


def image_augmentation(image):
    h = F.image_augmentation(image,
                             shape=image.shape,
                             min_scale=1.0,
                             max_scale=286.0/256.0,  # == 1.1171875
                             flip_lr=True)
    h.persistent = True
    return h


def train(args):
    # Settings
    b, c, h, w = 1, 3, 256, 256
    beta1 = 0.5
    beta2 = 0.999
    pool_size = 50
    lambda_recon = args.lambda_recon
    lambda_idt = args.lambda_idt
    base_lr = args.learning_rate
    init_method = args.init_method

    # Context
    extension_module = args.context
    if args.context is None:
        extension_module = 'cpu'
    logger.info("Running in %s" % extension_module)
    ctx = get_extension_context(extension_module,
                                device_id=args.device_id, type_config=args.type_config)
    nn.set_default_context(ctx)

    # Inputs
    x_raw = nn.Variable([b, c, h, w], need_grad=False)
    y_raw = nn.Variable([b, c, h, w], need_grad=False)
    x_real = image_augmentation(x_raw)
    y_real = image_augmentation(y_raw)
    x_history = nn.Variable([b, c, h, w])
    y_history = nn.Variable([b, c, h, w])
    x_real_test = nn.Variable([b, c, h, w], need_grad=False)
    y_real_test = nn.Variable([b, c, h, w], need_grad=False)

    # Models for training
    # Generate
    y_fake = models.g(x_real, unpool=args.unpool, init_method=init_method)
    x_fake = models.f(y_real, unpool=args.unpool, init_method=init_method)
    y_fake.persistent, x_fake.persistent = True, True
    # Reconstruct
    x_recon = models.f(y_fake, unpool=args.unpool, init_method=init_method)
    y_recon = models.g(x_fake, unpool=args.unpool, init_method=init_method)
    # Discriminate
    d_y_fake = models.d_y(y_fake, init_method=init_method)
    d_x_fake = models.d_x(x_fake, init_method=init_method)
    d_y_real = models.d_y(y_real, init_method=init_method)
    d_x_real = models.d_x(x_real, init_method=init_method)
    d_y_history = models.d_y(y_history, init_method=init_method)
    d_x_history = models.d_x(x_history, init_method=init_method)

    # Models for test
    y_fake_test = models.g(
        x_real_test, unpool=args.unpool, init_method=init_method)
    x_fake_test = models.f(
        y_real_test, unpool=args.unpool, init_method=init_method)
    y_fake_test.persistent, x_fake_test.persistent = True, True
    # Reconstruct
    x_recon_test = models.f(
        y_fake_test, unpool=args.unpool, init_method=init_method)
    y_recon_test = models.g(
        x_fake_test, unpool=args.unpool, init_method=init_method)

    # Losses
    # Reconstruction Loss
    loss_recon = models.recon_loss(x_recon, x_real) \
        + models.recon_loss(y_recon, y_real)
    # Generator loss
    loss_gen = models.lsgan_loss(d_y_fake) \
        + models.lsgan_loss(d_x_fake) \
        + lambda_recon * loss_recon
    # Idendity loss
    if lambda_idt != 0:
        logger.info("Idendity loss was added.")
        # Identity
        y_idt = models.g(y_real, unpool=args.unpool, init_method=init_method)
        x_idt = models.f(x_real, unpool=args.unpool, init_method=init_method)
        loss_idt = models.recon_loss(x_idt, x_real) \
            + models.recon_loss(y_idt, y_real)
        loss_gen += lambda_recon * lambda_idt * loss_idt
    # Discriminator losses
    loss_dis_y = models.lsgan_loss(d_y_history, d_y_real)
    loss_dis_x = models.lsgan_loss(d_x_history, d_x_real)

    # Solvers
    solver_gen = S.Adam(base_lr, beta1, beta2)
    solver_dis_x = S.Adam(base_lr, beta1, beta2)
    solver_dis_y = S.Adam(base_lr, beta1, beta2)
    with nn.parameter_scope('generator'):
        solver_gen.set_parameters(nn.get_parameters())
    with nn.parameter_scope('discriminator'):
        with nn.parameter_scope("x"):
            solver_dis_x.set_parameters(nn.get_parameters())
        with nn.parameter_scope("y"):
            solver_dis_y.set_parameters(nn.get_parameters())

    # Datasets
    rng = np.random.RandomState(313)
    ds_train_B = cycle_gan_data_source(
        args.dataset, train=True, domain="B", shuffle=True, rng=rng)
    ds_train_A = cycle_gan_data_source(
        args.dataset, train=True, domain="A", shuffle=True, rng=rng)
    ds_test_B = cycle_gan_data_source(
        args.dataset, train=False, domain="B", shuffle=False, rng=rng)
    ds_test_A = cycle_gan_data_source(
        args.dataset, train=False, domain="A", shuffle=False, rng=rng)
    di_train_B = cycle_gan_data_iterator(ds_train_B, args.batch_size)
    di_train_A = cycle_gan_data_iterator(ds_train_A, args.batch_size)
    di_test_B = cycle_gan_data_iterator(ds_test_B, args.batch_size)
    di_test_A = cycle_gan_data_iterator(ds_test_A, args.batch_size)

    # Monitors
    monitor = Monitor(args.monitor_path)

    def make_monitor(name):
        return MonitorSeries(name, monitor, interval=1)
    monitor_loss_gen = make_monitor('generator_loss')
    monitor_loss_dis_x = make_monitor('discriminator_B_domain_loss')
    monitor_loss_dis_y = make_monitor('discriminator_A_domain_loss')

    def make_monitor_image(name):
        return MonitorImage(name, monitor, interval=1,
                            normalize_method=lambda x: (x + 1.0) * 127.5)
    monitor_train_gx = make_monitor_image('fake_images_train_A')
    monitor_train_fy = make_monitor_image('fake_images_train_B')
    monitor_train_x_recon = make_monitor_image('fake_images_B_recon_train')
    monitor_train_y_recon = make_monitor_image('fake_images_A_recon_train')
    monitor_test_gx = make_monitor_image('fake_images_test_A')
    monitor_test_fy = make_monitor_image('fake_images_test_B')
    monitor_test_x_recon = make_monitor_image('fake_images_recon_test_B')
    monitor_test_y_recon = make_monitor_image('fake_images_recon_test_A')
    monitor_train_list = [
        (monitor_train_gx, y_fake),
        (monitor_train_fy, x_fake),
        (monitor_train_x_recon, x_recon),
        (monitor_train_y_recon, y_recon),
        (monitor_loss_gen, loss_gen),
        (monitor_loss_dis_x, loss_dis_x),
        (monitor_loss_dis_y, loss_dis_y),
    ]
    monitor_test_list = [
        (monitor_test_gx, y_fake_test),
        (monitor_test_fy, x_fake_test),
        (monitor_test_x_recon, x_recon_test),
        (monitor_test_y_recon, y_recon_test)]

    # ImagePool
    pool_x = ImagePool(pool_size)
    pool_y = ImagePool(pool_size)

    # Training loop
    epoch = 0
    n_images = np.max([ds_train_B.size, ds_train_A.size]
                      )  # num. images for each domain
    max_iter = args.max_epoch * n_images // args.batch_size
    for i in range(max_iter):
        # Validation
        if int((i+1) % (n_images // args.batch_size)) == 0:
            logger.info("Mode:Test,Epoch:{}".format(epoch))
            # Monitor for train
            for monitor, v in monitor_train_list:
                monitor.add(i, v.d)
            # Use training graph since there are no test mode
            x_data, _ = di_test_B.next()
            y_data, _ = di_test_A.next()
            x_real_test.d = x_data
            y_real_test.d = y_data
            x_recon_test.forward()
            y_recon_test.forward()
            # Monitor for test
            for monitor, v in monitor_test_list:
                monitor.add(i, v.d)
            # Save model
            nn.save_parameters(os.path.join(
                args.model_save_path, 'params_%06d.h5' % i))
            # Learning rate decay
            for solver in [solver_gen, solver_dis_x, solver_dis_y]:
                linear_decay(solver, base_lr, epoch, args.max_epoch)
            epoch += 1

        # Get data
        x_data, _ = di_train_B.next()
        y_data, _ = di_train_A.next()
        x_raw.d = x_data
        y_raw.d = y_data

        # Train Generators
        loss_gen.forward(clear_no_need_grad=False)
        solver_gen.zero_grad()
        loss_gen.backward(clear_buffer=True)
        solver_gen.update()

        # Insert and Get to/from pool
        x_history.d = pool_x.insert_then_get(x_fake.d)
        y_history.d = pool_y.insert_then_get(y_fake.d)

        # Train Discriminator Y
        loss_dis_y.forward(clear_no_need_grad=False)
        solver_dis_y.zero_grad()
        loss_dis_y.backward(clear_buffer=True)
        solver_dis_y.update()

        # Train Discriminator X
        loss_dis_x.forward(clear_no_need_grad=False)
        solver_dis_x.zero_grad()
        loss_dis_x.backward(clear_buffer=True)
        solver_dis_x.update()


def main():
    args = get_args()
    save_args(args)
    train(args)


if __name__ == '__main__':
    main()
