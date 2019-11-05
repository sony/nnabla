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
from nnabla.monitor import Monitor, MonitorSeries, MonitorTimeElapsed
from nnabla.ext_utils import get_extension_context
from nnabla.utils.image_utils import imsave

import models
import loss
import functools
from args import get_args
from dataloader import insta_gan_data_source, insta_gan_data_iterator


def save(path, img):
    img = (img * 0.5) + 0.5
    imsave(path, img, channel_first=True)


def save_images(args, i, source, result, domain="x", reconstructed=None):
    source_img = source.d[0, :3]
    result_img = result.d[0, :3]
    save("{}/{}_at_{}_source.png".format(args.monitor_path, domain, i), source_img)
    save("{}/{}_at_{}_result.png".format(args.monitor_path, domain, i), result_img)
    if reconstructed:
        recon_img = reconstructed.d[0, :3]
        save("{}/{}_at_{}_recon.png".format(args.monitor_path, domain, i), recon_img)


def image_augmentation(args, img, seg):
    imgseg = F.concatenate(img, seg, axis=1)
    imgseg = F.random_crop(imgseg, shape=(args.fineSizeH, args.fineSizeW))
    if not args.no_flip:
        imgseg = F.random_flip(imgseg, axes=(3,))
    return imgseg


class Monitors(object):
    """
        Helper class for using monitors.
    """

    def __init__(self, args, monitoring_targets):
        super(Monitors, self).__init__()
        self.targets = monitoring_targets
        self.monitor = Monitor(args.monitor_path)
        self.loss_monitors = dict()
        self.log_step = args.log_step
        self.create_loss_monitors()

    def create_loss_monitors(self):
        for name in self.targets.keys():
            self.loss_monitors[name] = MonitorSeries(
                name, self.monitor, self.log_step)

    def add(self, i):
        for name, var in self.targets.items():
            self.loss_monitors[name].add(i, var.d.item())


def train(args):

    # Variable size.
    bs, ch, h, w = args.batch_size, 3, args.loadSizeH, args.loadSizeW

    # Determine normalization method.
    if args.norm == "instance":
        norm_layer = functools.partial(
            PF.instance_normalization, fix_parameters=True, no_bias=True, no_scale=True)
    else:
        norm_layer = PF.batch_normalization

    # Prepare Generator and Discriminator based on user config.
    generator = functools.partial(models.generator, input_nc=args.input_nc, output_nc=args.output_nc,
                                  ngf=args.ngf, norm_layer=norm_layer, use_dropout=False, n_blocks=9, padding_type='reflect')
    discriminator = functools.partial(models.discriminator, input_nc=args.output_nc,
                                      ndf=args.ndf, n_layers=args.n_layers_D, norm_layer=norm_layer, use_sigmoid=False)

    # --------------------- Computation Graphs --------------------

    # Input images and masks of both source / target domain
    x = nn.Variable([bs, ch, h, w], need_grad=False)
    a = nn.Variable([bs, 1, h, w], need_grad=False)

    y = nn.Variable([bs, ch, h, w], need_grad=False)
    b = nn.Variable([bs, 1, h, w], need_grad=False)

    # Apply image augmentation and get an unlinked variable
    xa_aug = image_augmentation(args, x, a)
    xa_aug.persistent = True
    xa_aug_unlinked = xa_aug.get_unlinked_variable()

    yb_aug = image_augmentation(args, y, b)
    yb_aug.persistent = True
    yb_aug_unlinked = yb_aug.get_unlinked_variable()

    # variables used for Image Pool
    x_history = nn.Variable([bs, ch, h, w])
    a_history = nn.Variable([bs, 1, h, w])
    y_history = nn.Variable([bs, ch, h, w])
    b_history = nn.Variable([bs, 1, h, w])

    # Generate Images (x -> y')
    with nn.parameter_scope("gen_x2y"):
        yb_fake = generator(xa_aug_unlinked)
    yb_fake.persistent = True
    yb_fake_unlinked = yb_fake.get_unlinked_variable()

    # Generate Images (y -> x')
    with nn.parameter_scope("gen_y2x"):
        xa_fake = generator(yb_aug_unlinked)
    xa_fake.persistent = True
    xa_fake_unlinked = xa_fake.get_unlinked_variable()

    # Reconstruct Images (y' -> x)
    with nn.parameter_scope("gen_y2x"):
        xa_recon = generator(yb_fake_unlinked)
    xa_recon.persistent = True

    # Reconstruct Images (x' -> y)
    with nn.parameter_scope("gen_x2y"):
        yb_recon = generator(xa_fake_unlinked)
    yb_recon.persistent = True

    # Use Discriminator on y' and x'
    with nn.parameter_scope("dis_y"):
        d_y_fake = discriminator(yb_fake_unlinked)
    d_y_fake.persistent = True

    with nn.parameter_scope("dis_x"):
        d_x_fake = discriminator(xa_fake_unlinked)
    d_x_fake.persistent = True

    # Use Discriminator on y and x
    with nn.parameter_scope("dis_y"):
        d_y_real = discriminator(yb_aug_unlinked)

    with nn.parameter_scope("dis_x"):
        d_x_real = discriminator(xa_aug_unlinked)

    # Identity Mapping (x -> x)
    with nn.parameter_scope("gen_y2x"):
        xa_idt = generator(xa_aug_unlinked)

    # Identity Mapping (y -> y)
    with nn.parameter_scope("gen_x2y"):
        yb_idt = generator(yb_aug_unlinked)

    # -------------------- Loss --------------------

    # (LS)GAN Loss (for Discriminator)
    loss_dis_x = (loss.lsgan_loss(d_y_fake, False) +
                  loss.lsgan_loss(d_y_real, True)) * 0.5
    loss_dis_y = (loss.lsgan_loss(d_x_fake, False) +
                  loss.lsgan_loss(d_x_real, True)) * 0.5
    loss_dis = loss_dis_x + loss_dis_y

    # Cycle Consistency Loss
    loss_cyc_x = args.lambda_cyc * loss.recon_loss(xa_recon, xa_aug_unlinked)
    loss_cyc_y = args.lambda_cyc * loss.recon_loss(yb_recon, yb_aug_unlinked)
    loss_cyc = loss_cyc_x + loss_cyc_y

    # Identity Mapping Loss
    loss_idt_x = args.lambda_idt * loss.recon_loss(xa_idt, xa_aug_unlinked)
    loss_idt_y = args.lambda_idt * loss.recon_loss(yb_idt, yb_aug_unlinked)
    loss_idt = loss_idt_x + loss_idt_y

    # Context Preserving Loss
    loss_ctx_x = args.lambda_ctx * \
        loss.context_preserving_loss(xa_aug_unlinked, yb_fake_unlinked)
    loss_ctx_y = args.lambda_ctx * \
        loss.context_preserving_loss(yb_aug_unlinked, xa_fake_unlinked)
    loss_ctx = loss_ctx_x + loss_ctx_y

    # (LS)GAN Loss (for Generator)
    d_loss_gen_x = loss.lsgan_loss(d_x_fake, True)
    d_loss_gen_y = loss.lsgan_loss(d_y_fake, True)
    d_loss_gen = d_loss_gen_x + d_loss_gen_y

    # Total Loss for Generator
    loss_gen = loss_cyc + loss_idt + loss_ctx + d_loss_gen

    # --------------------- Solvers --------------------

    # Initial learning rates
    G_lr = args.learning_rate_G
    #D_lr = args.learning_rate_D
    # As opposed to the description in the paper, D_lr is set the same as G_lr.
    D_lr = args.learning_rate_G

    # Define solvers
    solver_gen_x2y = S.Adam(G_lr, args.beta1, args.beta2)
    solver_gen_y2x = S.Adam(G_lr, args.beta1, args.beta2)
    solver_dis_x = S.Adam(D_lr, args.beta1, args.beta2)
    solver_dis_y = S.Adam(D_lr, args.beta1, args.beta2)

    # Set Parameters to each solver
    with nn.parameter_scope("gen_x2y"):
        solver_gen_x2y.set_parameters(nn.get_parameters())

    with nn.parameter_scope("gen_y2x"):
        solver_gen_y2x.set_parameters(nn.get_parameters())

    with nn.parameter_scope("dis_x"):
        solver_dis_x.set_parameters(nn.get_parameters())

    with nn.parameter_scope("dis_y"):
        solver_dis_y.set_parameters(nn.get_parameters())

    # create convenient functions manipulating Solvers
    def solvers_zero_grad():
        # Zeroing Gradients of all solvers
        solver_gen_x2y.zero_grad()
        solver_gen_y2x.zero_grad()
        solver_dis_x.zero_grad()
        solver_dis_y.zero_grad()

    def solvers_update_parameters(new_D_lr, new_G_lr):
        # Learning rate updater
        solver_gen_x2y.set_learning_rate(new_G_lr)
        solver_gen_y2x.set_learning_rate(new_G_lr)
        solver_dis_x.set_learning_rate(new_D_lr)
        solver_dis_y.set_learning_rate(new_D_lr)

    # -------------------- Data Iterators --------------------

    ds_train_A = insta_gan_data_source(
        args, train=True, domain="A", shuffle=True)
    di_train_A = insta_gan_data_iterator(ds_train_A, args.batch_size)

    ds_train_B = insta_gan_data_source(
        args, train=True, domain="B", shuffle=True)
    di_train_B = insta_gan_data_iterator(ds_train_B, args.batch_size)

    # -------------------- Monitors --------------------

    monitoring_targets_dis = {'discriminator_loss_x': loss_dis_x,
                              'discriminator_loss_y': loss_dis_y}
    monitors_dis = Monitors(args, monitoring_targets_dis)

    monitoring_targets_gen = {'generator_loss_x': d_loss_gen_x,
                              'generator_loss_y': d_loss_gen_y,
                              'reconstruction_loss_x': loss_cyc_x,
                              'reconstruction_loss_y': loss_cyc_y,
                              'identity_mapping_loss_x': loss_idt_x,
                              'identity_mapping_loss_y': loss_idt_y,
                              'content_preserving_loss_x': loss_ctx_x,
                              'content_preserving_loss_y': loss_ctx_y}
    monitors_gen = Monitors(args, monitoring_targets_gen)

    monitor_time = MonitorTimeElapsed(
        "Training_time", Monitor(args.monitor_path), args.log_step)

    # Training loop
    epoch = 0
    n_images = max([ds_train_B.size, ds_train_A.size])
    print("{} images exist.".format(n_images))
    max_iter = args.max_epoch * n_images // args.batch_size
    decay_iter = args.max_epoch - args.lr_decay_start_epoch

    for i in range(max_iter):
        if i % (n_images // args.batch_size) == 0 and i > 0:
            # Learning Rate Decay
            epoch += 1
            print("epoch {}".format(epoch))
            if epoch >= args.lr_decay_start_epoch:
                new_D_lr = D_lr * \
                    (1.0 - max(0, epoch - args.lr_decay_start_epoch - 1) /
                     float(decay_iter - 1))
                new_G_lr = G_lr * \
                    (1.0 - max(0, epoch - args.lr_decay_start_epoch - 1) /
                     float(decay_iter - 1))
                solvers_update_parameters(new_D_lr, new_G_lr)
                print("Current learning rate for Discriminator: {}".format(
                    solver_dis_x.learning_rate()))
                print("Current learning rate for Generator: {}".format(
                    solver_gen_x2y.learning_rate()))

        # Get data
        x_data, a_data = di_train_A.next()
        y_data, b_data = di_train_B.next()
        x.d, a.d = x_data, a_data
        y.d, b.d = y_data, b_data

        solvers_zero_grad()

        # Image Augmentation
        nn.forward_all([xa_aug, yb_aug], clear_buffer=True)

        # Generate fake images
        nn.forward_all([xa_fake, yb_fake], clear_no_need_grad=True)

        # -------- Train Discriminator --------

        loss_dis.forward(clear_no_need_grad=True)
        monitors_dis.add(i)

        loss_dis.backward(clear_buffer=True)
        solver_dis_x.update()
        solver_dis_y.update()

        # -------- Train Generators --------

        # since the gradients computed above remain, reset to zero.
        xa_fake_unlinked.grad.zero()
        yb_fake_unlinked.grad.zero()
        solvers_zero_grad()

        loss_gen.forward(clear_no_need_grad=True)

        monitors_gen.add(i)
        monitor_time.add(i)

        loss_gen.backward(clear_buffer=True)
        xa_fake.backward(grad=None, clear_buffer=True)
        yb_fake.backward(grad=None, clear_buffer=True)
        solver_gen_x2y.update()
        solver_gen_y2x.update()

        if i % (n_images // args.batch_size) == 0:
            # save translation results after every epoch.
            save_images(args, i, xa_aug, yb_fake,
                        domain="x", reconstructed=xa_recon)
            save_images(args, i, yb_aug, xa_fake,
                        domain="y", reconstructed=yb_recon)

    # save pretrained parameters
    nn.save_parameters(os.path.join(
        args.model_save_path, 'params_%06d.h5' % i))


def main():
    args = get_args()
    # Context
    extension_module = args.context
    ctx = get_extension_context(
        extension_module, device_id=args.device_id, type_config=args.type_config)
    nn.set_default_context(ctx)

    train(args)


if __name__ == '__main__':
    main()
