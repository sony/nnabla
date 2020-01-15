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

import argparse
import random
import numpy as np
import os
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
from nnabla.ext_utils import get_extension_context
import nnabla.communicators as C
from nnabla.monitor import Monitor, MonitorSeries
import glob
import cv2
from models import rrdb_net
from vgg19 import load_vgg19
from discriminator_arch import discriminator
from loss import *
from lr_scheduler import update_learning_rate_multistep
from datetime import datetime
from util import array_to_image, calculate_psnr
from args import get_args


def main():
    args = get_args()

    train_gt_path = sorted(glob.glob(args.gt_train + "/*.png"))
    train_lq_path = sorted(glob.glob(args.lq_train + "/*.png"))
    val_gt_path = sorted(glob.glob(args.gt_val + "/*.png"))
    val_lq_path = sorted(glob.glob(args.lq_val + "/*.png"))
    train_samples = len(train_gt_path)
    val_samples = len(val_gt_path)
    lr_steps = [50000, 100000, 200000, 300000]
    lr_g = 1e-4
    lr_d = 1e-4

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.distributed:
        # Communicator and Context
        extension_module = "cudnn"
        ctx = get_extension_context(extension_module, type_config='float')
        comm = C.MultiProcessDataParalellCommunicator(ctx)
        comm.init()
        n_devices = comm.size
        mpi_rank = comm.rank
        mpi_local_rank = comm.local_rank
        device_id = mpi_local_rank
        ctx.device_id = str(device_id)
        nn.set_default_context(ctx)
    else:
        # Get context.
        extension_module = args.context
        if args.context is None:
            extension_module = 'cpu'
        ctx = get_extension_context(extension_module, device_id=args.device_id)
        nn.set_default_context(ctx)
        n_devices = 1
        device_id = args.device_id

    lr_steps = [x//n_devices for x in lr_steps]

    # data iterators for train and val data
    from data_loader import data_iterator_sr
    data_iterator_train = data_iterator_sr(
        train_samples, args.batch_size_train, train_gt_path, train_lq_path, train=True, shuffle=True)
    data_iterator_val = data_iterator_sr(
        val_samples, args.batch_size_val, val_gt_path, val_lq_path, train=False, shuffle=False)

    if args.distributed:
        data_iterator_train = data_iterator_train.slice(
            rng=None, num_of_slices=n_devices, slice_pos=device_id)

    train_gt = nn.Variable(
        (args.batch_size_train, 3, args.gt_size, args.gt_size))
    train_lq = nn.Variable(
        (args.batch_size_train, 3, args.gt_size//args.scale, args.gt_size//args.scale))
    var_ref = nn.Variable(
        (args.batch_size_train, 3, args.gt_size, args.gt_size))

    # setting up monitors for logging
    monitor_path = './nnmonitor' + str(datetime.now().strftime("%Y%m%d%H%M%S"))
    monitor = Monitor(monitor_path)
    monitor_pixel_g = MonitorSeries(
        'l_g_pix per iteration', monitor, interval=100)
    monitor_feature_g = MonitorSeries(
        'l_g_fea per iteration', monitor, interval=100)
    monitor_gan_g = MonitorSeries(
        'l_g_gan per iteration', monitor, interval=100)
    monitor_gan_d = MonitorSeries(
        'l_d_total per iteration', monitor, interval=100)
    monitor_d_real = MonitorSeries(
        'D_real per iteration', monitor, interval=100)
    monitor_d_fake = MonitorSeries(
        'D_fake per iteration', monitor, interval=100)
    monitor_val = MonitorSeries(
        'Validation loss per epoch', monitor, interval=1)

    # pretrained weights used for generator which were obtained\
    # using PSNR based generator training.
    with nn.parameter_scope('gen'):
        nn.load_parameters(args.psnr_rrdb_pretrained)

    # Generator Loss
    # Pixel Loss (L1 loss)
    with nn.parameter_scope("gen"):
        fake_h = rrdb_net(train_lq, 64, 23)
        fake_h.persistent = True
    pixel_loss = F.mean(F.absolute_error(fake_h, train_gt))

    # Feature Loss (L1 Loss)
    real_fea = load_vgg19(train_gt)
    # need_grad set to False, to avoid BP to vgg19 network
    real_fea.need_grad = False
    fake_fea = load_vgg19(fake_h)
    feature_loss = F.mean(F.absolute_error(fake_fea, real_fea))
    feature_loss.persistent = True

    # Gan Loss Generator
    with nn.parameter_scope("dis"):
        pred_g_fake = discriminator(fake_h)
        pred_d_real = discriminator(var_ref)
    pred_d_real.persistent = True
    pred_g_fake.persistent = True
    unlinked_pred_d_real = pred_d_real.get_unlinked_variable()
    gan_loss = RelativisticAverageGanLoss(GanLoss())
    gan_loss_gen_out = None
    gan_loss_gen_out += gan_loss(unlinked_pred_d_real, pred_g_fake)
    loss_gan_gen = gan_loss_gen_out.generator_loss

    loss_gan_gen.persistent = True
    pixel_loss.persistent = True

    # Total Generator Loss
    total_g_loss = args.eta_pixel_loss * pixel_loss + args.feature_loss_weight * feature_loss + \
        args.lambda_gan_loss * loss_gan_gen

    # Gan Loss Discriminator
    with nn.parameter_scope("dis"):
        unlinked_fake_h = fake_h.get_unlinked_variable()
        pred_d_fake = discriminator(unlinked_fake_h)
    gan_loss = RelativisticAverageGanLoss(GanLoss())
    gan_loss_dis_out = None
    gan_loss_dis_out += gan_loss(pred_d_real, pred_d_fake)
    l_d_total = gan_loss_dis_out.discriminator_loss
    l_d_total.persistent = True

    # Create Solvers for generators and discriminators.
    solver_gen = S.Adam(lr_g, beta1=0.9, beta2=0.99)
    solver_dis = S.Adam(lr_d, beta1=0.9, beta2=0.99)
    with nn.parameter_scope("gen"):
        solver_gen.set_parameters(nn.get_parameters())
    with nn.parameter_scope("dis"):
        solver_dis.set_parameters(nn.get_parameters())

    train_size = int(train_samples / args.batch_size_train / n_devices)
    total_epochs = args.n_epochs
    if device_id == 0:
        print("total_epochs", total_epochs)
        print("train_samples", train_samples)
        print("val_samples", val_samples)
        print("train_size", train_size)
    start_epoch = 0
    current_iter = 0
    for epoch in range(start_epoch+1, total_epochs + 1):
        # training loop
        index = 0
        if device_id == 0:
            if not os.path.exists(args.savemodel):
                os.makedirs(args.savemodel)
            with nn.parameter_scope("gen"):
                nn.save_parameters(os.path.join(
                    args.savemodel, "generator_param_%06d.h5" % epoch))
            with nn.parameter_scope("dis"):
                nn.save_parameters(os.path.join(
                        args.savemodel, "discriminator_param_%06d.h5" % epoch))
        while index < train_size:
            if device_id == 0:
                current_iter += 1
            train_gt.d, train_lq.d = data_iterator_train.next()
            var_ref.d = train_gt.d

            pred_d_real.grad.zero()
            pred_d_real.forward(clear_no_need_grad=True)
            pred_d_real.need_grad = False
            # Generator update
            lr_g = update_learning_rate_multistep(current_iter, lr_steps, lr_g)
            solver_gen.zero_grad()
            total_g_loss.forward(clear_no_need_grad=True)
            # All-reduce gradients every 2MiB parameters during backward computation
            if args.distributed:
                params = [x.grad for x in nn.get_parameters().values()]
                total_g_loss.backward(clear_buffer=True, communicator_callbacks=comm.all_reduce_callback(
                    params, 1024 * 1024 * 2))
            else:
                total_g_loss.backward(clear_buffer=True)
            solver_gen.set_learning_rate(lr_g)
            solver_gen.update()
            if device_id == 0:
                monitor_pixel_g.add(current_iter, pixel_loss.d.copy())
                monitor_feature_g.add(current_iter, feature_loss.d.copy())
                monitor_gan_g.add(current_iter, loss_gan_gen.d.copy())

            # Discriminator Upate
            pred_d_real.need_grad = True
            lr_d = update_learning_rate_multistep(current_iter, lr_steps, lr_d)
            solver_dis.zero_grad()
            l_d_total.forward(clear_no_need_grad=True)
            if args.distributed:
                params = [x.grad for x in nn.get_parameters().values()]
                l_d_total.backward(clear_buffer=True, communicator_callbacks=comm.all_reduce_callback(
                    params, 1024 * 1024 * 2))
            else:
                l_d_total.backward(clear_buffer=True)
            solver_dis.set_learning_rate(lr_d)
            solver_dis.update()

            index += 1

            if device_id == 0:
                monitor_gan_d.add(current_iter, l_d_total.d.copy())
                monitor_d_real.add(current_iter, F.mean(pred_d_real.data).data)
                monitor_d_fake.add(current_iter, F.mean(pred_g_fake.data).data)

        # Validation Loop
        if device_id == 0:
            avg_psnr = 0.0
            for idx in range(val_samples):
                val_gt_im, val_lq_im = data_iterator_val.next()
                val_gt = nn.Variable.from_numpy_array(val_gt_im)
                val_lq = nn.Variable.from_numpy_array(val_lq_im)
                with nn.parameter_scope("gen"):
                    sr_img = rrdb_net(val_lq, 64, 23)
                    sr_img.persistent = True
                    sr_img.forward(clear_buffer=True)
                real_image = array_to_image(val_gt.d)
                sr_image = array_to_image(sr_img.d)
                img_name = os.path.splitext(
                    os.path.basename(val_lq_path[idx]))[0]
                img_dir = os.path.join(
                    args.save_results+"/resultsesrgan", img_name)
                if not os.path.exists(img_dir):
                    os.makedirs(img_dir)
                save_img_path = os.path.join(
                    img_dir, '{:s}_{:d}.png'.format(img_name, current_iter))
                cv2.imwrite(save_img_path, sr_image)
                crop_size = args.scale
                real_image = real_image / 255.
                sr_image = sr_image / 255.
                cropped_sr_image = sr_image[crop_size:-
                                            crop_size, crop_size:-crop_size, :]
                cropped_real_image = real_image[crop_size:-
                                                crop_size, crop_size:-crop_size, :]
                avg_psnr += calculate_psnr(cropped_sr_image *
                                           255, cropped_real_image * 255)
                print("validating", img_name)
            avg_psnr = avg_psnr / idx
            monitor_val.add(epoch, avg_psnr)

    if device_id == 0:
        with nn.parameter_scope("gen"):
            nn.save_parameters(os.path.join(
                args.savemodel, "generator_param_%06d.h5" % epoch))
        with nn.parameter_scope("dis"):
            nn.save_parameters(os.path.join(
                args.savemodel, "discriminator_param_%06d.h5" % epoch))


if __name__ == "__main__":
    main()
