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
import sys
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
from nnabla.ext_utils import get_extension_context
import nnabla.communicators as C
from nnabla.monitor import Monitor, MonitorSeries, MonitorTimeElapsed
import glob
import cv2
from utils import *
from models import rrdb_net
from utils.lr_scheduler import get_repeated_cosine_annealing_learning_rate, get_multistep_learning_rate
from datetime import datetime
from utils.util import array_to_image, calculate_psnr
from args import get_config

# Calculate PSNR and save validation image


def val_save(val_gt, val_lq, val_lq_path, idx, epoch, avg_psnr):
    conf = get_config()
    sr_img = rrdb_net(val_lq, 64, 23)
    real_image = array_to_image(val_gt.data)
    sr_image = array_to_image(sr_img.data)
    img_name = os.path.splitext(
        os.path.basename(val_lq_path[idx]))[0]
    img_dir = os.path.join(
        conf.val.save_results + "/results", img_name)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    save_img_path = os.path.join(
        img_dir, '{:s}_{:d}.png'.format(img_name, epoch))
    cv2.imwrite(save_img_path, sr_image)
    crop_size = conf.train.scale
    cropped_sr_image = sr_image[crop_size:-
                                crop_size, crop_size:-crop_size, :]
    cropped_real_image = real_image[crop_size:-
                                    crop_size, crop_size:-crop_size, :]
    avg_psnr += calculate_psnr(cropped_sr_image, cropped_real_image)
    print("validating", img_name)
    return avg_psnr


def main():
    conf = get_config()
    train_gt_path = sorted(glob.glob(conf.DIV2K.gt_train + "/*.png"))
    train_lq_path = sorted(glob.glob(conf.DIV2K.lq_train + "/*.png"))
    val_gt_path = sorted(glob.glob(conf.SET14.gt_val + "/*.png"))
    val_lq_path = sorted(glob.glob(conf.SET14.lq_val + "/*.png"))
    train_samples = len(train_gt_path)
    val_samples = len(val_gt_path)
    lr_g = conf.hyperparameters.lr_g
    lr_d = conf.hyperparameters.lr_d
    lr_steps = conf.train.lr_steps

    random.seed(conf.train.seed)
    np.random.seed(conf.train.seed)

    extension_module = conf.nnabla_context.context
    ctx = get_extension_context(
        extension_module, device_id=conf.nnabla_context.device_id)
    comm = CommunicatorWrapper(ctx)
    nn.set_default_context(comm.ctx)

    # data iterators for train and val data
    from data_loader import data_iterator_sr
    data_iterator_train = data_iterator_sr(
        train_samples, conf.train.batch_size, train_gt_path, train_lq_path, train=True, shuffle=True)
    data_iterator_val = data_iterator_sr(
        val_samples, conf.val.batch_size, val_gt_path, val_lq_path, train=False, shuffle=False)

    if comm.n_procs > 1:
        data_iterator_train = data_iterator_train.slice(
            rng=None, num_of_slices=comm.n_procs, slice_pos=comm.rank)

    train_gt = nn.Variable(
        (conf.train.batch_size, 3, conf.train.gt_size, conf.train.gt_size))
    train_lq = nn.Variable(
        (conf.train.batch_size, 3, conf.train.gt_size // conf.train.scale, conf.train.gt_size // conf.train.scale))

    # setting up monitors for logging
    monitor_path = './nnmonitor' + str(datetime.now().strftime("%Y%m%d%H%M%S"))
    monitor = Monitor(monitor_path)
    monitor_pixel_g = MonitorSeries(
        'l_g_pix per iteration', monitor, interval=100)
    monitor_val = MonitorSeries(
        'Validation loss per epoch', monitor, interval=1)
    monitor_time = MonitorTimeElapsed(
        "Training time per epoch", monitor, interval=1)

    with nn.parameter_scope("gen"):
        nn.load_parameters(conf.train.gen_pretrained)
        fake_h = rrdb_net(train_lq, 64, 23)
        fake_h.persistent = True
    pixel_loss = F.mean(F.absolute_error(fake_h, train_gt))
    pixel_loss.persistent = True
    gen_loss = pixel_loss

    if conf.model.esrgan:
        from esrgan_model import get_esrgan_gen, get_esrgan_dis, get_esrgan_monitors
        gen_model = get_esrgan_gen(conf, train_gt, train_lq, fake_h)
        gen_loss = conf.hyperparameters.eta_pixel_loss * pixel_loss + conf.hyperparameters.feature_loss_weight * gen_model.feature_loss + \
            conf.hyperparameters.lambda_gan_loss * gen_model.loss_gan_gen
        dis_model = get_esrgan_dis(fake_h, gen_model.pred_d_real)
        # Set Discriminator parameters
        solver_dis = S.Adam(lr_d, beta1=0.9, beta2=0.99)
        with nn.parameter_scope("dis"):
            solver_dis.set_parameters(nn.get_parameters())
        esr_mon = get_esrgan_monitors()

    # Set generator Parameters
    solver_gen = S.Adam(alpha=lr_g, beta1=0.9, beta2=0.99)
    with nn.parameter_scope("gen"):
        solver_gen.set_parameters(nn.get_parameters())

    train_size = int(
        train_samples / conf.train.batch_size / comm.n_procs)
    total_epochs = conf.train.n_epochs
    start_epoch = 0
    current_iter = 0
    if comm.rank == 0:
        print("total_epochs", total_epochs)
        print("train_samples", train_samples)
        print("val_samples", val_samples)
        print("train_size", train_size)

    for epoch in range(start_epoch + 1, total_epochs + 1):
        index = 0
        # Training loop for psnr rrdb model
        while index < train_size:
            current_iter += comm.n_procs
            train_gt.d, train_lq.d = data_iterator_train.next()

            if not conf.model.esrgan:
                lr_g = get_repeated_cosine_annealing_learning_rate(
                    current_iter, conf.hyperparameters.eta_max, conf.hyperparameters.eta_min, conf.train.cosine_period,
                    conf.train.cosine_num_period)

            if conf.model.esrgan:
                lr_g = get_multistep_learning_rate(
                    current_iter, lr_steps, lr_g)
                gen_model.var_ref.d = train_gt.d
                gen_model.pred_d_real.grad.zero()
                gen_model.pred_d_real.forward(clear_no_need_grad=True)
                gen_model.pred_d_real.need_grad = False

            # Generator update
            gen_loss.forward(clear_no_need_grad=True)
            solver_gen.zero_grad()
            # All-reduce gradients every 2MiB parameters during backward computation
            if comm.n_procs > 1:
                with nn.parameter_scope('gen'):
                    all_reduce_callback = comm.get_all_reduce_callback()
                    gen_loss.backward(clear_buffer=True,
                                      communicator_callbacks=all_reduce_callback)
            else:
                gen_loss.backward(clear_buffer=True)
            solver_gen.set_learning_rate(lr_g)
            solver_gen.update()

            # Discriminator Upate
            if conf.model.esrgan:
                gen_model.pred_d_real.need_grad = True
                lr_d = get_multistep_learning_rate(
                    current_iter, lr_steps, lr_d)
                solver_dis.zero_grad()
                dis_model.l_d_total.forward(clear_no_need_grad=True)
                if comm.n_procs > 1:
                    with nn.parameter_scope('dis'):
                        all_reduce_callback = comm.get_all_reduce_callback()
                    dis_model.l_d_total.backward(
                        clear_buffer=True, communicator_callbacks=all_reduce_callback)
                else:
                    dis_model.l_d_total.backward(clear_buffer=True)
                solver_dis.set_learning_rate(lr_d)
                solver_dis.update()

            index += 1
            if comm.rank == 0:
                monitor_pixel_g.add(
                    current_iter, pixel_loss.d.copy())
                monitor_time.add(epoch * comm.n_procs)
            if comm.rank == 0 and conf.model.esrgan:
                esr_mon.monitor_feature_g.add(
                    current_iter, gen_model.feature_loss.d.copy())
                esr_mon.monitor_gan_g.add(
                    current_iter, gen_model.loss_gan_gen.d.copy())
                esr_mon.monitor_gan_d.add(
                    current_iter, dis_model.l_d_total.d.copy())
                esr_mon.monitor_d_real.add(current_iter, F.mean(
                    gen_model.pred_d_real.data).data)
                esr_mon.monitor_d_fake.add(current_iter, F.mean(
                    gen_model.pred_g_fake.data).data)

        # Validation Loop
        if comm.rank == 0:
            avg_psnr = 0.0
            for idx in range(val_samples):
                val_gt_im, val_lq_im = data_iterator_val.next()
                val_gt = nn.NdArray.from_numpy_array(val_gt_im)
                val_lq = nn.NdArray.from_numpy_array(val_lq_im)
                with nn.parameter_scope("gen"):
                    avg_psnr = val_save(
                        val_gt, val_lq, val_lq_path, idx, epoch, avg_psnr)
            avg_psnr = avg_psnr / val_samples
            monitor_val.add(epoch, avg_psnr)

        # Save generator weights
        if comm.rank == 0:
            if not os.path.exists(conf.train.savemodel):
                os.makedirs(conf.train.savemodel)
            with nn.parameter_scope("gen"):
                nn.save_parameters(os.path.join(
                    conf.train.savemodel, "generator_param_%06d.h5" % epoch))
       # Save discriminator weights
        if comm.rank == 0 and conf.model.esrgan:
            with nn.parameter_scope("dis"):
                nn.save_parameters(os.path.join(
                    conf.train.savemodel, "discriminator_param_%06d.h5" % epoch))


if __name__ == "__main__":
    main()
