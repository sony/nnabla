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
import cv2
import glob
import argparse
import random
from datetime import datetime
import numpy as np
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
from nnabla.ext_utils import get_extension_context
import nnabla.communicators as C
from models import rrdb_net
from lr_scheduler import update_learning_rate_cosine
from util import array_to_image, calculate_psnr
from nnabla.monitor import Monitor, MonitorSeries
from args import get_args


def main():
    args = get_args()

    train_gt_path = sorted(glob.glob(args.gt_train + "/*.png"))
    train_lq_path = sorted(glob.glob(args.lq_train + "/*.png"))
    val_gt_path = sorted(glob.glob(args.gt_val + "/*.png"))
    val_lq_path = sorted(glob.glob(args.lq_val + "/*.png"))
    train_samples = len(train_gt_path)
    val_samples = len(val_gt_path)

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

    # prepare monitors
    monitor_path = './nnmonitor' + str(datetime.now().strftime("%Y%m%d%H%M%S"))
    monitor = Monitor(monitor_path)
    monitor_train_iter = MonitorSeries(
        'Training loss per iteration', monitor, interval=100)
    monitor_train_epoch = MonitorSeries(
        'Training loss per eopch', monitor, interval=1)
    monitor_val_epoch = MonitorSeries(
        'Validation loss per epoch', monitor, interval=1)

    # L1 loss definition fror training
    out = rrdb_net(train_lq, 64, 23)
    loss = F.mean(F.absolute_error(out, train_gt))
    loss.persistent = True

    solver = S.Adam(alpha=0.0002, beta1=0.9, beta2=0.99)
    # Set Parameters
    params = nn.get_parameters()
    solver.set_parameters(params)
    train_size = int(train_samples / args.batch_size_train/n_devices)
    total_epochs = args.n_epochs
    start_epoch = 0
    current_iter = 0
    for epoch in range(start_epoch+1, total_epochs + 1):
        # training loop
        index = 0
        while index < train_size:
            if device_id == 0:
                current_iter += 1
            train_gt.d, train_lq.d = data_iterator_train.next()
            lr = update_learning_rate_cosine(
                current_iter, eta_max, eta_min, n_devices)
            loss.forward(clear_no_need_grad=True)
            solver.zero_grad()
            # All-reduce gradients every 2MiB parameters during backward computation
            if args.distributed:
                params = [x.grad for x in nn.get_parameters().values()]
                loss.backward(clear_buffer=True, communicator_callbacks=comm.all_reduce_callback(
                    params, 1024 * 1024 * 2))
            else:
                loss.backward(clear_buffer=True)
            solver.set_learning_rate(lr)
            solver.update()
            index += 1
            if device_id == 0:
                monitor_train_iter.add(current_iter, loss.d.copy())
        if device_id == 0:
            monitor_train_epoch.add(epoch, loss.d.copy())

            # validation loop
        if device_id == 0:
            avg_psnr = 0.0
            for idx in range(val_samples):
                val_gt_im, val_lq_im = data_iterator_val.next()
                val_gt = nn.Variable.from_numpy_array(val_gt_im)
                val_lq = nn.Variable.from_numpy_array(val_lq_im)
                sr_img = rrdb_net(val_lq, 64, 23)
                sr_img.persistent = True
                sr_img.forward(clear_buffer=True)
                real_image = array_to_image(val_gt.d)
                sr_image = array_to_image(sr_img.d)
                img_name = os.path.splitext(
                    os.path.basename(val_lq_path[idx]))[0]
                img_dir = os.path.join(
                    args.save_results+"/resultspsnr", img_name)
                if not os.path.exists(img_dir):
                    os.makedirs(img_dir)
                save_img_path = os.path.join(
                    img_dir, '{:s}_{:d}.png'.format(img_name, epoch))
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
            monitor_val_epoch.add(epoch, avg_psnr)

        if device_id == 0:
            if not os.path.exists(args.savemodel):
                os.makedirs(args.savemodel)
            out_param_file = os.path.join(args.savemodel, str(epoch) + '.h5')
            nn.save_parameters(out_param_file)


if __name__ == "__main__":
    main()
