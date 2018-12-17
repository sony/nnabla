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
import nnabla.communicators as C

from nnabla.monitor import Monitor, MonitorImage, MonitorSeries, MonitorTimeElapsed
from nnabla.ext_utils import get_extension_context
from functools import reduce
from collections import OrderedDict

from args import get_args, save_args
from models import style_encoder, content_encoder, decoder, discriminators, recon_loss, lsgan_loss
from datasets import munit_data_iterator


def train(args):
    # Create Communicator and Context
    extension_module = "cudnn"
    ctx = get_extension_context(extension_module, type_config=args.type_config)
    comm = C.MultiProcessDataParalellCommunicator(ctx)
    comm.init()
    n_devices = comm.size
    mpi_rank = comm.rank
    mpi_local_rank = comm.local_rank
    device_id = mpi_local_rank
    ctx.device_id = str(device_id)
    nn.set_default_context(ctx)

    # Input
    b, c, h, w = args.batch_size, 3, args.image_size, args.image_size
    x_real_a = nn.Variable([b, c, h, w])
    x_real_b = nn.Variable([b, c, h, w])

    # Model
    # workaround for starting with the same model among devices.
    np.random.seed(412)
    maps = args.maps
    # within-domain reconstruction (domain A)
    x_content_a = content_encoder(x_real_a, maps, name="content-encoder-a")
    x_style_a = style_encoder(x_real_a, maps, name="style-encoder-a")
    x_recon_a = decoder(x_content_a, x_style_a, name="decoder-a")
    # within-domain reconstruction (domain B)
    x_content_b = content_encoder(x_real_b, maps, name="content-encoder-b")
    x_style_b = style_encoder(x_real_b, maps, name="style-encoder-b")
    x_recon_b = decoder(x_content_b, x_style_b, name="decoder-b")
    # generate over domains and reconstruction of content and style (domain A)
    z_style_a = F.randn(shape=x_style_a.shape)
    x_fake_a = decoder(x_content_b, z_style_a, name="decoder-a")
    x_content_rec_b = content_encoder(x_fake_a, maps, name="content-encoder-a")
    x_style_rec_a = style_encoder(x_fake_a, maps, name="style-encoder-a")
    # generate over domains and reconstruction of content and style (domain B)
    z_style_b = F.randn(shape=x_style_b.shape)
    x_fake_b = decoder(x_content_a, z_style_b, name="decoder-b")
    x_content_rec_a = content_encoder(x_fake_b, maps, name="content-encoder-b")
    x_style_rec_b = style_encoder(x_fake_b, maps, name="style-encoder-b")
    # discriminate (domain A)
    p_x_fake_a_list = discriminators(x_fake_a)
    p_x_real_a_list = discriminators(x_real_a)
    p_x_fake_b_list = discriminators(x_fake_b)
    p_x_real_b_list = discriminators(x_real_b)

    # Loss
    # within-domain reconstruction
    loss_recon_x_a = recon_loss(x_recon_a, x_real_a).apply(persistent=True)
    loss_recon_x_b = recon_loss(x_recon_b, x_real_b).apply(persistent=True)
    # content and style reconstruction
    loss_recon_x_style_a = recon_loss(
        x_style_rec_a, z_style_a).apply(persistent=True)
    loss_recon_x_content_b = recon_loss(
        x_content_rec_b, x_content_b).apply(persistent=True)
    loss_recon_x_style_b = recon_loss(
        x_style_rec_b, z_style_b).apply(persistent=True)
    loss_recon_x_content_a = recon_loss(
        x_content_rec_a, x_content_a).apply(persistent=True)
    # adversarial

    def f(x, y): return x + y
    loss_gen_a = reduce(f, [lsgan_loss(p_f)
                            for p_f in p_x_fake_a_list]).apply(persistent=True)
    loss_dis_a = reduce(f, [lsgan_loss(p_f, p_r) for p_f, p_r in
                            zip(p_x_fake_a_list, p_x_real_a_list)]).apply(persistent=True)
    loss_gen_b = reduce(f, [lsgan_loss(p_f)
                            for p_f in p_x_fake_b_list]).apply(persistent=True)
    loss_dis_b = reduce(f, [lsgan_loss(p_f, p_r) for p_f, p_r in
                            zip(p_x_fake_b_list, p_x_real_b_list)]).apply(persistent=True)
    # loss for generator-related models
    loss_gen = loss_gen_a + loss_gen_b \
        + args.lambda_x * (loss_recon_x_a + loss_recon_x_b) \
        + args.lambda_c * (loss_recon_x_content_a + loss_recon_x_content_b) \
        + args.lambda_s * (loss_recon_x_style_a + loss_recon_x_style_b)
    # loss for discriminators
    loss_dis = loss_dis_a + loss_dis_b

    # Solver
    lr_g, lr_d, beta1, beta2 = args.lr_g, args.lr_d, args.beta1, args.beta2
    # solver for generator-related models
    solver_gen = S.Adam(lr_g, beta1, beta2)
    with nn.parameter_scope("generator"):
        params_gen = nn.get_parameters()
    solver_gen.set_parameters(params_gen)
    # solver for discriminators
    solver_dis = S.Adam(lr_d, beta1, beta2)
    with nn.parameter_scope("discriminators"):
        params_dis = nn.get_parameters()
    solver_dis.set_parameters(params_dis)

    # Monitor
    monitor = Monitor(args.monitor_path)
    # time
    monitor_time = MonitorTimeElapsed("Training time", monitor, interval=10)
    # reconstruction
    monitor_loss_recon_x_a = MonitorSeries(
        "Recon Loss Image A", monitor, interval=10)
    monitor_loss_recon_x_content_b = MonitorSeries(
        "Recon Loss Content B", monitor, interval=10)
    monitor_loss_recon_x_style_a = MonitorSeries(
        "Recon Loss Style A", monitor, interval=10)
    monitor_loss_recon_x_b = MonitorSeries(
        "Recon Loss Image B", monitor, interval=10)
    monitor_loss_recon_x_content_a = MonitorSeries(
        "Recon Loss Content A", monitor, interval=10)
    monitor_loss_recon_x_style_b = MonitorSeries(
        "Recon Loss Style B", monitor, interval=10)
    # adversarial
    monitor_loss_gen_a = MonitorSeries("Gen Loss A", monitor, interval=10)
    monitor_loss_dis_a = MonitorSeries("Dis Loss A", monitor, interval=10)
    monitor_loss_gen_b = MonitorSeries("Gen Loss B", monitor, interval=10)
    monitor_loss_dis_b = MonitorSeries("Dis Loss B", monitor, interval=10)
    monitor_losses = [
        # reconstruction
        (monitor_loss_recon_x_a, loss_recon_x_a),
        (monitor_loss_recon_x_content_b, loss_recon_x_content_b),
        (monitor_loss_recon_x_style_a, loss_recon_x_style_a),
        (monitor_loss_recon_x_b, loss_recon_x_b),
        (monitor_loss_recon_x_content_a, loss_recon_x_content_a),
        (monitor_loss_recon_x_style_b, loss_recon_x_style_b),
        # adaversarial
        (monitor_loss_gen_a, loss_gen_a),
        (monitor_loss_dis_a, loss_dis_a),
        (monitor_loss_gen_b, loss_gen_b),
        (monitor_loss_dis_b, loss_dis_b)
    ]
    # image
    monitor_image_a = MonitorImage(
        "Fake Image B to A Train", monitor, interval=1)
    monitor_image_b = MonitorImage(
        "Fake Image A to B Train", monitor, interval=1)
    monitor_images = [
        (monitor_image_a, x_fake_a),
        (monitor_image_b, x_fake_b),
    ]

    # DataIterator
    rng_a = np.random.RandomState(device_id)
    rng_b = np.random.RandomState(device_id + n_devices)
    di_a = munit_data_iterator(args.img_path_a, args.batch_size, rng=rng_a)
    di_b = munit_data_iterator(args.img_path_b, args.batch_size, rng=rng_b)

    # Train
    for i in range(args.max_iter // n_devices):
        ii = i * n_devices
        # Train generator-related models
        x_data_a, x_data_b = di_a.next()[0], di_b.next()[0]
        x_real_a.d, x_real_b.d = x_data_a, x_data_b
        solver_gen.zero_grad()
        loss_gen.forward(clear_no_need_grad=True)
        loss_gen.backward(clear_buffer=True)
        comm.all_reduce([w.grad for w in params_gen.values()])
        solver_gen.weight_decay(args.weight_decay_rate)
        solver_gen.update()

        # Train discriminators
        x_data_a, x_data_b = di_a.next()[0], di_b.next()[0]
        x_real_a.d, x_real_b.d = x_data_a, x_data_b
        x_fake_a.need_grad, x_fake_b.need_grad = False, False
        solver_dis.zero_grad()
        loss_dis.forward(clear_no_need_grad=True)
        loss_dis.backward(clear_buffer=True)
        comm.all_reduce([w.grad for w in params_dis.values()])
        solver_dis.weight_decay(args.weight_decay_rate)
        solver_dis.update()
        x_fake_a.need_grad, x_fake_b.need_grad = True, True

        # LR schedule
        if (i + 1) % (args.lr_decay_at_every // n_devices) == 0:
            lr_d = solver_dis.learning_rate() * args.lr_decay_rate
            lr_g = solver_gen.learning_rate() * args.lr_decay_rate
            solver_dis.set_learning_rate(lr_d)
            solver_gen.set_learning_rate(lr_g)

        if mpi_local_rank == 0:
            # Monitor
            monitor_time.add(ii)
            for mon, loss in monitor_losses:
                mon.add(ii, loss.d)
            # Save
            if (i + 1) % (args.model_save_interval // n_devices) == 0:
                for mon, x in monitor_images:
                    mon.add(ii, x.d)
                nn.save_parameters(os.path.join(
                    args.monitor_path, "param_{:05d}.h5".format(i)))

    if mpi_local_rank == 0:
        # Monitor
        for mon, loss in monitor_losses:
            mon.add(ii, loss.d)
        # Save
        for mon, x in monitor_images:
            mon.add(ii, x.d)
        nn.save_parameters(os.path.join(
            args.monitor_path, "param_{:05d}.h5".format(i)))


def main():
    args = get_args()
    save_args(args)
    train(args)


if __name__ == '__main__':
    main()
