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
import datetime
import numpy as np
import nnabla as nn
import nnabla.functions as F
import nnabla.solvers as S
from nnabla.ext_utils import get_extension_context
from nnabla.monitor import Monitor
from tecogan_model import get_tecogan_model, get_frvsr_model, get_common_monitors, get_tecogan_monitors
from utils import CommunicatorWrapper, save_checkpoint, load_checkpoint
from utils.utils import ExponentialMovingAverage
from args import get_config


def main():
    conf = get_config()
    extension_module = conf.nnabla_context.context
    ctx = get_extension_context(
        extension_module, device_id=conf.nnabla_context.device_id)
    comm = CommunicatorWrapper(ctx)
    nn.set_default_context(comm.ctx)
    print("comm rank", comm.rank)

    # data iterators for train and val data
    from data_loader import data_iterator_sr, get_sample_name_grid, nn_data_gauss_down_quad

    sample_names = get_sample_name_grid(conf)
    num_samples = len(sample_names[0])
    print("No of training samples :", num_samples)

    tar_size = conf.train.crop_size
    tar_size = (conf.train.crop_size * 4) + int(1.5 * 3.0) * \
        2  # crop_size * 4, and Gaussian blur margin

    data_iterator_train = data_iterator_sr(
        conf, num_samples, sample_names, tar_size, shuffle=True)

    if comm.n_procs > 1:
        data_iterator_train = data_iterator_train.slice(
            rng=None, num_of_slices=comm.n_procs, slice_pos=comm.rank)

    train_hr = nn.Variable(
        (conf.train.batch_size, conf.train.rnn_n, conf.train.crop_size*4, conf.train.crop_size*4, 3))
    data_hr = nn.Variable(
        (conf.train.batch_size, conf.train.rnn_n, tar_size, tar_size, 3))
    train_lr = nn_data_gauss_down_quad(data_hr.reshape(
        (conf.train.batch_size * conf.train.rnn_n, tar_size, tar_size, 3)))
    train_lr = F.reshape(
        train_lr, (conf.train.batch_size, conf.train.rnn_n, conf.train.crop_size, conf.train.crop_size, 3))

    # setting up monitors for logging
    monitor_path = './nnmonitor' + \
        str(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    monitor = Monitor(monitor_path)
    common_monitor = get_common_monitors(monitor)

    # Change max_iter and learning_rate when batch size or no. of gpu devices change.
    div_factor = conf.train.batch_size * comm.n_procs
    max_iter = (conf.train.max_iter * 4) // div_factor
    learning_rate = conf.train.learning_rate * \
        (conf.train.batch_size / 4) * comm.n_procs

    if comm.rank == 0:
        print("maximum iterations", max_iter)

    scope_name = 'frvsr/'
    if conf.train.tecogan:
        scope_name = 'tecogan/'
        if not conf.train.checkpoint:
            print('loading pretrained FRVSR model',
                  conf.train.pre_trained_frvsr_weights)
            with nn.parameter_scope(scope_name):
                nn.load_parameters(conf.train.pre_trained_frvsr_weights)
                params_from_pre_trained_model = []
                for key, val in nn.get_parameters().items():
                    params_from_pre_trained_model.append(scope_name + key)

            network = get_tecogan_model(conf, train_lr, train_hr, scope_name)
            params_from_graph = nn.get_parameters()

            # Set the Generator parameters which are not in FRVSR to zero,
            # as done in orig implementation.
            for key, val in params_from_graph.items():
                if key in params_from_pre_trained_model or key.startswith('vgg') or key.startswith('disc'):
                    continue
                print(key)
                val.data.zero()  # fill with zero

        else:
            network = get_tecogan_model(conf, train_lr, train_hr, scope_name)

        # Define discriminator optimizer/solver
        solver_disc = S.Adam(alpha=learning_rate,
                             beta1=conf.train.beta, eps=conf.train.adameps)
        # Set discriminator Parameters
        with nn.parameter_scope("discriminator"):
            solver_disc.set_parameters(nn.get_parameters())

        # setting up monitors for TecoGAN
        tecogan_monitor = get_tecogan_monitors(monitor)

    else:
        network = get_frvsr_model(conf, train_lr, train_hr, scope_name)

    # Define generator and fnet optimizer/solver
    solver_gen = S.Adam(alpha=learning_rate,
                        beta1=conf.train.beta, eps=conf.train.adameps)
    solver_fnet = S.Adam(alpha=learning_rate,
                         beta1=conf.train.beta, eps=conf.train.adameps)

    # Set generator and fnet Parameters
    with nn.parameter_scope(scope_name + "generator"):
        solver_gen.set_parameters(nn.get_parameters())
    with nn.parameter_scope(scope_name + "fnet"):
        solver_fnet.set_parameters(nn.get_parameters())

    if conf.train.tecogan:
        solver_dict = {"gen": solver_gen,
                       "fnet": solver_fnet, "disc": solver_disc}
    else:
        solver_dict = {"gen": solver_gen, "fnet": solver_fnet}

    start_point = 0
    if conf.train.checkpoint:
        # Load optimizer/solver information and model weights from checkpoint
        start_point = load_checkpoint(conf.train.checkpoint, solver_dict)

    # Exponential Moving Average Calculation for tb
    ema = ExponentialMovingAverage(conf.train.decay)
    tb = 0

    # Create output directory if it doesn't exist
    if not os.path.exists(conf.data.output_dir):
        os.makedirs(conf.data.output_dir)

    # Training loop.
    for i in range(start_point, max_iter):
        # Get Training Data
        data_hr.d, train_hr.d = data_iterator_train.next()

        if conf.train.tecogan:
            network.t_discrim_loss.forward(clear_no_need_grad=True)
            if np.less(tb, 0.4):  # train gen with d
                # Compute grads for discriminator and update
                solver_disc.zero_grad()
                # Stop back-propagation from t_discrim_loss to generator
                network.t_gen_output.need_grad = False
                if comm.n_procs > 1:
                    all_reduce_callback = comm.get_all_reduce_callback()
                    network.t_discrim_loss.backward(clear_buffer=True,
                                                    communicator_callbacks=all_reduce_callback)
                else:
                    network.t_discrim_loss.backward(clear_buffer=True)
                solver_disc.update()  # Update grads
                # Enable back propagation from fnet_loss to Generator
                network.t_gen_output.need_grad = True

        # Compute grads for fnet and generator together using fnet_loss
        solver_fnet.zero_grad()
        solver_gen.zero_grad()
        # Apply forward and backward propagation on fnet_loss
        network.fnet_loss.forward(clear_no_need_grad=True)
        if comm.n_procs > 1:
            all_reduce_callback = comm.get_all_reduce_callback()
            network.fnet_loss.backward(clear_buffer=True,
                                       communicator_callbacks=all_reduce_callback)
        else:
            network.fnet_loss.backward(clear_buffer=True)
        # Update grads for fnet and generator
        solver_gen.update()
        solver_fnet.update()

        if conf.train.tecogan:
            if comm.n_procs > 1:
                comm.all_reduce([network.t_discrim_real_loss.data,
                                 network.t_adversarial_loss.data], division=True, inplace=True)
            t_balance = F.mean(network.t_discrim_real_loss.data) + \
                network.t_adversarial_loss.data
            if i == 0:
                ema.register(t_balance)
            else:
                tb = ema(t_balance)
            if comm.rank == 0:
                tecogan_monitor.monitor_pp_loss.add(
                    i, network.pp_loss.d.copy())
                tecogan_monitor.monitor_vgg_loss.add(
                    i, network.vgg_loss.d.copy())
                tecogan_monitor.monitor_sum_layer_loss.add(
                    i, network.sum_layer_loss.d.copy())
                tecogan_monitor.monitor_adv_loss.add(
                    i, network.t_adversarial_loss.d.copy())
                tecogan_monitor.monitor_disc_loss.add(
                    i, network.t_discrim_loss.d.copy())
                tecogan_monitor.monitor_tb.add(i, tb)

        if comm.rank == 0:
            common_monitor.monitor_content_loss.add(
                i, network.content_loss.d.copy())
            common_monitor.monitor_gen_loss.add(i, network.gen_loss.d.copy())
            common_monitor.monitor_warp_loss.add(i, network.warp_loss.d.copy())
            common_monitor.monitor_lr.add(i, learning_rate)
            common_monitor.monitor_time.add(i)
            if (i % conf.train.save_freq) == 0:
                # Save intermediate model parameters
                with nn.parameter_scope(scope_name):
                    nn.save_parameters(os.path.join(
                        conf.data.output_dir, "model_param_%08d.h5" % i))

                # Save intermediate check_points
                save_checkpoint(conf.data.output_dir, i, solver_dict)

    # save final Generator and Fnet network parameters
    if comm.rank == 0:
        with nn.parameter_scope(scope_name):
            nn.save_parameters(os.path.join(
                conf.data.output_dir, "model_param_%08d.h5" % i))


if __name__ == "__main__":
    main()
