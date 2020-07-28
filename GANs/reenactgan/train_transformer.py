# Copyright (c) 2020 Sony Corporation. All Rights Reserved.
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
import argparse
import numpy as np
import yaml
import pprint

import nnabla as nn
import nnabla.logger as logger
import nnabla.monitor as nm
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S

import data
import models
from config import load_transformer_config
from utils import MonitorManager, combine_images


def lsgan_loss(pred, target):
    return F.mean(F.squared_error(pred, target))


def recon_loss(x, y):
    return F.mean(F.absolute_error(x, y))


def train_transformer(config, netG, netD, solver_netG, solver_netD, train_iterators, monitor):

    netG_A2B, netG_B2A = netG['netG_A2B'], netG['netG_B2A']
    netD_A, netD_B = netD['netD_A'], netD['netD_B']
    solver_netG_AB, solver_netG_BA = solver_netG['netG_A2B'], solver_netG['netG_B2A']
    solver_netD_A, solver_netD_B = solver_netD['netD_A'], solver_netD['netD_B']

    train_iterator_src, train_iterator_trg = train_iterators

    if config["train"]["cycle_loss"] and config["train"]["cycle_loss"]["lambda"] > 0:
        print(f'Applying Cycle Loss, weight: {config["train"]["cycle_loss"]["lambda"]}.')
        with_cycle_loss = True
    else:
        with_cycle_loss = False

    if config["train"]["shape_loss"] and config["train"]["shape_loss"]["lambda"] > 0:
        print(f'Applying Shape Loss using PCA, weight: {config["train"]["shape_loss"]["lambda"]}.')
        with_shape_loss = True
    else:
        with_shape_loss = False

    # Load boundary image to get Variable shapes
    bod_map_A = train_iterator_src.next()[0]
    bod_map_B = train_iterator_trg.next()[0]
    real_bod_map_A = nn.Variable(bod_map_A.shape)
    real_bod_map_B = nn.Variable(bod_map_B.shape)
    real_bod_map_A.persistent, real_bod_map_B.persistent = True, True

    ################### Graph Construction ####################
    # Generator
    with nn.parameter_scope('netG_transformer'):
        with nn.parameter_scope('netG_A2B'):
            fake_bod_map_B = netG_A2B(
                real_bod_map_A, test=False, norm_type=config["norm_type"])  # (1, 15, 64, 64)
        with nn.parameter_scope('netG_B2A'):
            fake_bod_map_A = netG_B2A(
                real_bod_map_B, test=False, norm_type=config["norm_type"])  # (1, 15, 64, 64)
    fake_bod_map_B.persistent, fake_bod_map_A.persistent = True, True

    fake_bod_map_B_unlinked = fake_bod_map_B.get_unlinked_variable()
    fake_bod_map_A_unlinked = fake_bod_map_A.get_unlinked_variable()

    # Reconstruct images if cycle loss is applied.
    if with_cycle_loss:
        with nn.parameter_scope('netG_transformer'):
            with nn.parameter_scope('netG_B2A'):
                recon_bod_map_A = netG_B2A(
                    fake_bod_map_B_unlinked, test=False, norm_type=config["norm_type"])  # (1, 15, 64, 64)
            with nn.parameter_scope('netG_A2B'):
                recon_bod_map_B = netG_A2B(
                    fake_bod_map_A_unlinked, test=False, norm_type=config["norm_type"])  # (1, 15, 64, 64)
        recon_bod_map_A.persistent, recon_bod_map_B.persistent = True, True

    # Discriminator
    with nn.parameter_scope('netD_transformer'):
        with nn.parameter_scope('netD_A'):
            pred_fake_A = netD_A(fake_bod_map_A_unlinked, test=False)
            pred_real_A = netD_A(real_bod_map_A, test=False)
        with nn.parameter_scope('netD_B'):
            pred_fake_B = netD_B(fake_bod_map_B_unlinked, test=False)
            pred_real_B = netD_B(real_bod_map_B, test=False)
    real_target = F.constant(1, pred_fake_A.shape)
    fake_target = F.constant(0, pred_real_A.shape)

    ################### Loss Definition ####################
    # Generator loss
    # LSGAN loss
    loss_gan_A = lsgan_loss(pred_fake_A, real_target)
    loss_gan_B = lsgan_loss(pred_fake_B, real_target)
    loss_gan_A.persistent, loss_gan_B.persistent = True, True
    loss_gan = loss_gan_A + loss_gan_B

    # Cycle loss
    if with_cycle_loss:
        loss_cycle_A = recon_loss(recon_bod_map_A, real_bod_map_A)
        loss_cycle_B = recon_loss(recon_bod_map_B, real_bod_map_B)
        loss_cycle_A.persistent, loss_cycle_B.persistent = True, True
        loss_cycle = loss_cycle_A + loss_cycle_B

    # Shape loss
    if with_shape_loss:
        with nn.parameter_scope("Align"):
            nn.load_parameters(
                config["train"]["shape_loss"]["align_param_path"])
            shape_bod_map_real_A = models.align_resnet(
                real_bod_map_A, fix_parameters=True)
            shape_bod_map_fake_B = models.align_resnet(
                fake_bod_map_B_unlinked, fix_parameters=True)

            shape_bod_map_real_B = models.align_resnet(
                real_bod_map_B, fix_parameters=True)
            shape_bod_map_fake_A = models.align_resnet(
                fake_bod_map_A_unlinked, fix_parameters=True)

        with nn.parameter_scope("PCA"):
            nn.load_parameters(config["train"]["shape_loss"]["PCA_param_path"])
            shape_bod_map_real_A = PF.affine(
                shape_bod_map_real_A, 212, fix_parameters=True)
            shape_bod_map_real_A = shape_bod_map_real_A[:, :3]

            shape_bod_map_fake_B = PF.affine(
                shape_bod_map_fake_B, 212, fix_parameters=True)
            shape_bod_map_fake_B = shape_bod_map_fake_B[:, :3]

            shape_bod_map_real_B = PF.affine(
                shape_bod_map_real_B, 212, fix_parameters=True)
            shape_bod_map_real_B = shape_bod_map_real_B[:, :3]

            shape_bod_map_fake_A = PF.affine(
                shape_bod_map_fake_A, 212, fix_parameters=True)
            shape_bod_map_fake_A = shape_bod_map_fake_A[:, :3]

        shape_bod_map_real_A.persistent, shape_bod_map_fake_A.persistent = True, True
        shape_bod_map_real_B.persistent, shape_bod_map_fake_B.persistent = True, True

        loss_shape_A = recon_loss(shape_bod_map_real_A, shape_bod_map_fake_B)
        loss_shape_B = recon_loss(shape_bod_map_real_B, shape_bod_map_fake_A)
        loss_shape_A.persistent, loss_shape_B.persistent = True, True
        loss_shape = loss_shape_A + loss_shape_B

    # Total Generator Loss
    loss_netG = loss_gan

    if with_cycle_loss:
        loss_netG += loss_cycle * config["train"]["cycle_loss"]["lambda"]

    if with_shape_loss:
        loss_netG += loss_shape * config["train"]["shape_loss"]["lambda"]

    # Discriminator loss
    loss_netD_A = lsgan_loss(pred_real_A, real_target) + \
        lsgan_loss(pred_fake_A, fake_target)
    loss_netD_B = lsgan_loss(pred_real_B, real_target) + \
        lsgan_loss(pred_fake_B, fake_target)
    loss_netD_A.persistent, loss_netD_B.persistent = True, True

    loss_netD = loss_netD_A + loss_netD_B

    ################### Setting Solvers ####################
    # Generator solver
    with nn.parameter_scope('netG_transformer'):
        with nn.parameter_scope('netG_A2B'):
            solver_netG_AB.set_parameters(nn.get_parameters())
        with nn.parameter_scope('netG_B2A'):
            solver_netG_BA.set_parameters(nn.get_parameters())

    # Discrimintar solver
    with nn.parameter_scope('netD_transformer'):
        with nn.parameter_scope('netD_A'):
            solver_netD_A.set_parameters(nn.get_parameters())
        with nn.parameter_scope('netD_B'):
            solver_netD_B.set_parameters(nn.get_parameters())

    ################### Create Monitors ####################
    interval = config["monitor"]["interval"]
    monitors_G_dict = {'loss_netG': loss_netG,
                       'loss_gan_A': loss_gan_A, 'loss_gan_B': loss_gan_B}

    if with_cycle_loss:
        monitors_G_dict.update(
            {'loss_cycle_A': loss_cycle_A, 'loss_cycle_B': loss_cycle_B})

    if with_shape_loss:
        monitors_G_dict.update(
            {'loss_shape_A': loss_shape_A, 'loss_shape_B': loss_shape_B})

    monitors_G = MonitorManager(monitors_G_dict, monitor, interval=interval)

    monitors_D_dict = {'loss_netD': loss_netD,
                       'loss_netD_A': loss_netD_A, 'loss_netD_B': loss_netD_B}
    monitors_D = MonitorManager(monitors_D_dict, monitor, interval=interval)

    monitor_time = nm.MonitorTimeElapsed(
        'time_training', monitor, interval=interval)
    monitor_vis = nm.MonitorImage(
        'result', monitor, interval=1, num_images=4, normalize_method=lambda x: x)

    # Dump training information
    with open(os.path.join(monitor._save_path, "training_info.yaml"), "w", encoding="utf-8") as f:
        f.write(yaml.dump(config))

    # Training
    epoch = config["train"]["epochs"]
    i = 0
    iter_per_epoch = train_iterator_src.size // config["train"]["batch_size"] + 1
    for e in range(epoch):
        logger.info(f'Epoch = {e} / {epoch}')
        train_iterator_src._reset()  # rewind the iterator
        train_iterator_trg._reset()  # rewind the iterator
        for _ in range(iter_per_epoch):
            bod_map_A = train_iterator_src.next()[0]
            bod_map_B = train_iterator_trg.next()[0]
            real_bod_map_A.d, real_bod_map_B.d = bod_map_A, bod_map_B

            # Generate fake image
            fake_bod_map_B.forward(clear_no_need_grad=True)
            fake_bod_map_A.forward(clear_no_need_grad=True)

            # Update Discriminator
            solver_netD_A.zero_grad()
            solver_netD_B.zero_grad()
            loss_netD.forward(clear_no_need_grad=True)
            loss_netD.backward(clear_buffer=True)
            if config["train"]["weight_decay"]:
                solver_netD_A.weight_decay(config["train"]["weight_decay"])
                solver_netD_B.weight_decay(config["train"]["weight_decay"])
            solver_netD_A.update()
            solver_netD_B.update()

            # Update Generator
            solver_netG_BA.zero_grad()
            solver_netG_AB.zero_grad()
            solver_netD_A.zero_grad()
            solver_netD_B.zero_grad()
            fake_bod_map_B_unlinked.grad.zero()
            fake_bod_map_A_unlinked.grad.zero()
            loss_netG.forward(clear_no_need_grad=True)
            loss_netG.backward(clear_buffer=True)
            fake_bod_map_B.backward(grad=None)
            fake_bod_map_A.backward(grad=None)
            solver_netG_AB.update()
            solver_netG_BA.update()

            # Monitors
            monitor_time.add(i)
            monitors_G.add(i)
            monitors_D.add(i)

            i += 1

        images_to_visualize = [real_bod_map_A.d,
                               fake_bod_map_B.d, real_bod_map_B.d]
        if with_cycle_loss:
            images_to_visualize.extend(
                [recon_bod_map_A.d, fake_bod_map_A.d, recon_bod_map_B.d])
        else:
            images_to_visualize.extend([fake_bod_map_A.d])
        visuals = combine_images(images_to_visualize)
        monitor_vis.add(i, visuals)

        if e % config["monitor"]["save_interval"] == 0 or e == epoch - 1:
            # Save parameters of networks
            netG_B2A_save_path = os.path.join(monitor._save_path,
                                              f'netG_transformer_B2A_{e}.h5')
            netG_A2B_save_path = os.path.join(monitor._save_path,
                                              f'netG_transformer_A2B_{e}.h5')
            with nn.parameter_scope('netG_transformer'):
                with nn.parameter_scope('netG_A2B'):
                    nn.save_parameters(netG_A2B_save_path)
                with nn.parameter_scope('netG_B2A'):
                    nn.save_parameters(netG_B2A_save_path)

            netD_A_save_path = os.path.join(monitor._save_path,
                                            f'netD_transformer_A_{e}.h5')
            netD_B_save_path = os.path.join(monitor._save_path,
                                            f'netD_transformer_B_{e}.h5')
            with nn.parameter_scope('netD_transformer'):
                with nn.parameter_scope('netD_A'):
                    nn.save_parameters(netD_A_save_path)
                with nn.parameter_scope('netD_B'):
                    nn.save_parameters(netD_B_save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None, type=str)
    parser.add_argument('--info', default=None, type=str)
    args = parser.parse_args()

    config = load_transformer_config(args.config)
    if args.info:
        config["experiment_name"] += args.info

    pprint.pprint(config)

    #########################
    # Context Setting
    # Get context.
    from nnabla.ext_utils import get_extension_context
    logger.info(f'Running in {config["context"]}.')
    ctx = get_extension_context(
        config["context"], device_id=config["device_id"])
    nn.set_default_context(ctx)
    #########################

    # Data Loading
    logger.info('Initialing Datasource')
    train_iterator_src = data.celebv_data_iterator(dataset_mode="transformer",
                                                   celeb_name=config["src_celeb_name"],
                                                   data_dir=config["train_dir"],
                                                   ref_dir=config["ref_dir"],
                                                   mode=config["mode"],
                                                   batch_size=config["train"]["batch_size"],
                                                   shuffle=config["train"]["shuffle"],
                                                   with_memory_cache=config["train"]["with_memory_cache"],
                                                   with_file_cache=config["train"]["with_file_cache"],
                                                   resize_size=config["preprocess"]["resize_size"],
                                                   line_thickness=config["preprocess"]["line_thickness"],
                                                   gaussian_kernel=config["preprocess"]["gaussian_kernel"],
                                                   gaussian_sigma=config["preprocess"]["gaussian_sigma"]
                                                   )

    train_iterator_trg = data.celebv_data_iterator(dataset_mode="transformer",
                                                   celeb_name=config["trg_celeb_name"],
                                                   data_dir=config["train_dir"],
                                                   ref_dir=config["ref_dir"],
                                                   mode=config["mode"],
                                                   batch_size=config["train"]["batch_size"],
                                                   shuffle=config["train"]["shuffle"],
                                                   with_memory_cache=config["train"]["with_memory_cache"],
                                                   with_file_cache=config["train"]["with_file_cache"],
                                                   resize_size=config["preprocess"]["resize_size"],
                                                   line_thickness=config["preprocess"]["line_thickness"],
                                                   gaussian_kernel=config["preprocess"]["gaussian_kernel"],
                                                   gaussian_sigma=config["preprocess"]["gaussian_sigma"]
                                                   )
    train_iterators = (train_iterator_src, train_iterator_trg)
    # monitor
    monitor = nm.Monitor(os.path.join(
                    config["logdir"], "transformer",
                    f'{config["src_celeb_name"]}2{config["trg_celeb_name"]}', config["experiment_name"]))

    # Network
    netG = {'netG_A2B': models.netG_transformer,
            'netG_B2A': models.netG_transformer}
    netD = {'netD_A': models.netD_transformer,
            'netD_B': models.netD_transformer}

    # Optimizer
    solver_netG = {'netG_A2B': S.Adam(alpha=config["train"]["lr"], beta1=config["train"]["beta1"], beta2=config["train"]["beta2"]),
                   'netG_B2A': S.Adam(alpha=config["train"]["lr"], beta1=config["train"]["beta1"], beta2=config["train"]["beta2"])}

    solver_netD = {'netD_A': S.Adam(alpha=0.5 * config["train"]["lr"], beta1=config["train"]["beta1"], beta2=config["train"]["beta2"]),
                   'netD_B': S.Adam(alpha=0.5 * config["train"]["lr"], beta1=config["train"]["beta1"], beta2=config["train"]["beta2"])}

    train_transformer(config, netG, netD, solver_netG,
                      solver_netD, train_iterators, monitor)


if __name__ == '__main__':
    main()
