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
import yaml
import pprint
import argparse
import numpy as np

import nnabla as nn
import nnabla.logger as logger
import nnabla.monitor as nm
import nnabla.functions as F
import nnabla.solvers as S

import data
import models
from config import load_decoder_config
from utils import MonitorManager, combine_images


def gan_loss(x, target):
    return F.mean(F.binary_cross_entropy(x, target))


def recon_loss(x, y):
    return F.mean(F.absolute_error(x, y))


def vgg16_perceptual_loss(fake, real):
    '''
        VGG perceptual loss based on VGG-16 network.
        Assuming the values in fake and real are in [0, 255].
    '''
    from nnabla.models.imagenet import VGG16

    class VisitFeatures(object):
        def __init__(self):
            self.features = []
            self.relu_counter = 0
            # self.features_at = set([1, 4, 7, 10]) : ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']
            self.features_at = set([4, 7])

        def __call__(self, f):
            if not f.name.startswith('ReLU'):
                return
            if self.relu_counter in self.features_at:
                self.features.append(f.outputs[0])
            self.relu_counter += 1

    vgg = VGG16()

    def get_features(x):
        o = vgg(x, use_up_to='lastconv')
        f = VisitFeatures()
        o.visit(f)
        return f

    with nn.parameter_scope("vgg16_loss"):
        fake_features = get_features(fake)
        real_features = get_features(real)

    return sum([F.mean(F.squared_error(ff, fr)) for ff, fr in zip(fake_features.features, real_features.features)])


def train(config, netG, netD, solver_netG, solver_netD, train_iterator, monitor):

    if config["train"]["feature_loss"] and config["train"]["feature_loss"]["lambda"] > 0:
        print(f'Applying VGG feature Loss, weight: {config["train"]["feature_loss"]["lambda"]}.')
        with_feature_loss = True
    else:
        with_feature_loss = False

    # Load image and boundary image to get Variable shapes
    img, bod_map, bod_map_resize = train_iterator.next()

    real_img = nn.Variable(img.shape)
    real_bod_map = nn.Variable(bod_map.shape)
    real_bod_map_resize = nn.Variable(bod_map_resize.shape)

    ################### Graph Construction ####################
    # Generator
    with nn.parameter_scope('netG_decoder'):
        fake_img = netG(real_bod_map, test=False)
    fake_img.persistent = True

    fake_img_unlinked = fake_img.get_unlinked_variable()

    # Discriminator
    with nn.parameter_scope('netD_decoder'):
        pred_fake = netD(F.concatenate(real_bod_map_resize,
                                       fake_img_unlinked, axis=1), test=False)
        pred_real = netD(F.concatenate(
            real_bod_map_resize, real_img, axis=1), test=False)
    real_target = F.constant(1, pred_fake.shape)
    fake_target = F.constant(0, pred_real.shape)

    ################### Loss Definition ####################
    # for Generator
    gan_loss_G = gan_loss(pred_fake, real_target)
    gan_loss_G.persistent = True

    weight_L1 = config["train"]["weight_L1"]
    L1_loss = recon_loss(fake_img_unlinked, real_img)
    L1_loss.persistent = True
    loss_netG = gan_loss_G + weight_L1 * L1_loss

    if with_feature_loss:
        feature_loss = vgg16_perceptual_loss(
            127.5 * (fake_img_unlinked + 1.), 127.5 * (real_img + 1.))
        feature_loss.persistent = True
        loss_netG += feature_loss * config["train"]["feature_loss"]["lambda"]

    # for Discriminator
    loss_netD = (gan_loss(pred_real, real_target) +
                 gan_loss(pred_fake, fake_target)) * 0.5

    ################### Setting Solvers ####################
    # for Generator
    with nn.parameter_scope('netG_decoder'):
        solver_netG.set_parameters(nn.get_parameters())

    # for Discrimintar
    with nn.parameter_scope('netD_decoder'):
        solver_netD.set_parameters(nn.get_parameters())

    ################### Create Monitors ####################
    interval = config["monitor"]["interval"]
    monitors_G_dict = {'loss_netG': loss_netG,
                       'loss_gan': gan_loss_G,
                       'L1_loss': L1_loss}

    if with_feature_loss:
        monitors_G_dict.update({'vgg_feature_loss': feature_loss})

    monitors_G = MonitorManager(monitors_G_dict, monitor, interval=interval)

    monitors_D_dict = {'loss_netD': loss_netD}
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
    lr_decay_start_at = config["train"]["lr_decay_start_at"]
    iter_per_epoch = train_iterator.size // config["train"]["batch_size"] + 1
    for e in range(epoch):
        logger.info(f'Epoch = {e} / {epoch}')
        train_iterator._reset()  # rewind the iterator
        if e > lr_decay_start_at:
            decay_coeff = 1.0 - max(0, e - lr_decay_start_at) / 50.
            lr_decayed = config["train"]["lr"] * decay_coeff
            print(f"learning rate decayed to {lr_decayed}")
            solver_netG.set_learning_rate(lr_decayed)
            solver_netD.set_learning_rate(lr_decayed)

        for _ in range(iter_per_epoch):
            img, bod_map, bod_map_resize = train_iterator.next()
            # bod_map_noize = np.random.random_sample(bod_map.shape) * 0.01
            # bod_map_resize_noize = np.random.random_sample(bod_map_resize.shape) * 0.01

            real_img.d = img
            real_bod_map.d = bod_map  # + bod_map_noize
            real_bod_map_resize.d = bod_map_resize  # + bod_map_resize_noize

            # Generate fake image
            fake_img.forward(clear_no_need_grad=True)

            # Update Discriminator
            solver_netD.zero_grad()
            solver_netG.zero_grad()
            loss_netD.forward(clear_no_need_grad=True)
            loss_netD.backward(clear_buffer=True)
            solver_netD.update()

            # Update Generator
            solver_netD.zero_grad()
            solver_netG.zero_grad()
            fake_img_unlinked.grad.zero()
            loss_netG.forward(clear_no_need_grad=True)
            loss_netG.backward(clear_buffer=True)
            fake_img.backward(grad=None)
            solver_netG.update()

            # Monitors
            monitor_time.add(i)
            monitors_G.add(i)
            monitors_D.add(i)

            i += 1

        images_to_visualize = [real_bod_map_resize.d, fake_img.d, img]
        visuals = combine_images(images_to_visualize)
        monitor_vis.add(i, visuals)

        if e % config["monitor"]["save_interval"] == 0 or e == epoch - 1:
            # Save parameters of networks
            netG_save_path = os.path.join(monitor._save_path, f'netG_decoder_{e}.h5')
            with nn.parameter_scope('netG_decoder'):
                nn.save_parameters(netG_save_path)
            netD_save_path = os.path.join(monitor._save_path, f'netD_decoder_{e}.h5')
            with nn.parameter_scope('netD_decoder'):
                nn.save_parameters(netD_save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None, type=str)
    parser.add_argument('--info', default=None, type=str)
    args = parser.parse_args()

    config = load_decoder_config(args.config)
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
    train_iterator = data.celebv_data_iterator(dataset_mode="decoder",
                                               celeb_name=config["trg_celeb_name"],
                                               data_dir=config["train_dir"],
                                               ref_dir=config["ref_dir"],
                                               mode=config["mode"],
                                               batch_size=config["train"]["batch_size"],
                                               shuffle=config["train"]["shuffle"],
                                               with_memory_cache=config["train"]["with_memory_cache"],
                                               with_file_cache=config["train"]["with_file_cache"],
                                               )

    monitor = nm.Monitor(os.path.join(
                    config["logdir"], "decoder", config["trg_celeb_name"], config["experiment_name"]))
    # Optimizer
    solver_netG = S.Adam(alpha=config["train"]
                         ["lr"], beta1=config["train"]["beta1"])
    solver_netD = S.Adam(alpha=config["train"]
                         ["lr"], beta1=config["train"]["beta1"])

    # Network
    netG = models.netG_decoder
    netD = models.netD_decoder

    train(config, netG, netD, solver_netG,
          solver_netD, train_iterator, monitor)


if __name__ == '__main__':
    main()
