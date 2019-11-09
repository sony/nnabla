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
import argparse
import numpy as np
from tqdm import trange

import nnabla as nn
from nnabla.logger import logger
import nnabla.solvers as S
import nnabla.functions as F

from models import SpadeGenerator, encode_inputs
from utils import *


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", "-C", default="./config.yaml")
    args, subargs = parser.parse_known_args()

    conf = read_yaml(args.cfg)

    # args for nnabla config
    parser.add_argument("--ext_name", "-e", default=conf.ext_name)
    parser.add_argument("--device_id", "-d", default=conf.device_id, type=int)
    parser.add_argument("--type_config", "-t", default=conf.type_config)

    parser.add_argument("--batch_size", "-b",
                        default=conf.batch_size, type=int)

    parser.add_argument("--save_path", "-S", default=conf.save_path)
    parser.add_argument("--load_params", "-L", default=None)

    parser.add_argument("--dataset", "-D", default=conf.dataset)

    args = parser.parse_args()

    conf.update(args.__dict__)

    # refine config
    conf.g_lr = float(conf.g_lr)
    conf.d_lr = float(conf.d_lr)
    conf.image_shape = tuple(conf[conf.dataset]["image_shape"])

    dir_name = "{}_{}_{}x{}".format(
        get_current_time(), conf.dataset, *conf.image_shape)
    conf.save_path = os.path.join(conf.save_path, dir_name)

    return conf


def train():
    rng = np.random.RandomState(803)

    conf = get_config()

    comm = init_nnabla(conf)

    # create data iterator
    if conf.dataset == "cityscapes":
        data_list = get_cityscape_datalist(
            conf.cityscapes, save_file=comm.rank == 0)
        n_class = conf.cityscapes.n_label_ids
        use_inst = True

        data_iter = create_cityscapes_iterator(conf.batch_size, data_list, comm=comm,
                                               image_shape=conf.image_shape, rng=rng,
                                               flip=conf.use_flip)

    elif conf.dataset == "ade20k":
        data_list = get_ade20k_datalist(
            conf.ade20k, save_file=comm.rank == 0)
        n_class = conf.ade20k.n_label_ids + 1  # class id + unknown
        use_inst = False

        load_shape = tuple(
            x + 30 for x in conf.image_shape) if conf.use_crop else conf.image_shape
        data_iter = create_ade20k_iterator(conf.batch_size, data_list, comm=comm,
                                           load_shape=load_shape, crop_shape=conf.image_shape,
                                           rng=rng, flip=conf.use_flip)

    else:
        raise NotImplementedError(
            "Currently dataset {} is not supported.".format(conf.dataset))

    real = nn.Variable(shape=(conf.batch_size, 3) + conf.image_shape)
    obj_mask = nn.Variable(shape=(conf.batch_size, ) + conf.image_shape)

    if use_inst:
        ist_mask = nn.Variable(shape=(conf.batch_size,) + conf.image_shape)
        obj_onehot, bm = encode_inputs(ist_mask, obj_mask, n_ids=n_class)
        mask = F.concatenate(obj_onehot, bm, axis=1)
    else:
        om = obj_mask
        if len(om.shape) == 3:
            om = F.reshape(om, om.shape + (1,))
        obj_onehot = F.one_hot(om, shape=(n_class,))
        mask = F.transpose(obj_onehot, (0, 3, 1, 2))

    # generator
    generator = SpadeGenerator(conf.g_ndf, image_shape=conf.image_shape)
    z = F.randn(shape=(conf.batch_size, conf.z_dim))
    fake = generator(z, mask)

    # unlinking
    ul_mask, ul_fake = get_unlinked_all(mask, fake)

    # discriminator
    discriminator = PatchGAN(n_scales=conf.d_n_scales)
    d_input_real = F.concatenate(real, ul_mask, axis=1)
    d_input_fake = F.concatenate(ul_fake, ul_mask, axis=1)
    d_real_out, d_real_feats = discriminator(d_input_real)
    d_fake_out, d_fake_feats = discriminator(d_input_fake)

    g_gan, g_feat, d_real, d_fake = discriminator.get_loss(d_real_out, d_real_feats,
                                                           d_fake_out, d_fake_feats,
                                                           use_fm=conf.use_fm, fm_lambda=conf.lambda_fm,
                                                           gan_loss_type=conf.gan_loss_type)

    def _rescale(x): return rescale_values(x, input_min=-1,
                                           input_max=1, output_min=0, output_max=255)

    g_vgg = vgg16_perceptual_loss(_rescale(ul_fake),
                                  _rescale(real)) * conf.lambda_vgg

    set_persistent_all(fake, mask, g_gan, g_feat, d_real, d_fake, g_vgg)

    # loss
    g_loss = g_gan + g_feat + g_vgg
    d_loss = (d_real + d_fake) / 2

    # load params
    if conf.load_params is not None:
        print("load parameters from {}".format(conf.load_params))
        nn.load_parameters(conf.load_params)

    # Setup Solvers
    g_solver = S.Adam(beta1=0.)
    g_solver.set_parameters(get_params_startswith("spade_generator"))

    d_solver = S.Adam(beta1=0.)
    d_solver.set_parameters(get_params_startswith("discriminator"))

    # lr scheduler
    g_lrs = LinearDecayScheduler(
        start_lr=conf.g_lr, end_lr=0., start_iter=100, end_iter=200)
    d_lrs = LinearDecayScheduler(
        start_lr=conf.d_lr, end_lr=0., start_iter=100, end_iter=200)

    ipe = get_iteration_per_epoch(
        data_iter._size, conf.batch_size, round="ceil")

    if not conf.show_interval:
        conf.show_interval = ipe
    if not conf.save_interval:
        conf.save_interval = ipe
    if not conf.niter:
        conf.niter = 200 * ipe

    # Setup Reporter
    losses = {"g_gan": g_gan, "g_feat": g_feat,
              "g_vgg": g_vgg, "d_real": d_real, "d_fake": d_fake}
    reporter = Reporter(comm, losses, conf.save_path,
                        nimage_per_epoch=min(conf.batch_size, 5), show_interval=10)
    progress_iterator = trange(conf.niter, disable=comm.rank > 0)
    reporter.start(progress_iterator)

    colorizer = Colorize(n_class)

    # output all config and dump to file
    if comm.rank == 0:
        conf.dump_to_stdout()
        write_yaml(os.path.join(conf.save_path, "config.yaml"), conf)

    epoch = 0
    for itr in progress_iterator:
        if itr % ipe == 0:
            g_lr = g_lrs(epoch)
            d_lr = d_lrs(epoch)
            g_solver.set_learning_rate(g_lr)
            d_solver.set_learning_rate(d_lr)
            if comm.rank == 0:
                print("\n[epoch {}] update lr to ... g_lr: {}, d_lr: {}".format(
                    epoch, g_lr, d_lr))

            epoch += 1

        if conf.dataset == "cityscapes":
            im, ist, obj = data_iter.next()
            ist_mask.d = ist
        elif conf.dataset == "ade20k":
            im, obj = data_iter.next()
        else:
            raise NotImplemented()

        real.d = im
        obj_mask.d = obj

        # text embedding and create fake
        fake.forward()

        # update discriminator
        d_solver.zero_grad()
        d_loss.forward()
        d_loss.backward(clear_buffer=True)
        comm.all_reduced_solver_update(d_solver)

        # update generator
        ul_fake.grad.zero()
        g_solver.zero_grad()
        g_loss.forward()
        g_loss.backward(clear_buffer=True)

        # backward generator
        fake.backward(grad=None, clear_buffer=True)
        comm.all_reduced_solver_update(g_solver)

        # report iteration progress
        reporter()

        # report epoch progress
        show_epoch = itr // conf.show_interval
        if (itr % conf.show_interval) == 0:
            show_images = {"RealImages": real.data.get_data("r").transpose((0, 2, 3, 1)),
                           "ObjectMask": colorizer(obj).astype(np.uint8),
                           "GeneratedImage": fake.data.get_data("r").transpose((0, 2, 3, 1))}

            reporter.step(show_epoch, show_images)

        if (itr % conf.save_interval) == 0 and comm.rank == 0:
            nn.save_parameters(os.path.join(
                conf.save_path, 'param_{:03d}.h5'.format(show_epoch)))

    if comm.rank == 0:
        nn.save_parameters(os.path.join(conf.save_path, 'param_final.h5'))


if __name__ == '__main__':
    train()
