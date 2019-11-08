import os
import numpy as np
from tqdm import trange

import nnabla as nn
import nnabla.solvers as S
import nnabla.functions as F
from nnabla.logger import logger

from utils import *
from models import LocalGenerator, encode_inputs


class Trainer(object):
    def __init__(self, tconf, mconf, comm, data_list):
        rng = np.random.RandomState(tconf.random_seed)

        self.train_conf = tconf
        self.model_conf = mconf

        self.bs = tconf.batch_size

        self.image_shape = tuple(
            x * mconf.g_n_scales for x in mconf.base_image_shape)
        self.data_iter = create_cityscapes_iterator(self.bs, data_list,
                                                    image_shape=self.image_shape,
                                                    rng=rng, flip=tconf.flip)
        if comm.n_procs > 1:
            self.data_iter = self.data_iter.slice(
                rng, num_of_slices=comm.n_procs, slice_pos=comm.rank)

        self.comm = comm
        self.fix_global_epoch = max(tconf.fix_global_epoch, 0)
        self.use_encoder = False  # currently encoder is not supported.

        self.load_path = tconf.load_path

    def train(self):
        real = nn.Variable(shape=(self.bs, 3) + self.image_shape)
        inst_label = nn.Variable(shape=(self.bs,) + self.image_shape)
        id_label = nn.Variable(shape=(self.bs,) + self.image_shape)

        id_onehot, bm = encode_inputs(
            inst_label, id_label, n_ids=self.model_conf.n_label_ids, use_encoder=self.use_encoder)

        x = F.concatenate(id_onehot, bm, axis=1)

        # generator
        # Note that only global generator would be used in the case of g_scales = 1.
        generator = LocalGenerator()
        fake, _, = generator(x,
                             lg_channels=self.model_conf.lg_channels,
                             gg_channels=self.model_conf.gg_channels,
                             n_scales=self.model_conf.g_n_scales,
                             lg_n_residual_layers=self.model_conf.lg_num_residual_loop,
                             gg_n_residual_layers=self.model_conf.gg_num_residual_loop)
        unlinked_fake = fake.get_unlinked_variable(need_grad=True)

        # discriminator
        discriminator = PatchGAN(n_scales=self.model_conf.d_n_scales, use_spectral_normalization=False)
        d_real_out, d_real_feats = discriminator(F.concatenate(real, x, axis=1))
        d_fake_out, d_fake_feats = discriminator(F.concatenate(unlinked_fake, x, axis=1))
        g_gan, g_feat, d_real, d_fake = discriminator.get_loss(d_real_out, d_real_feats,
                                                               d_fake_out, d_fake_feats,
                                                               use_fm=True,
                                                               fm_lambda=self.train_conf.lambda_feat,
                                                               gan_loss_type="ls")

        g_vgg = vgg16_perceptual_loss(
            real, unlinked_fake) * self.train_conf.lambda_perceptual

        set_persistent_all(bm, fake, fake, g_gan,
                           g_feat, g_vgg, d_real, d_fake)

        g_loss = g_gan + g_feat + g_vgg
        d_loss = 0.5 * (d_real + d_fake)

        # load parameters
        if self.load_path:
            if not os.path.exists(self.load_path):
                logger.warn("Path to load params is not found."
                            " Loading params is skipped. ({})".format(self.load_path))
            else:
                nn.load_parameters(self.load_path)

        # Setup Solvers
        g_solver = S.Adam(beta1=0.5)
        g_solver.set_parameters(get_params_startswith("generator/local"))

        d_solver = S.Adam(beta1=0.5)
        d_solver.set_parameters(get_params_startswith("discriminator"))

        # lr scheduler
        lr_schduler = LinearDecayScheduler(self.train_conf.base_lr, 0.,
                                           start_iter=self.train_conf.lr_decay_starts,
                                           end_iter=self.train_conf.max_epochs)

        # Setup Reporter
        losses = {"g_gan": g_gan, "g_feat": g_feat,
                  "g_vgg": g_vgg, "d_real": d_real, "d_fake": d_fake}
        reporter = Reporter(self.comm, losses, self.train_conf.save_path)

        # for label2color
        label2color = Colorize(self.model_conf.n_label_ids)

        for epoch in range(self.train_conf.max_epochs):
            if epoch == self.fix_global_epoch:
                g_solver.set_parameters(get_params_startswith(
                    "generator"), reset=False, retain_state=True)

            # update learning rate for current epoch
            lr = lr_schduler(epoch)
            g_solver.set_learning_rate(lr)
            d_solver.set_learning_rate(lr)

            progress_iterator = trange(self.data_iter._size // self.bs,
                                       desc="[epoch {}]".format(epoch), disable=self.comm.rank > 0)

            reporter.start(progress_iterator)

            for i in progress_iterator:
                image, instance_id, object_id = self.data_iter.next()

                real.d = image
                inst_label.d = instance_id
                id_label.d = object_id

                # create fake
                fake.forward()

                # update discriminator
                d_solver.zero_grad()
                d_loss.forward()
                d_loss.backward(clear_buffer=True)

                if self.comm.n_procs > 1:
                    params = [
                        x.grad for x in d_solver.get_parameters().values()]
                    self.comm.all_reduce(params, division=False, inplace=False)
                d_solver.update()

                # update generator
                unlinked_fake.grad.zero()
                g_solver.zero_grad()
                g_loss.forward()
                g_loss.backward(clear_buffer=True)

                # backward generator
                fake.backward(grad=None, clear_buffer=True)

                if self.comm.n_procs > 1:
                    params = [
                        x.grad for x in g_solver.get_parameters().values()]
                    self.comm.all_reduce(params, division=False, inplace=False)
                g_solver.update()

                # report iteration progress
                reporter()

            # report epoch progress
            show_images = {"InputImage": label2color(id_label.data.get_data("r")).astype(np.uint8),
                           # "InputBoundary": bm.data.get_data("r").transpose((0, 2, 3, 1)),
                           "GeneratedImage": fake.data.get_data("r").transpose((0, 2, 3, 1)),
                           "RealImagse": real.data.get_data("r").transpose((0, 2, 3, 1))}
            reporter.step(epoch, show_images)

            if (epoch % 10) == 0 and self.comm.rank == 0:
                nn.save_parameters(os.path.join(
                    self.train_conf.save_path, 'param_{:03d}.h5'.format(epoch)))

        if self.comm.rank == 0:
            nn.save_parameters(os.path.join(
                self.train_conf.save_path, 'param_final.h5'))
