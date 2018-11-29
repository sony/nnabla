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


from nnabla import Variable
from nnabla import logger
from nnabla.monitor import Monitor, MonitorSeries, MonitorTimeElapsed

import datasets
from helpers import MonitorImageTileWithName
from networks import Generator, Discriminator
from functions import pixel_wise_feature_vector_normalization

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import numpy as np


class Trainer:
    def __init__(self, data_iterator,
                 generator, discriminator,
                 solver_gen, solver_dis,
                 monitor_path,
                 monitor_loss_gen, monitor_loss_dis,
                 monitor_p_fake, monitor_p_real,
                 monitor_time,
                 monitor_image_tile,
                 resolution_list, channel_list, n_latent,
                 n_critic=1,
                 save_image_interval=1,
                 hyper_sphere=True,
                 l2_fake_weight=1.):
        # Config
        self.resolution_list = resolution_list
        self.channel_list = channel_list
        self.n_latent = n_latent
        self.n_critic = n_critic
        self.save_image_interval = save_image_interval
        self.hyper_sphere = hyper_sphere
        self.l2_fake_weight = l2_fake_weight
        self.global_itr = 0

        # DataIterator
        self.di = data_iterator

        # Model
        self.gen = generator
        self.dis = discriminator

        # Solver
        self.solver_gen = solver_gen
        self.solver_dis = solver_dis

        # Monitor
        self.monitor_path = monitor_path
        self.monitor_loss_dis = monitor_loss_dis
        self.monitor_loss_gen = monitor_loss_gen
        self.monitor_p_real = monitor_p_real
        self.monitor_p_fake = monitor_p_fake
        self.monitor_time = monitor_time
        self.monitor_image_tile = monitor_image_tile

    def train(self, epoch_per_resolution):
        # Add new resolution and channel
        self.gen.grow(self.resolution_list[0], self.channel_list[0])
        self.dis.grow(self.resolution_list[0], self.channel_list[0])

        # Fix test random
        self.z_test = np.random.randn(
            self.di.batch_size, self.n_latent, 1, 1)  # Fix random seed for test

        # TODO: change batchsize when the spatial size is greater than 128.
        for i in range(len(self.resolution_list) - 1):
            # Train at this resolution
            self._train(epoch_per_resolution)
            self.gen.save_parameters(
                self.monitor_path, "Gen_phase_{}".format(self.resolution_list[i]))
            self.dis.save_parameters(
                self.monitor_path, "Dis_phase_{}".format(self.resolution_list[i]))

            # Add new resolution and channel
            self.gen.grow(
                self.resolution_list[i + 1], self.channel_list[i + 1])
            self.dis.grow(
                self.resolution_list[i + 1], self.channel_list[i + 1])

            # Train in transition period
            self._transition(epoch_per_resolution)

            # Monitor
            self.monitor_time.add(i)

            # Save parameter
            self.gen.save_parameters(self.monitor_path, "Gen_phase_{}to{}".format(
                self.resolution_list[i], self.resolution_list[i + 1]))
            self.dis.save_parameters(self.monitor_path, "Dis_phase_{}to{}".format(
                self.resolution_list[i], self.resolution_list[i + 1]))

            # Clear unnecessary memory (i.e., memory related to_RGB at each resolution)
            import nnabla_ext.cuda
            nnabla_ext.cuda.clear_memory_cache()

        # Train at the final resolution
        self._train(epoch_per_resolution, each_save=True)
        self.monitor_time.add(i)

    def _train(self, epoch_per_resolution, each_save=False):
        batch_size = self.di.batch_size
        resolution = self.gen.resolution_list[-1]
        logger.info("phase : {}".format(resolution))

        kernel_size = self.resolution_list[-1] // resolution
        kernel = (kernel_size, kernel_size)

        img_name = "original_phase_{}".format(resolution)
        img, _ = self.di.next()
        self.monitor_image_tile.add(img_name, img)

        for epoch in range(epoch_per_resolution):
            logger.info("epoch : {}".format(epoch + 1))
            itr = 0
            current_epoch = self.di.epoch
            while self.di.epoch == current_epoch:
                img, _ = self.di.next()
                x = nn.Variable.from_numpy_array(img)
                z = F.randn(shape=(batch_size, self.n_latent, 1, 1))
                z = pixel_wise_feature_vector_normalization(
                    z) if self.hyper_sphere else z
                y = self.gen(z, test=True)

                y.need_grad = False
                x_r = F.average_pooling(x, kernel=kernel)

                p_real = self.dis(x_r)
                p_fake = self.dis(y)
                p_real.persistent, p_fake.persistent = True, True

                loss_dis = F.mean(F.pow_scalar((p_real - 1), 2.)
                                  + F.pow_scalar(p_fake, 2.) * self.l2_fake_weight)
                loss_dis.persistent = True

                if itr % self.n_critic + 1 == self.n_critic:
                    with nn.parameter_scope("discriminator"):
                        self.solver_dis.set_parameters(nn.get_parameters(),
                                                       reset=False, retain_state=True)
                        self.solver_dis.zero_grad()
                        loss_dis.backward(clear_buffer=True)
                        self.solver_dis.update()
                z = F.randn(shape=(batch_size, self.n_latent, 1, 1))
                z = pixel_wise_feature_vector_normalization(
                    z) if self.hyper_sphere else z
                y = self.gen(z, test=False)
                p_fake = self.dis(y)
                p_fake.persistent = True

                loss_gen = F.mean(F.pow_scalar((p_fake - 1), 2.))
                loss_gen.persistent = True

                with nn.parameter_scope("generator"):
                    self.solver_gen.set_parameters(nn.get_parameters(),
                                                   reset=False, retain_state=True)
                    self.solver_gen.zero_grad()
                    loss_gen.backward(clear_buffer=True)
                    self.solver_gen.update()

                # Monitor
                self.monitor_p_real.add(
                    self.global_itr, p_real.d.copy().mean())
                self.monitor_p_fake.add(
                    self.global_itr, p_fake.d.copy().mean())
                self.monitor_loss_dis.add(self.global_itr, loss_dis.d.copy())
                self.monitor_loss_gen.add(self.global_itr, loss_gen.d.copy())

                itr += 1
                self.global_itr += 1

            if epoch % self.save_image_interval + 1 == self.save_image_interval:
                z = nn.Variable.from_numpy_array(self.z_test)
                z = pixel_wise_feature_vector_normalization(
                    z) if self.hyper_sphere else z
                y = self.gen(z, test=True)
                img_name = "phase_{}_epoch_{}".format(resolution, epoch + 1)
                self.monitor_image_tile.add(
                    img_name, F.unpooling(y, kernel=kernel))

            if each_save:
                self.gen.save_parameters(self.monitor_path, "Gen_phase_{}_epoch_{}".format(
                    self.resolution_list[-1], epoch+1))
                self.dis.save_parameters(self.monitor_path, "Dis_phase_{}_epoch_{}".format(
                    self.resolution_list[-1], epoch+1))

    def _transition(self, epoch_per_resolution):
        batch_size = self.di.batch_size
        resolution = self.gen.resolution_list[-1]
        phase = "{}to{}".format(
            self.gen.resolution_list[-2], self.gen.resolution_list[-1])
        logger.info("phase : {}".format(phase))

        kernel_size = self.resolution_list[-1] // resolution
        kernel = (kernel_size, kernel_size)

        total_itr = (self.di.size // batch_size + 1) * epoch_per_resolution
        global_itr = 1.
        alpha = global_itr / total_itr

        for epoch in range(epoch_per_resolution):
            logger.info("epoch : {}".format(epoch + 1))
            itr = 0
            current_epoch = self.di.epoch
            while self.di.epoch == current_epoch:
                img, _ = self.di.next()
                x = nn.Variable.from_numpy_array(img)

                z = F.randn(shape=(batch_size, self.n_latent, 1, 1))
                z = pixel_wise_feature_vector_normalization(
                    z) if self.hyper_sphere else z
                y = self.gen.transition(z, alpha, test=True)

                y.need_grad = False
                x_r = F.average_pooling(x, kernel=kernel)

                p_real = self.dis.transition(x_r, alpha)
                p_fake = self.dis.transition(y, alpha)

                loss_dis = F.mean(F.pow_scalar((p_real - 1), 2.)
                                  + F.pow_scalar(p_fake, 2.) * self.l2_fake_weight)

                if itr % self.n_critic + 1 == self.n_critic:
                    with nn.parameter_scope("discriminator"):
                        self.solver_dis.set_parameters(nn.get_parameters(),
                                                       reset=False, retain_state=True)
                        self.solver_dis.zero_grad()
                        loss_dis.backward(clear_buffer=True)
                        self.solver_dis.update()

                z = F.randn(shape=(batch_size, self.n_latent, 1, 1))
                z = pixel_wise_feature_vector_normalization(
                    z) if self.hyper_sphere else z
                y = self.gen.transition(z, alpha, test=False)
                p_fake = self.dis.transition(y, alpha)

                loss_gen = F.mean(F.pow_scalar((p_fake - 1), 2))
                with nn.parameter_scope("generator"):
                    self.solver_gen.set_parameters(
                        nn.get_parameters(), reset=False, retain_state=True)
                    self.solver_gen.zero_grad()
                    loss_gen.backward(clear_buffer=True)
                    self.solver_gen.update()

                itr += 1
                global_itr += 1.
                alpha = global_itr / total_itr

            if epoch % self.save_image_interval + 1 == self.save_image_interval:
                z = nn.Variable.from_numpy_array(self.z_test)
                z = pixel_wise_feature_vector_normalization(
                    z) if self.hyper_sphere else z
                y = self.gen.transition(z, alpha)
                img_name = "phase_{}_epoch_{}".format(phase, epoch + 1)
                self.monitor_image_tile.add(
                    img_name, F.unpooling(y, kernel=kernel))
