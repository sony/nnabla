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

import nnabla as nn
import nnabla.functions as F
import numpy as np
import os
from tqdm import trange

from .base import BaseTrainer


class VQVAEtrainer(BaseTrainer):

    def __init__(self, model, solver, data_loader, val_data_loader, monitor_train_loss, monitor_train_recon,
                 monitor_val_loss, monitor_val_recon, config, comm):

        super(VQVAEtrainer, self).__init__(solver, data_loader, val_data_loader, monitor_train_loss, monitor_train_recon,
                                           monitor_val_loss, monitor_val_recon, config, comm)

        self.model = model

        self.train_recon_path = os.path.join(
            config['monitor']['path'], config['monitor']['train_recon'])
        self.val_recon_path = os.path.join(
            config['monitor']['path'], config['monitor']['val_recon'])
        os.makedirs(self.train_recon_path, exist_ok=True)
        os.makedirs(self.val_recon_path, exist_ok=True)

        if self.dataset_name != 'imagenet':
            self.data_variance = np.var(
                self.data_loader._data_source._images/255.0)
        else:
            self.data_variance = 1

    def forward_pass(self, img_var, test=False):
        vq_loss, img_recon, perplexity = self.model(img_var, test=test)
        recon_loss = F.mean(F.squared_error(
            img_recon, img_var))/self.data_variance
        loss = recon_loss + vq_loss
        return loss, recon_loss, perplexity, img_recon

    def train(self, epoch):
        pbar = trange(self.iterations_per_epoch//self.comm.n_procs,
                      desc='Train at epoch '+str(epoch), disable=self.comm.rank > 0)
        epoch_loss = 0

        if epoch in self.learning_rate_decay_epochs:
            self.solver.set_learning_rate(
                self.solver.learning_rate()*self.learning_rate_decay_factor)

        for i in pbar:
            data = self.data_loader.next()
            if self.dataset_name == 'imagenet':
                img_var = nn.Variable(data[0].shape)
                img_var.data = data[0]
            else:
                img_var = self.convert_to_var(data[0])
            loss, recon_loss, perplexity, img_recon = self.forward_pass(
                img_var)

            pbar.set_description('Batch Loss: {}'.format(loss.d))
            epoch_loss += loss.d

            self.solver.set_parameters(
                nn.get_parameters(), reset=False, retain_state=True)
            self.solver.zero_grad()

            loss.backward(clear_buffer=True)

            params = [x.grad for x in nn.get_parameters().values()]
            self.comm.all_reduce(params, division=False, inplace=True)

            self.solver.weight_decay(self.weight_decay)
            self.solver.update()

        avg_epoch_loss = epoch_loss/self.iterations_per_epoch
        self.log_loss(epoch, avg_epoch_loss, train=True)
        self.save_image(img_var, os.path.join(
            self.train_recon_path, 'original_epoch_{}.png'.format(epoch)))
        self.save_image(img_recon, os.path.join(
            self.train_recon_path, 'recon_epoch_{}.png'.format(epoch)))

    def validate(self, epoch):
        pbar = trange(self.val_iterations_per_epoch,
                      desc='Validate at epoch '+str(epoch), disable=self.comm.rank > 0)
        epoch_loss = 0

        for i in pbar:
            data = self.val_data_loader.next()
            if self.dataset_name == 'imagenet':
                img_var = nn.Variable(data[0].shape)
                img_var.data = data[0]
            else:
                img_var = self.convert_to_var(data[0])
            loss, _, _, img_recon = self.forward_pass(img_var, test=True)

            pbar.set_description('Batch Loss: {}'.format(loss.d))
            epoch_loss += loss.d

        avg_epoch_loss = epoch_loss/self.iterations_per_epoch
        self.log_loss(epoch, avg_epoch_loss, train=False)
        self.save_image(img_var, os.path.join(
            self.val_recon_path, 'original_epoch_{}.png'.format(epoch)))
        self.save_image(img_recon, os.path.join(
            self.val_recon_path, 'recon_epoch_{}.png'.format(epoch)))
