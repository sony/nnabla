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

from abc import ABC
from pathlib import Path

import nnabla as nn
import nnabla.functions as F
import numpy as np
from scipy.io import wavfile
from tqdm import trange

from .logger import ProgressMeter


class Trainer(ABC):
    r"""Implementation of Trainer.

    Args:
        model (model.module.Module): WaveGlow model.
        dataloader (dict): A dataloader.
        optimizer (Optimizer): An optimizer used to update the parameters.
        hparams (HParams): Hyper-parameters.
    """

    def __init__(self, model, dataloader, optimizer, hparams):
        self.model = model
        self.dataloader = dataloader
        self.hparams = hparams
        self.one_epoch_train = dataloader['train'].size // hparams.batch_size
        self.one_epoch_valid = dataloader['valid'].size // hparams.batch_size
        self.placeholder = dict()
        self.optimizer = optimizer
        self.monitor = ProgressMeter(
            self.one_epoch_train, hparams.output_path, quiet=hparams.comm.rank > 0)
        hparams.save(Path(hparams.output_path) / 'settings.json')

    def update_graph(self, key='train'):
        r"""Builds the graph and update the placeholder.

        Args:
            key (str, optional): Type of computational graph. Defaults to 'train'.
        """
        pass

    def callback_on_start(self):
        r"""Calls this on starting the training."""
        self.update_graph('train')
        params = self.model.get_parameters(grad_only=True)
        self.optimizer.set_parameters(params)
        self.update_graph('valid')
        self.loss = nn.NdArray.from_numpy_array(np.zeros((1,)))
        if self.hparams.comm.n_procs > 1:
            self._grads = [x.grad for x in params.values()]

    def run(self):
        r"""Run the training process."""
        self.callback_on_start()
        for cur_epoch in range(self.hparams.epoch):

            self.monitor.reset()
            lr = self.optimizer.get_learning_rate()

            self.monitor.info(f'Running epoch={cur_epoch}\tlr={lr:.5f}\n')
            self.cur_epoch = cur_epoch

            for i in range(self.one_epoch_train):
                self.train_on_batch()
                if i % (self.hparams.print_frequency) == 0:
                    self.monitor.display(i, ['train/l_net'])

            for i in trange(self.one_epoch_valid, disable=self.hparams.comm.rank > 0):
                self.valid_on_batch()

            self.callback_on_epoch_end()

        self.callback_on_finish()
        self.monitor.close()

    def train_on_batch(self):
        r"""Calls this on traning batch."""

    def valid_on_batch(self):
        r"""Calls this on validation batch."""
        pass

    def callback_on_epoch_end(self):
        r"""Calls this on finishing one epoch."""
        pass

    def callback_on_finish(self):
        r"""Calls this on finishing the run method."""
        if self.hparams.comm.rank == 0:
            path = str(Path(self.hparams.output_path) / 'model.h5')
            self.model.save_parameters(path)
