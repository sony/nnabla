
# Copyright (c) 2019-2020 Sony Corporation. All Rights Reserved.
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

from __future__ import division

import sys
import os

# Set path to neu
common_utils_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'utils'))
sys.path.append(common_utils_path)

from neu.yaml_wrapper import read_yaml, write_yaml
from neu.comm import CommunicatorWrapper

import nnabla as nn
import nnabla.communicators as C
import nnabla.monitor as M
import nnabla.solvers as S

import numpy as np
from tqdm import trange


def ceil_to_multiple(x, mul):
    '''
    Get a minimum integer >= x of a multiple of ``mul``.
    '''
    return (x + mul - 1) // mul


class MomentumNoWeightDecayBn(object):
    '''Momentum solver with disabling weight decay for BN scales and biases (beta and gamma).
    '''

    def __init__(self, learning_rate, momentum=0.9):
        self.solver = S.Momentum(learning_rate, momentum)
        self.solver_bn = S.Momentum(learning_rate, momentum)

    def set_parameters(self, params):
        params_bn = {k: v for k, v in params.items() if k.endswith(
            '/gamma') or k.endswith('/beta')}
        params = {k: v for k, v in params.items() if not (
            k.endswith('/gamma') or k.endswith('/beta'))}
        self.solver_bn.set_parameters(params_bn)
        self.solver.set_parameters(params)

    def weight_decay(self, wd):
        # Weight decay is not performed for BN scales and biases
        self.solver.weight_decay(wd)

    def update(self):
        self.solver.update()
        self.solver_bn.update()

    def zero_grad(self):
        self.solver.zero_grad()
        self.solver_bn.zero_grad()

    def learning_rate(self):
        return self.solver.learning_rate()

    def set_learning_rate(self, lr):
        self.solver.set_learning_rate(lr)
        self.solver_bn.set_learning_rate(lr)


class EpochReporter(object):
    def __init__(self, name, monitor, comm, loss, error, batch_size, time=True, flush_interval=10):
        self.name = name
        self.comm = comm
        self.epoch_loss = 0.0
        self.epoch_error = 0
        self.batch_counter = 0
        self.loss = loss
        self.error = error
        self.batch_size = batch_size
        if self.comm.rank == 0:
            self.monitor_loss = M.MonitorSeries(
                "%s loss" % name, monitor, interval=1)
            self.monitor_err = M.MonitorSeries(
                "%s error" % name, monitor, interval=1)
            self.monitor_time = None
            if time:
                self.monitor_time = M.MonitorTimeElapsed(
                    "Epoch time", monitor, interval=1)
        self.flush_interval = flush_interval

    def reset_buff(self):
        for b in self.buff:
            b.zero()

    def reset(self, epoch, pbar):
        self.epoch = epoch
        self.epoch_loss = 0.0
        self.epoch_error = 0
        self.batch_counter = 0
        self.pbar = pbar
        self.buff = [nn.NdArray(), nn.NdArray()]
        self.reset_buff()
        self.flush = True

    def flush_buff(self):
        error_per_iter = int(np.round(self.buff[0].get_data('r')))
        loss_per_iter = float(self.buff[1].get_data('r'))
        self.reset_buff()
        self.epoch_error += error_per_iter
        self.epoch_loss += loss_per_iter

    def update(self):
        with nn.context_scope(self.comm.ctx_float):
            self.buff[0] += self.error.data
            self.buff[1] += self.loss.data
        self.batch_counter += 1
        self.flush = False

    def __call__(self, lr=0, force=False):
        if not force and (
                self.batch_counter == 0 or
                self.batch_counter % self.flush_interval != 0):
            return
        self.flush_buff()
        self.pbar.set_description(
            '{} epoch {} (lr={:.2e}): loss={:.4} error={:.4}'.format(
                self.name,
                self.epoch, lr,
                self.epoch_loss / self.batch_counter,
                self.epoch_error / (self.batch_counter * self.batch_size)))

    def on_epoch_end(self):
        if not self.flush:
            self.flush_buff()
        epoch_loss = nn.NdArray.from_numpy_array(
            np.asarray(self.epoch_loss / self.batch_counter, dtype=np.float32))
        epoch_error = nn.NdArray.from_numpy_array(
            np.asarray(float(self.epoch_error) / (self.batch_counter * self.batch_size), dtype=np.float32))
        self.comm.all_reduce(epoch_loss, division=True, inplace=True)
        self.comm.all_reduce(epoch_error, division=True, inplace=True)
        if self.comm.rank == 0:
            self.monitor_loss.add(self.epoch, epoch_loss.data.copy())
            self.monitor_err.add(self.epoch, epoch_error.data.copy())
            if self.monitor_time is not None:
                self.monitor_time.add(self.epoch)


class EpochTrainer(object):
    def __init__(self, model, solver, learning_rate_scheduler, data, comm, monitor,
                 loss_scaling, weight_decay, stream_event_handler, mixup=None):
        batch_size = model.image.shape[0]
        self.model = model

        self.solver = solver
        self.data = data
        self.comm = comm
        self.learning_rate_scheduler = learning_rate_scheduler
        self.batch_size = batch_size
        # LR is multiplied by batch size because the learning rate returned by
        # scheduler is defined as learning rate per example.
        # Also, to cancel loss scaling, learning rate is divided by loss_scaling.
        # Note this assumes legacy SGD w/ moemntum implementation,
        # otherwise, it is recommended to apply division at gradient itself
        # using scale_grad for example.
        self.lr_factor = batch_size / loss_scaling
        self.loss_scaling = loss_scaling
        # Weight decay is multiplied by loss_scaling to cancel the effect of loss_scaling
        # cancelling at learning rate.
        # Also, note that is is multiplied by number GPUs (processes),
        # because all-reduce sum over GPUs is performed before applying weight decay.
        self.weight_decay = weight_decay * loss_scaling * comm.n_procs
        self.stream_event_handler = stream_event_handler
        self.num_iter_per_epoch = ceil_to_multiple(data.size, batch_size)
        self.reporter = EpochReporter(
            'Train', monitor, comm, model.loss, model.error, batch_size)
        self.mixup = mixup

    def run(self, epoch):

        # Update epoch counter of lr scheduler
        self.learning_rate_scheduler.set_epoch(epoch)

        # Training loop
        pbar = trange(self.num_iter_per_epoch,
                      desc='Train at epoch %d' % epoch,
                      disable=self.comm.rank > 0)

        # pbar = range(self.num_iter_per_epoch)

        self.reporter.reset(epoch, pbar)

        for i in pbar:
            # nvtx.range_push("train_{}".format(i))

            # Update learning rate
            lr = self.learning_rate_scheduler.get_lr_and_update()
            self.solver.set_learning_rate(lr * self.lr_factor)

            # wait here until back-prop has been finished
            self.stream_event_handler.event_synchronize()

            next_image, next_label = self.data.next()
            self.model.image.data = next_image
            self.model.label.data = next_label

            # Sample mixup ratios
            if self.mixup is not None:
                self.mixup.reset_mixup_ratio()

            # Synchronizing null-stream and host here makes update faster. I'm not sure why.
            self.stream_event_handler.default_stream_synchronize()

            self.reporter(lr)

            nn.forward_all([self.model.loss, self.model.error],
                           clear_no_need_grad=True)
            # self.model.loss.forward(clear_no_need_grad=True, function_pre_hook=None)
            comm_callback = self.comm.get_all_reduce_callback()
            self.solver.zero_grad()
            self.model.loss.backward(
                self.loss_scaling, clear_buffer=True,
                communicator_callbacks=comm_callback
            )

            # Subscript event
            self.stream_event_handler.add_default_stream_event()

            # # Update
            self.solver.weight_decay(self.weight_decay)
            self.solver.update()
            self.reporter.update()

            # if i == 10:
            #     import sys

            # nvtx.range_pop()

        self.reporter(lr, force=True)
        self.reporter.on_epoch_end()


class EpochValidator(object):
    def __init__(self, model, data, comm, monitor, stream_event_handler):
        batch_size = model.image.shape[0]
        self.model = model
        self.data = data
        self.comm = comm
        self.stream_event_handler = stream_event_handler
        self.batch_size = batch_size
        self.num_iter_per_epoch = ceil_to_multiple(self.data.size, batch_size)
        self.reporter = EpochReporter(
            'Validation', monitor, comm, model.loss, model.error, batch_size,
            time=False)

    def run(self, epoch):

        pbar = trange(self.num_iter_per_epoch,
                      desc='Val at epoch %d' % epoch,
                      disable=self.comm.rank > 0)

        self.reporter.reset(epoch, pbar)
        for i in pbar:
            # wait here until forward-prop has been finished
            self.stream_event_handler.event_synchronize()
            next_image, next_label = self.data.next()
            self.reporter(0)
            self.model.image.data = next_image
            self.model.label.data = next_label
            nn.forward_all([self.model.loss, self.model.error],
                           clear_buffer=True)
            self.stream_event_handler.add_default_stream_event()
            self.reporter.update()

        self.reporter(0, force=True)
        self.reporter.on_epoch_end()
