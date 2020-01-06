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


# This file was forked from https://github.com/marvis/pytorch-yolo2 ,
# licensed under the MIT License (see LICENSE.external for more details).


from __future__ import division

import dataset
from utils import *
from arg_utils import Yolov2OptionTraining
import os

import nnabla as nn
import nnabla.solvers as S

from train_graph import TrainGraph

import random
random.seed(0)
import numpy as np
np.seterr(**{key: 'raise' for key in ['divide', 'over', 'invalid']})


def get_current_batch(seen, args):
    '''
    Get the current batch number from number of processed examples.
    '''
    return int(seen // (args.batch_size * args.accum_times))


class YoloSolver(object):
    def __init__(self, args):

        # Create solvers (only Convolution kernels require weight decay).
        param_convweights = {
            k: v for k, v in nn.get_parameters().items() if k.endswith("conv/W")}
        param_others = {k: v for k, v in nn.get_parameters().items()
                        if not k.endswith("conv/W")}
        convweights = S.Momentum(args.learning_rate, args.momentum)
        others = S.Momentum(args.learning_rate, args.momentum)
        convweights.set_parameters(param_convweights)
        others.set_parameters(param_others)

        # Init parameter gradients.
        convweights.zero_grad()
        others.zero_grad()

        # Set attributes.
        self.convweights = convweights
        self.others = others
        self.args = args
        self.batch_size = args.batch_size * args.accum_times
        self.rate = args.accum_times
        self.count = 0

    def get_current_lr(self, seen):
        '''
        Get the current learning rate by the current number of processed examples.

        Burn-in and step learning rate scheduling are employed.
        '''
        args = self.args
        batch_num = get_current_batch(seen, args)
        lr = float(args.learning_rate)

        # Burn-in (burn_in=1000, burn_in_power=4 by default)
        if batch_num < args.burn_in:
            return lr * ((float(batch_num) / args.burn_in) ** args.burn_in_power)

        # Steps is only implemented
        for step, scale in zip(args.steps, args.scales):
            if step > batch_num:
                return lr
            lr *= scale
        return lr

    def update_at_rate(self, lr, force=False):
        '''

        Returns: int
            Current learning rate
        '''
        if force and self.count > 0:
            self.update(lr)
            return True
        self.count = (self.count + 1) % self.rate
        if self.count == 0:
            self.update(lr)
            return True
        return False

    def update(self, lr):
        # Get current learning rate and set to solvers
        self.convweights.set_learning_rate(lr / self.batch_size)
        self.others.set_learning_rate(lr / self.batch_size)

        # Update
        self.convweights.weight_decay(self.args.decay * self.batch_size)
        self.convweights.update()
        self.others.update()

        # Reset parameter gradients and accumulation count
        self.convweights.zero_grad()
        self.others.zero_grad()
        self.count = 0


def train(args, sample_iter, nsamples, epoch, max_epochs, train_graph, yolo_solver, on_memory_data, total_batches):

    total_loss = []
    epoch_seen = 0
    tic = time.time()
    for each_batch in range(0, int(total_batches)):
        data_tensor, target_tensor = sample_iter.next()
        lr = yolo_solver.get_current_lr(train_graph.seen)
        nB = data_tensor.shape[0]
        print('size={}, '.format(data_tensor.shape[2]), end='')
        stats = train_graph.forward_backward(data_tensor, target_tensor)
        print('s=%d/%d, b=%d/%d, e=%d/%d, lr %g, mIOU %.3f, nGT %d, recall %d,'
              'proposals %d, loss: %.3f, exec: %.1f [ms]' % (
                    (stats.seen % nsamples), nsamples,
                    get_current_batch(stats.seen, args), args.max_batches,
                    epoch, max_epochs, lr, stats.mIoU, stats.nGT,
                    stats.nCorrect, stats.nProposals, stats.loss, stats.time))
        epoch_seen += nB
        total_loss.append(float(stats.loss))
        # Update parameters for every accum_times iterations.
        updated = yolo_solver.update_at_rate(lr)

    # Update if gradients are computed from previous update.
    yolo_solver.update_at_rate(lr, force=True)
    if (epoch+1) % args.save_interval == 0:
        logging('save weights to %s/%06d.h5' % (args.output, epoch+1))
        nn.save_parameters('%s/%06d.h5' % (args.output, epoch+1))
    return np.sum(total_loss) / epoch_seen


def main():
    # Training settings
    args = Yolov2OptionTraining().parse_args()

    nsamples = file_lines(args.train)

    set_default_context_by_args(args)

    # Training parameters
    max_epochs = args.max_batches * args.batch_size * args.accum_times / nsamples + 1

    # total number of batches
    total_batches = (nsamples+args.batch_size-1)//args.batch_size

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    ###############

    # Load parameters
    print("Load", args.weight, "...")
    if args.fine_tune:
        nn.load_parameters(args.weight)
        nn.parameter.pop_parameter("detection/conv/W")
        nn.parameter.pop_parameter("detection/conv/b")
    else:
        nn.load_parameters(args.weight)

    train_graph = TrainGraph(args, (args.size_aug[-1], args.size_aug[-1]))
    yolo_solver = YoloSolver(args)

    if args.on_memory_data:
        on_memory_data = dataset.load_on_memory_data(args.train)
    else:
        on_memory_data = None

    from nnabla.monitor import Monitor, MonitorSeries, MonitorTimeElapsed
    monitor = Monitor(args.output)
    monitor_loss = MonitorSeries('Training loss', monitor, interval=1)
    monitor_time = MonitorTimeElapsed('Time per epoch', monitor, interval=1)

    sample_iter = dataset.data_iterator_yolo(
        args.train, args,
        shuffle=True,
        train=True,
        image_sizes=args.size_aug,
        image_size_change_freq=args.batch_size * args.accum_times * 10,
        on_memory_data=on_memory_data,
        use_cv2=not args.disable_cv2, batch_size=args.batch_size)

    # Epoch loop
    for epoch in range(0, int(max_epochs)):
        loss = train(args, sample_iter, nsamples, epoch, max_epochs, train_graph,
                     yolo_solver, on_memory_data, total_batches)
        monitor_loss.add(epoch, loss)
        monitor_time.add(epoch)

    # Save the final parameters
    logging('save weights to %s/%06d.h5' % (args.output, epoch+1))
    nn.save_parameters('%s/%06d.h5' % (args.output, epoch+1))


if __name__ == '__main__':
    main()
