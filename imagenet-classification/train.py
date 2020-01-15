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


import numpy as np

import nnabla as nn
import nnabla.logger as logger
import nnabla.monitor as M
import nnabla.functions as F
import nnabla.parametric_functions as PF

from args import get_args
import model_resnet_nhwc

import os
from collections import namedtuple

from utils import (
    CommunicatorWrapper,
    MomentumNoWeightDecayBn,
    EpochTrainer,
    EpochValidator,
    EpochReporter,
    LearningRateScheduler,
)

from data import (
    get_data_iterators,
)


def loss_function(pred, label, label_smoothing=0.1):
    l = F.softmax_cross_entropy(pred, label)
    if label_smoothing <= 0:
        return l
    return (1 - label_smoothing) * l - label_smoothing * F.mean(F.log_softmax(pred), axis=1, keepdims=True)


def get_model(args, num_classes, test=False, channel_last=False, with_error=True):
    """
    Create computation graph and variables.
    """
    nn_in_size = 224
    if channel_last:
        image = nn.Variable([args.batch_size, nn_in_size, nn_in_size, 4])
    else:
        image = nn.Variable([args.batch_size, 4, nn_in_size, nn_in_size])
    label = nn.Variable([args.batch_size, 1])
    pred, hidden = model_resnet_nhwc.resnet_imagenet(
        image, num_classes, args.num_layers, args.shortcut_type, test=test, tiny=False, channel_last=channel_last)
    pred.persistent = True
    loss = F.mean(loss_function(pred, label, args.label_smoothing))
    error = F.sum(F.top_n_error(pred, label, n=1))
    Model = namedtuple('Model',
                       ['image', 'label', 'pred', 'loss', 'error', 'hidden'])
    return Model(image, label, pred, loss, error, hidden)


def train():
    """
    Main script for training.
    """

    args = get_args()

    num_classes = 1000

    # Communicator and Context
    from nnabla.ext_utils import get_extension_context
    extension_module = "cudnn"  # TODO: Hard coded!!!
    ctx = get_extension_context(
        extension_module, device_id=args.device_id, type_config=args.type_config)
    comm = CommunicatorWrapper(ctx)
    nn.set_default_context(comm.ctx)

    from nnabla_ext.cuda import StreamEventHandler
    stream_event_handler = StreamEventHandler(int(comm.ctx.device_id))

    # Create data iterater
    data, vdata = get_data_iterators(args, comm, stream_event_handler)

    # Network for training
    t_model = get_model(args, num_classes,
                        test=False, channel_last=args.channel_last)

    # Network for validation
    v_model = get_model(args, num_classes,
                        test=True, channel_last=args.channel_last)

    # Solver
    loss_scaling = args.loss_scaling if args.type_config == 'half' else 1
    # To cancel loss scaling, learning rate is divided by loss_scaling.
    # Note this assumes legacy SGD w/ moemntum implementation,
    # otherwise, it is recommended to apply division at gradient itself
    # using scale_grad for example.
    base_learning_rate = args.learning_rate / loss_scaling

    # Weight decay is multiplied by loss_scaling to cancel the effect of loss_scaling
    # cancelling at learning rate.
    # Also, note that is is multiplied by number GPUs (processes),
    # because all-reduce sum over GPUs is performed before applying weight decay.
    weight_decay = args.weight_decay * loss_scaling * comm.n_procs
    solver = MomentumNoWeightDecayBn(base_learning_rate, 0.9)
    solver.set_parameters(nn.get_parameters())

    # Learning rate scheduler
    decay_rate = 0.1
    learning_rate_scheduler = LearningRateScheduler(
        base_learning_rate, args.learning_rate_decay_at, decay_rate, args.warmup_epochs)

    # Monitors
    monitor = None
    if comm.rank == 0:
        if not os.path.isdir(args.monitor_path):
            os.makedirs(args.monitor_path)
        monitor = M.Monitor(args.monitor_path)

    # Epoch runner
    train_epoch = EpochTrainer(t_model, solver, learning_rate_scheduler,
                               data, comm, monitor, loss_scaling, weight_decay,
                               stream_event_handler)
    val_epoch = None
    if args.val_interval > 0:
        val_epoch = EpochValidator(
            v_model, vdata, comm, monitor, stream_event_handler)

    # Epoch loop
    for epoch in range(args.max_epochs):
        # Save parameters
        if epoch > 0 and epoch % (args.model_save_interval) == 0 and comm.rank == 0:
            nn.save_parameters(os.path.join(
                args.monitor_path, 'param_%03d.h5' % epoch))

        # Run validation for examples in an epoch
        if val_epoch is not None \
           and epoch > 0 \
           and epoch % args.val_interval == 0:
            val_epoch.run(epoch)

        # Run training for examples in an epoch
        train_epoch.run(epoch)

    # Run final validation
    if val_epoch is not None:
        val_epoch.run(args.max_epochs)

    # Save the final model.
    if comm.rank == 0:
        nn.save_parameters(os.path.join(
            args.monitor_path,
            'param_%03d.h5' % (args.max_epochs)))


if __name__ == '__main__':
    train()
