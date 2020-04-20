# Copyright (c) 2017-2020 Sony Corporation. All Rights Reserved.
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

import numpy as np

import nnabla as nn
import nnabla.logger as logger
import nnabla.monitor as M
import nnabla.functions as F
import nnabla.parametric_functions as PF

from args import get_train_args

import os
from collections import namedtuple

from utils import (
    CommunicatorWrapper,
    MomentumNoWeightDecayBn,
    EpochTrainer,
    EpochValidator,
    EpochReporter,
    save_args
)

from data import (
    get_data_iterators,
)


def create_mixup_or_none(alpha, num_classes, comm):
    from utils.mixup import MixUp
    # Create different random generators over workers.
    rng = np.random.RandomState(726 + comm.rank)
    if alpha > 0:
        return MixUp(alpha, num_classes, rng)
    return None


def create_learning_rate_scheduler(cfg):
    from utils import learning_rate_scheduler as lrs
    lrs_cfg = cfg.lr_scheduler
    lrs_type = getattr(lrs, lrs_cfg.type)

    # Overwrite config
    if 'epochs' in lrs_cfg.args and lrs_cfg.args.epochs is None:
        lrs_cfg.args.epochs = cfg.epochs

    return lrs_type(**lrs_cfg.args)


def get_model(args, num_classes, test=False, channel_last=False,
              mixup=None, channels=4, spatial_size=224, label_smoothing=0,
              ctx_for_loss=None):
    """
    Create computation graph and variables.
    """
    from models import build_network
    from utils.loss import softmax_cross_entropy_with_label_smoothing

    if hasattr(spatial_size, '__len__'):
        assert len(spatial_size) == 2, \
            f'Spatial size must be a scalar or a tuple of two ints. Given {spatial_size}'
        spatial_shape = tuple(spatial_size)
    else:
        spatial_shape = (spatial_size, spatial_size)
    if channel_last:
        image = nn.Variable(
            (args.batch_size, spatial_shape[0], spatial_shape[1], channels))
    else:
        image = nn.Variable((args.batch_size, channels) + spatial_shape)
    label = nn.Variable([args.batch_size, 1])

    in_image = image
    in_label = label
    if mixup is not None:
        image, label = mixup.mix_data(image, label)
    pred, hidden = build_network(
        image, num_classes, args.arch,
        test=test, channel_last=channel_last)
    pred.persistent = True

    def define_loss(pred, in_label, label, label_smoothing):
        loss = F.mean(softmax_cross_entropy_with_label_smoothing(
            pred, label, label_smoothing))
        error = F.sum(F.top_n_error(pred, in_label, n=1))
        return loss, error

    # Use specified context if possible.
    # We use it when we pass float32 context to avoid nan issue
    if ctx_for_loss is not None:
        with nn.context_scope(ctx_for_loss):
            loss, error = define_loss(pred, in_label, label, label_smoothing)
    else:
        loss, error = define_loss(pred, in_label, label, label_smoothing)
    Model = namedtuple('Model',
                       ['image', 'label', 'pred', 'loss', 'error', 'hidden'])
    return Model(in_image, in_label, pred, loss, error, hidden)


def train():
    """
    Main script for training.
    """
    args, train_config = get_train_args()

    num_classes = args.num_classes

    # Communicator and Context
    from nnabla.ext_utils import get_extension_context
    extension_module = "cudnn"  # TODO: Hard coded!!!
    ctx = get_extension_context(
        extension_module, device_id=args.device_id, type_config=args.type_config)
    comm = CommunicatorWrapper(ctx)
    nn.set_default_context(comm.ctx)
    # To utilize TensorCore in FP16
    channels = 4 if args.type_config == 'half' else 3

    from nnabla_ext.cuda import StreamEventHandler
    stream_event_handler = StreamEventHandler(int(comm.ctx.device_id))

    # Create data iterater
    data, vdata = get_data_iterators(args, comm, channels)

    # Create mixup object
    mixup = create_mixup_or_none(train_config.mixup, num_classes, comm)

    # Network for training
    t_model = get_model(args, num_classes,
                        test=False, channel_last=args.channel_last,
                        mixup=mixup, channels=channels,
                        label_smoothing=train_config.label_smoothing,
                        ctx_for_loss=comm.ctx_float)

    # Network for validation
    v_model = get_model(args, num_classes,
                        test=True, channel_last=args.channel_last,
                        channels=channels)

    # Solver
    # lr will be set later
    solver = MomentumNoWeightDecayBn(1, train_config.momentum)
    solver.set_parameters(nn.get_parameters())

    # Learning rate scheduler
    learning_rate_scheduler = create_learning_rate_scheduler(train_config)

    # Monitors
    monitor = None
    if comm.rank == 0:
        if not os.path.isdir(args.monitor_path):
            os.makedirs(args.monitor_path)
        monitor = M.Monitor(args.monitor_path)
        save_args(args, train_config)

    # Epoch runner
    loss_scaling = train_config.loss_scaling if args.type_config == 'half' else 1
    train_epoch = EpochTrainer(t_model, solver, learning_rate_scheduler,
                               data, comm, monitor, loss_scaling,
                               train_config.weight_decay,
                               stream_event_handler, mixup)
    val_epoch = None
    if args.val_interval > 0:
        val_epoch = EpochValidator(
            v_model, vdata, comm, monitor, stream_event_handler)

    # Epoch loop
    for epoch in range(train_config.epochs):
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
        val_epoch.run(train_config.epochs)

    # Save the final model.
    if comm.rank == 0:
        nn.save_parameters(os.path.join(
            args.monitor_path,
            'param_%03d.h5' % (train_config.epochs)))


if __name__ == '__main__':
    train()
