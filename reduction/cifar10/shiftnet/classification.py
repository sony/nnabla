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

from __future__ import absolute_import
from six.moves import range

import os
import sys
import functools

import nnabla as nn
import nnabla.logger as logger
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import nnabla.utils.save as save

from args import get_args
from cifar10_data import data_iterator_cifar10
from models import cifar10_resnet23_prediction, cifar10_shift_prediction, categorical_error


def train():
    args = get_args()

    # Get context.
    from nnabla.ext_utils import get_extension_context
    logger.info("Running in %s" % args.context)
    ctx = get_extension_context(
        args.context, device_id=args.device_id, type_config=args.type_config)
    nn.set_default_context(ctx)

    # Create CNN network for both training and testing.
    if args.net == "cifar10_resnet23_prediction":
        model_prediction = cifar10_resnet23_prediction
    if args.net == "cifar10_shift_csc_prediction":
        model_prediction = functools.partial(
            cifar10_shift_prediction, p=args.p, module="csc")
    if args.net == "cifar10_shift_sc2_prediction":
        model_prediction = functools.partial(
            cifar10_shift_prediction, p=args.p, module="sc2")

    data_iterator = data_iterator_cifar10
    c = 3
    h = w = 32
    n_train = 50000
    n_valid = 10000

    # TRAIN
    maps = args.maps

    # Create input variables.
    image = nn.Variable([args.batch_size, c, h, w])
    label = nn.Variable([args.batch_size, 1])
    # Create model_prediction graph.
    pred = model_prediction(image, maps=maps, test=False)
    pred.persistent = True
    # Create loss function.
    loss = F.mean(F.softmax_cross_entropy(pred, label))

    # TEST
    # Create input variables.
    vimage = nn.Variable([args.batch_size, c, h, w])
    vlabel = nn.Variable([args.batch_size, 1])
    # Create predition graph.
    vpred = model_prediction(vimage, maps=maps, test=True)

    # Create Solver.
    solver = S.Adam(args.learning_rate)
    solver.set_parameters(nn.get_parameters())

    # Create monitor.
    from nnabla.monitor import Monitor, MonitorSeries, MonitorTimeElapsed
    monitor = Monitor(args.monitor_path)
    monitor_loss = MonitorSeries("Training loss", monitor, interval=10)
    monitor_err = MonitorSeries("Training error", monitor, interval=10)
    monitor_time = MonitorTimeElapsed("Training time", monitor, interval=100)
    monitor_verr = MonitorSeries("Test error", monitor, interval=1)

    # Initialize DataIterator
    data = data_iterator(args.batch_size, True)
    vdata = data_iterator(args.batch_size, False)
    best_ve = 1.0
    ve = 1.0
    # Training loop.
    for i in range(args.max_iter):
        if i % args.val_interval == 0:
            # Validation
            ve = 0.0
            for j in range(int(n_valid / args.batch_size)):
                vimage.d, vlabel.d = vdata.next()
                vpred.forward(clear_buffer=True)
                ve += categorical_error(vpred.d, vlabel.d)
            ve /= int(n_valid / args.batch_size)
            monitor_verr.add(i, ve)
        if ve < best_ve:
            nn.save_parameters(os.path.join(
                args.model_save_path, 'params_%06d.h5' % i))
            best_ve = ve
        # Training forward
        image.d, label.d = data.next()
        solver.zero_grad()
        loss.forward(clear_no_need_grad=True)
        loss.backward(clear_buffer=True)
        solver.weight_decay(args.weight_decay)
        solver.update()
        e = categorical_error(pred.d, label.d)
        monitor_loss.add(i, loss.d.copy())
        monitor_err.add(i, e)
        monitor_time.add(i)

    ve = 0.0
    for j in range(int(n_valid / args.batch_size)):
        vimage.d, vlabel.d = vdata.next()
        vpred.forward(clear_buffer=True)
        ve += categorical_error(vpred.d, vlabel.d)
    ve /= int(n_valid / args.batch_size)
    monitor_verr.add(i, ve)

    parameter_file = os.path.join(
        args.model_save_path, 'params_{:06}.h5'.format(args.max_iter))
    nn.save_parameters(parameter_file)


if __name__ == '__main__':
    train()
