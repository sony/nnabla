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


import os
import time
from args import get_args
from cifar10_data import data_iterator_cifar10
from cifar100_data import data_iterator_cifar100
import nnabla as nn
import nnabla.communicators as C
from nnabla.contrib.context import extension_context
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import numpy as np
import functools
from models import (resnet23_prediction, categorical_error, loss_function)


def train():
    """
    Naive Multi-Device Training

    NOTE: the communicator exposes low-level interfaces

    * Parse command line arguments.
    * Instantiate a communicator and set parameter variables.
    * Specify contexts for computation.
    * Initialize DataIterator.
    * Construct a computation graph for training and one for validation.
    * Initialize solver and set parameter variables to that.
    * Create monitor instances for saving and displaying training stats.
    * Training loop
      * Computate error rate for validation data (periodically)
      * Get a next minibatch.
      * Execute forwardprop
      * Set parameter gradients zero
      * Execute backprop.
      * Solver updates parameters by using gradients computed by backprop.
      * Compute training error
    """
    # Parse args
    args = get_args()
    n_train_samples = 50000
    bs_valid = args.batch_size
    rng = np.random.RandomState(313)
    if args.net == "cifar10_resnet23":
        prediction = functools.partial(
            resnet23_prediction, rng=rng, ncls=10, nmaps=64, act=F.relu)
        data_iterator = data_iterator_cifar10
    if args.net == "cifar100_resnet23":
        prediction = functools.partial(
            resnet23_prediction, rng=rng, ncls=100, nmaps=384, act=F.elu)
        data_iterator = data_iterator_cifar100

    # Communicator and Context
    extension_module = "cuda.cudnn"
    ctx = extension_context(extension_module)
    comm = C.MultiProcessDataParalellCommunicator(ctx)
    comm.init()
    n_devices = comm.size
    mpi_rank = comm.rank
    mpi_local_rank = comm.local_rank
    device_id = mpi_local_rank
    ctx = extension_context(extension_module, device_id=device_id)
    nn.set_default_context(ctx)

    # Create training graphs
    test = False
    image_train = nn.Variable((args.batch_size, 3, 32, 32))
    label_train = nn.Variable((args.batch_size, 1))
    pred_train = prediction(image_train, test)
    loss_train = loss_function(pred_train, label_train)
    input_image_train = {"image": image_train, "label": label_train}

    # add parameters to communicator
    comm.add_context_and_parameters((ctx, nn.get_parameters()))

    # Create validation graph
    test = True
    image_valid = nn.Variable((bs_valid, 3, 32, 32))
    pred_valid = prediction(image_valid, test)
    input_image_valid = {"image": image_valid}

    # Solvers
    solver = S.Adam()
    solver.set_parameters(nn.get_parameters())
    base_lr = args.learning_rate
    warmup_iter = int(1. * n_train_samples /
                      args.batch_size / n_devices) * args.warmup_epoch
    warmup_slope = base_lr * (n_devices - 1) / warmup_iter
    solver.set_learning_rate(base_lr)

    # Create monitor
    from nnabla.monitor import Monitor, MonitorSeries, MonitorTimeElapsed
    monitor = Monitor(args.monitor_path)
    monitor_loss = MonitorSeries("Training loss", monitor, interval=10)
    monitor_err = MonitorSeries("Training error", monitor, interval=10)
    monitor_time = MonitorTimeElapsed("Training time", monitor, interval=10)
    monitor_verr = MonitorSeries("Test error", monitor, interval=10)

    # Data Iterator
    rng = np.random.RandomState(device_id)
    tdata = data_iterator(args.batch_size, True, rng)
    vdata = data_iterator(args.batch_size, False)

    # Training-loop
    for i in range(int(args.max_iter / n_devices)):
        # Validation
        if device_id == 0:
            if i % int(n_train_samples / args.batch_size / n_devices) == 0:
                ve = 0.
                for j in range(args.val_iter):
                    image, label = vdata.next()
                    input_image_valid["image"].d = image
                    pred_valid.forward()
                    ve += categorical_error(pred_valid.d, label)
                ve /= args.val_iter
                monitor_verr.add(i * n_devices, ve)
            if i % int(args.model_save_interval / n_devices) == 0:
                nn.save_parameters(os.path.join(
                    args.model_save_path, 'params_%06d.h5' % i))

        # Forward/Zerograd/Backward
        image, label = tdata.next()
        input_image_train["image"].d = image
        input_image_train["label"].d = label
        loss_train.forward()
        solver.zero_grad()
        loss_train.backward()

        # Allreduce
        comm.allreduce(division=False, inplace=False)

        # Solvers update
        solver.update()

        # Linear Warmup
        if i <= warmup_iter:
            lr = base_lr + warmup_slope * i
            solver.set_learning_rate(lr)

        if device_id == 0:
            e = categorical_error(
                pred_train.d, input_image_train["label"].d)
            monitor_loss.add(i * n_devices, loss_train.d.copy())
            monitor_err.add(i * n_devices, e)
            monitor_time.add(i * n_devices)

    if device_id == 0:
        nn.save_parameters(os.path.join(
            args.model_save_path,
            'params_%06d.h5' % (args.max_iter / n_devices)))


if __name__ == '__main__':
    """
    Call this script with `mpirun` or `mpiexec`

    $ mpirun -n 4 python multi_device_multi_process.py --context "cuda.cudnn" -bs 64

    """
    train()
