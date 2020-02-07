# Copyright (c) 2019 Sony Corporation. All Rights Reserved.
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
from data_cifar10 import *
import nnabla as nn
from nnabla.ext_utils import get_extension_context
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import numpy as np
import functools
from models import *
from MixedDataLearning import *


def single_image_augment(image):
    image = F.image_augmentation(image, contrast=1.0, angle=0.0, flip_lr=True)
    image = F.random_shift(image, shifts=(4, 4), border_mode="reflect")
    return image


def train():
    """
    Main script.

    Steps:

    * Parse command line arguments.
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
    n_valid_samples = 10000
    bs_valid = args.batch_size
    extension_module = args.context
    ctx = get_extension_context(
        extension_module, device_id=args.device_id, type_config=args.type_config)
    nn.set_default_context(ctx)

    # Dataset
    data_iterator = data_iterator_cifar10
    n_class = 10

    # Model architecture
    if args.net == "resnet18":
        prediction = functools.partial(
            resnet18_prediction, ncls=n_class, nmaps=64, act=F.relu)
    if args.net == "resnet34":
        prediction = functools.partial(
            resnet34_prediction, ncls=n_class, nmaps=64, act=F.relu)

    # Create training graphs
    test = False
    if args.mixtype == "mixup":
        mdl = MixupLearning(args.batch_size, alpha=args.alpha)
    elif args.mixtype == "cutmix":
        mdl = CutmixLearning((args.batch_size, 3, 32, 32),
                             alpha=args.alpha, cutmix_prob=1.0)
    elif args.mixtype == "vhmixup":
        mdl = VHMixupLearning((args.batch_size, 3, 32, 32), alpha=args.alpha)
    else:
        print("[ERROR] Unknown mixtype: " + args.mixtype)
        return
    image_train = nn.Variable((args.batch_size, 3, 32, 32))
    label_train = nn.Variable((args.batch_size, 1))
    mix_image, mix_label = mdl.mix_data(single_image_augment(
        image_train), F.one_hot(label_train, (n_class, )))
    pred_train = prediction(mix_image, test)
    loss_train = mdl.loss(pred_train, mix_label)
    input_train = {"image": image_train, "label": label_train}

    # Create validation graph
    test = True
    image_valid = nn.Variable((bs_valid, 3, 32, 32))
    pred_valid = prediction(image_valid, test)
    input_valid = {"image": image_valid}

    # Solvers
    if args.solver == "Adam":
        solver = S.Adam()
    elif args.solver == "Momentum":
        solver = S.Momentum(lr=args.learning_rate)
    solver.set_parameters(nn.get_parameters())

    # Create monitor
    from nnabla.monitor import Monitor, MonitorSeries, MonitorTimeElapsed
    monitor = Monitor(args.save_path)
    monitor_loss = MonitorSeries("Training loss", monitor, interval=10)
    monitor_time = MonitorTimeElapsed("Training time", monitor, interval=10)
    monitor_verr = MonitorSeries("Test error", monitor, interval=1)

    # Data Iterator
    tdata = data_iterator(args.batch_size, True)
    vdata = data_iterator(args.batch_size, False)

    print("Size of the training data: %d " % tdata.size)
    # Training-loop
    for i in range(args.max_iter):
        # Forward/Zerograd/Backward
        image, label = tdata.next()
        input_train["image"].d = image
        input_train["label"].d = label
        mdl.set_mix_ratio()
        loss_train.forward()
        solver.zero_grad()
        loss_train.backward()

        # Model update by solver
        if args.solver == "Momentum":
            if i == args.max_iter / 2:
                solver.set_learning_rate(args.learning_rate / 10.0)
            if i == args.max_iter / 4 * 3:
                solver.set_learning_rate(args.learning_rate / 10.0**2)
        solver.update()

        # Validation
        if (i+1) % args.val_interval == 0 or i == 0:
            ve = 0.
            vdata._reset()
            vdata_pred = np.zeros((n_valid_samples, n_class))
            vdata_label = np.zeros((n_valid_samples, 1), dtype=np.int32)
            for j in range(0, n_valid_samples, args.batch_size):
                image, label = vdata.next()
                input_valid["image"].d = image
                pred_valid.forward()
                vdata_pred[j:min(j+args.batch_size, n_valid_samples)
                           ] = pred_valid.d[:min(args.batch_size, n_valid_samples-j)]
                vdata_label[j:min(j+args.batch_size, n_valid_samples)
                            ] = label[:min(args.batch_size, n_valid_samples-j)]
            ve = categorical_error(vdata_pred, vdata_label)
            monitor_verr.add(i+1, ve)

        if int((i+1) % args.model_save_interval) == 0:
            nn.save_parameters(os.path.join(
                args.save_path, 'params_%06d.h5' % (i+1)))

        # Monitering
        monitor_loss.add(i+1, loss_train.d.copy())
        monitor_time.add(i+1)

    nn.save_parameters(os.path.join(args.save_path,
                                    'params_%06d.h5' % (args.max_iter)))


if __name__ == '__main__':
    """
    $ python classification.py --context "cudnn" -b 64

    """
    train()
