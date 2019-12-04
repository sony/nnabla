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
from _checkpoint_nnp_util import save_checkpoint, load_checkpoint, save_nnp

import os
import sys

import nnabla as nn
import nnabla.logger as logger
import nnabla.functions as F
import nnabla.solvers as S
import nnabla.utils.save as save

import model
from _mnist_data import data_iterator_mnist
from reconstruct_tweaked_capsules import model_tweak_digitscaps


def categorical_error(pred, label):
    """
    Compute categorical error given score vectors and labels as
    numpy.ndarray.
    """
    pred_label = pred.argmax(1)
    return (pred_label != label.flat).mean()


def get_args(monitor_path='tmp.monitor.capsnet'):
    """
    Get command line arguments.
    """
    import argparse
    import os
    description = "Capsule Net."
    parser = argparse.ArgumentParser(description)
    parser.add_argument("--batch-size", "-b", type=int, default=100)
    parser.add_argument("--learning-rate", "-l",
                        type=float, default=0.001)
    parser.add_argument("--monitor-path", "-m",
                        type=str, default=monitor_path,
                        help='Path monitoring logs saved.')
    parser.add_argument("--max-epochs", "-e", type=int, default=50,
                        help='Max epochs of training.')
    parser.add_argument('--context', '-c', type=str,
                        default='cudnn', help="Extension modules. ex) 'cpu', 'cudnn'.")
    parser.add_argument("--device-id", "-d", type=str, default='0',
                        help='Device ID the training run on. This is only valid if you specify `-c cudnn`.')
    parser.add_argument("--type-config", "-t", type=str, default='float',
                        help='Type of computation. e.g. "float", "half".')
    parser.add_argument("--disable-grad-dynamic-routing", "-g",
                        dest='grad_dynamic_routing',
                        action='store_false', default=True,
                        help='Disable gradient computation at dynamic routing.')
    parser.add_argument("--checkpoint", type=str,
                        help="path to the checkpoint")
    args = parser.parse_args()
    if not os.path.isdir(args.monitor_path):
        os.makedirs(args.monitor_path)
    return args


def train():
    '''
    Main script.
    '''
    args = get_args()

    from numpy.random import seed
    seed(0)

    # Get context.
    from nnabla.ext_utils import get_extension_context
    logger.info("Running in %s" % args.context)
    ctx = get_extension_context(
        args.context, device_id=args.device_id, type_config=args.type_config)
    nn.set_default_context(ctx)

    # TRAIN
    image = nn.Variable([args.batch_size, 1, 28, 28])
    label = nn.Variable([args.batch_size, 1])
    x = image / 255.0
    t_onehot = F.one_hot(label, (10,))
    with nn.parameter_scope("capsnet"):
        c1, pcaps, u_hat, caps, pred = model.capsule_net(
            x, test=False, aug=True, grad_dynamic_routing=args.grad_dynamic_routing)
    with nn.parameter_scope("capsnet_reconst"):
        recon = model.capsule_reconstruction(caps, t_onehot)
    loss_margin, loss_reconst, loss = model.capsule_loss(
        pred, t_onehot, recon, x)
    pred.persistent = True

    # TEST
    # Create input variables.
    vimage = nn.Variable([args.batch_size, 1, 28, 28])
    vlabel = nn.Variable([args.batch_size, 1])
    vx = vimage / 255.0
    with nn.parameter_scope("capsnet"):
        _, _, _, _, vpred = model.capsule_net(vx, test=True, aug=False)

    # Create Solver.
    solver = S.Adam(args.learning_rate)
    solver.set_parameters(nn.get_parameters())

    # Create monitor.
    from nnabla.monitor import Monitor, MonitorSeries, MonitorTimeElapsed
    train_iter = int(60000 / args.batch_size)
    val_iter = int(10000 / args.batch_size)
    logger.info("#Train: {} #Validation: {}".format(train_iter, val_iter))
    monitor = Monitor(args.monitor_path)
    monitor_loss = MonitorSeries("Training loss", monitor, interval=1)
    monitor_mloss = MonitorSeries("Training margin loss", monitor, interval=1)
    monitor_rloss = MonitorSeries(
        "Training reconstruction loss", monitor, interval=1)
    monitor_err = MonitorSeries("Training error", monitor, interval=1)
    monitor_time = MonitorTimeElapsed("Training time", monitor, interval=1)
    monitor_verr = MonitorSeries("Test error", monitor, interval=1)
    monitor_lr = MonitorSeries("Learning rate", monitor, interval=1)

    # To_save_nnp
    m_image, m_label, m_noise, m_recon = model_tweak_digitscaps(
        args.batch_size)
    contents = save_nnp({'x1': m_image, 'x2': m_label, 'x3': m_noise}, {
                          'y': m_recon}, args.batch_size)
    save.save(os.path.join(args.monitor_path,
                           'capsnet_epoch0_result.nnp'), contents)

    # Initialize DataIterator for MNIST.
    from numpy.random import RandomState
    data = data_iterator_mnist(args.batch_size, True, rng=RandomState(1223))
    vdata = data_iterator_mnist(args.batch_size, False)
    start_point = 0

    if args.checkpoint is not None:
        # load weights and solver state info from specified checkpoint file.
        start_point = load_checkpoint(args.checkpoint, solver)
    # Training loop.
    for e in range(start_point, args.max_epochs):

        # Learning rate decay
        learning_rate = solver.learning_rate()
        if e != 0:
            learning_rate *= 0.9
        solver.set_learning_rate(learning_rate)
        monitor_lr.add(e, learning_rate)

        # Training
        train_error = 0.0
        train_loss = 0.0
        train_mloss = 0.0
        train_rloss = 0.0
        for i in range(train_iter):
            image.d, label.d = data.next()
            solver.zero_grad()
            loss.forward(clear_no_need_grad=True)
            loss.backward(clear_buffer=True)
            solver.update()
            train_error += categorical_error(pred.d, label.d)
            train_loss += loss.d
            train_mloss += loss_margin.d
            train_rloss += loss_reconst.d
        train_error /= train_iter
        train_loss /= train_iter
        train_mloss /= train_iter
        train_rloss /= train_iter

        # Validation
        val_error = 0.0
        for j in range(val_iter):
            vimage.d, vlabel.d = vdata.next()
            vpred.forward(clear_buffer=True)
            val_error += categorical_error(vpred.d, vlabel.d)
        val_error /= val_iter

        # Monitor
        monitor_time.add(e)
        monitor_loss.add(e, train_loss)
        monitor_mloss.add(e, train_mloss)
        monitor_rloss.add(e, train_rloss)
        monitor_err.add(e, train_error)
        monitor_verr.add(e, val_error)
        save_checkpoint(args.monitor_path, e, solver)

    # To_save_nnp
    contents = save_nnp({'x1': m_image, 'x2': m_label, 'x3': m_noise}, {
                          'y': m_recon}, args.batch_size)
    save.save(os.path.join(args.monitor_path,
                           'capsnet_result.nnp'), contents)


if __name__ == '__main__':
    train()
