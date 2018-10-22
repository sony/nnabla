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


import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.monitor as M
import nnabla.solver as S
from nnabla.logger import logger

from mnist_data import data_iterator_mnist
from args import get_args

import numpy as np
import time
import os


def vae(x, shape_z, test=False):
    """
    Function for calculate Elbo(evidence lowerbound) loss.
    This sample is a Bernoulli generator version.

    Args:
        x(`~nnabla.Variable`): N-D array
        shape_z(tuple of int): size of z
        test : True=train, False=test

    Returns:
        ~nnabla.Variable: Elbo loss

    """

    #############################################
    # Encoder of 2 fully connected layers       #
    #############################################

    # Normalize input
    xa = x / 256.
    batch_size = x.shape[0]

    # 2 fully connected layers, and Elu replaced from original Softplus.
    h = F.elu(PF.affine(xa, (500,), name='fc1'))
    h = F.elu(PF.affine(h, (500,), name='fc2'))

    # The outputs are the parameters of Gauss probability density.
    mu = PF.affine(h, shape_z, name='fc_mu')
    logvar = PF.affine(h, shape_z, name='fc_logvar')
    sigma = F.exp(0.5 * logvar)

    # The prior variable and the reparameterization trick
    if not test:
        # training with reparameterization trick
        epsilon = F.randn(mu=0, sigma=1, shape=(batch_size,) + shape_z)
        z = mu + sigma * epsilon
    else:
        # test without randomness
        z = mu

    #############################################
    # Decoder of 2 fully connected layers       #
    #############################################

    # 2 fully connected layers, and Elu replaced from original Softplus.
    h = F.elu(PF.affine(z, (500,), name='fc3'))
    h = F.elu(PF.affine(h, (500,), name='fc4'))

    # The outputs are the parameters of Bernoulli probabilities for each pixel.
    prob = PF.affine(h, (1, 28, 28), name='fc5')

    #############################################
    # Elbo components and loss objective        #
    #############################################

    # Binarized input
    xb = F.greater_equal_scalar(xa, 0.5)

    # E_q(z|x)[log(q(z|x))]
    # without some constant terms that will canceled after summation of loss
    logqz = 0.5 * F.sum(1.0 + logvar, axis=1)

    # E_q(z|x)[log(p(z))]
    # without some constant terms that will canceled after summation of loss
    logpz = 0.5 * F.sum(mu * mu + sigma * sigma, axis=1)

    # E_q(z|x)[log(p(x|z))]
    logpx = F.sum(F.sigmoid_cross_entropy(prob, xb), axis=(1, 2, 3))

    # Vae loss, the negative evidence lowerbound
    loss = F.mean(logpx + logpz - logqz)

    return loss


def main():
    """
    Main script.
    Steps:
    * Setup calculation environment
    * Initialize data iterator.
    * Create Networks
    * Create Solver.
    * Training Loop.
    *   Training
    *   Test
    * Save
    """

    # Set args
    args = get_args(monitor_path='tmp.monitor.vae', max_iter=60000,
                    model_save_path=None, learning_rate=3e-4, batch_size=100, weight_decay=0)

    # Get context.
    from nnabla.ext_utils import get_extension_context
    logger.info("Running in %s" % args.context)
    ctx = get_extension_context(
        args.context, device_id=args.device_id, type_config=args.type_config)
    nn.set_default_context(ctx)

    # Initialize data provider
    di_l = data_iterator_mnist(args.batch_size, True)
    di_t = data_iterator_mnist(args.batch_size, False)

    # Network
    shape_x = (1, 28, 28)
    shape_z = (50,)
    x = nn.Variable((args.batch_size,) + shape_x)
    loss_l = vae(x, shape_z, test=False)
    loss_t = vae(x, shape_z, test=True)

    # Create solver
    solver = S.Adam(args.learning_rate)
    solver.set_parameters(nn.get_parameters())

    # Monitors for training and validation
    monitor = M.Monitor(args.model_save_path)
    monitor_training_loss = M.MonitorSeries(
        "Training loss", monitor, interval=600)
    monitor_test_loss = M.MonitorSeries("Test loss", monitor, interval=600)
    monitor_time = M.MonitorTimeElapsed("Elapsed time", monitor, interval=600)

    # Training Loop.
    for i in range(args.max_iter):

        # Initialize gradients
        solver.zero_grad()

        # Forward, backward and update
        x.d, _ = di_l.next()
        loss_l.forward(clear_no_need_grad=True)
        loss_l.backward(clear_buffer=True)
        solver.weight_decay(args.weight_decay)
        solver.update()

        # Forward for test
        x.d, _ = di_t.next()
        loss_t.forward(clear_no_need_grad=True)

        # Monitor for logging
        monitor_training_loss.add(i, loss_l.d.copy())
        monitor_test_loss.add(i, loss_t.d.copy())
        monitor_time.add(i)

    # Save the model
    nn.save_parameters(
        os.path.join(args.model_save_path, 'params_%06d.h5' % args.max_iter))


if __name__ == '__main__':
    main()
