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

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solver as S
from nnabla.logger import logger
import nnabla.utils.save as save
from nnabla.utils.data_iterator import data_iterator_simple
from _checkpoint_nnp_util import save_nnp

import numpy as np
import time
import os


def get_args():
    """
    Get command line arguments.
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_class", "-nc", type=int, default=10)
    parser.add_argument("--n_train", "-nt", type=int, default=60000)
    parser.add_argument("--n_valid", "-nv", type=int, default=10000)
    parser.add_argument("--n_labeled", "-nl", type=int, default=100)
    parser.add_argument("--n_units", "-nu", type=int, default=1200)
    parser.add_argument("--batchsize_l", "-bl", type=int, default=100)
    parser.add_argument("--batchsize_u", "-bu", type=int, default=250)
    parser.add_argument("--batchsize_v", "-bv", type=int, default=100)
    parser.add_argument("--learning-rate", "-l", type=float, default=2e-3)
    parser.add_argument("--max-iter", "-i", type=int, default=24000)
    parser.add_argument("--val-interval", "-v", type=int, default=240)
    parser.add_argument("--val-iter", "-j", type=int, default=100)
    parser.add_argument("--iter-per-epoch", "-e", type=int, default=240)
    parser.add_argument("--weight-decay", "-w", type=float, default=0)
    parser.add_argument("--learning-rate-decay",
                        "-ld", type=float, default=0.9)
    parser.add_argument("--n-iter-for-power-method",
                        "-np", type=int, default=1)
    parser.add_argument("--xi-for-vat", "-er", type=float, default=10.0)
    parser.add_argument("--eps-for-vat", "-el", type=float, default=1.5)
    parser.add_argument("--device-id", "-d", type=str, default='0',
                        help='Device ID the training run on. This is only valid if you specify `-c cudnn`.')
    parser.add_argument("--type-config", "-t", type=str, default='float',
                        help='Type of computation. e.g. "float", "half".')
    parser.add_argument("--model-save-path", "-o",
                        type=str, default="tmp.monitor.vat")
    parser.add_argument('--context', '-c', type=str,
                        default="cpu", help="Extension path. ex) cpu, cudnn.")
    return parser.parse_args()


def mlp_net(x, n_h, n_y, test=False):
    """
    Function for building multi-layer-perceptron with batch_normalization

    Args:
        x(`~nnabla.Variable`): N-D array
        n_h(int): number of units in an intermediate layer
        n_y(int): number of classes
        test: operation type train=True, test=False

    Returns:
        ~nnabla.Variable: h
    """

    h = x
    with nn.parameter_scope("fc1"):
        h = F.relu(PF.batch_normalization(
            PF.affine(h, n_h), batch_stat=not test), inplace=True)
    with nn.parameter_scope("fc2"):
        h = F.relu(PF.batch_normalization(
            PF.affine(h, n_h), batch_stat=not test), inplace=True)
    with nn.parameter_scope("fc3"):
        h = PF.affine(h, n_y)
    return h


def distance(y0, y1):
    """
    Distance function is Kullback-Leibler Divergence for categorical distribution
    """
    return F.kl_multinomial(F.softmax(y0), F.softmax(y1))


def calc_validation_error(di_v, xv, tv, err, val_iter):
    """
    Calculate validation error rate

    Args:
        di_v; validation dataset
        xv: variable for input
        tv: variable for label
        err: variable for error estimation
        val_iter: number of iteration

    Returns:
        error rate
    """
    ve = 0.0
    for j in range(val_iter):
        xv.d, tv.d = di_v.next()
        xv.d = xv.d / 255
        err.forward(clear_buffer=True)
        ve += err.d
    return ve / val_iter


def main():
    """
    Main script.

    Steps:
    * Get and set context.
    * Load Dataset
    * Initialize DataIterator.
    * Create Networks
    *   Net for Labeled Data
    *   Net for Unlabeled Data
    *   Net for Test Data
    * Create Solver.
    * Training Loop.
    *   Test
    *   Training
    *     by Labeled Data
    *       Calculate Supervised Loss
    *     by Unlabeled Data
    *       Calculate Virtual Adversarial Noise
    *       Calculate Unsupervised Loss
    """

    args = get_args()

    # Get context.
    from nnabla.ext_utils import get_extension_context
    logger.info("Running in %s" % args.context)
    ctx = get_extension_context(
        args.context, device_id=args.device_id, type_config=args.type_config)
    nn.set_default_context(ctx)

    shape_x = (1, 28, 28)
    n_h = args.n_units
    n_y = args.n_class

    # Load MNIST Dataset
    from mnist_data import load_mnist, data_iterator_mnist
    images, labels = load_mnist(train=True)
    rng = np.random.RandomState(706)
    inds = rng.permutation(len(images))

    def feed_labeled(i):
        j = inds[i]
        return images[j], labels[j]

    def feed_unlabeled(i):
        j = inds[i]
        return images[j], labels[j]
    di_l = data_iterator_simple(feed_labeled, args.n_labeled,
                                args.batchsize_l, shuffle=True, rng=rng, with_file_cache=False)
    di_u = data_iterator_simple(feed_unlabeled, args.n_train,
                                args.batchsize_u, shuffle=True, rng=rng, with_file_cache=False)
    di_v = data_iterator_mnist(args.batchsize_v, train=False)

    # Create networks
    # feed-forward-net building function
    def forward(x, test=False):
        return mlp_net(x, n_h, n_y, test)

    # Net for learning labeled data
    xl = nn.Variable((args.batchsize_l,) + shape_x, need_grad=False)
    yl = forward(xl, test=False)
    tl = nn.Variable((args.batchsize_l, 1), need_grad=False)
    loss_l = F.mean(F.softmax_cross_entropy(yl, tl))

    # Net for learning unlabeled data
    xu = nn.Variable((args.batchsize_u,) + shape_x, need_grad=False)
    yu = forward(xu, test=False)
    y1 = yu.get_unlinked_variable()
    y1.need_grad = False

    noise = nn.Variable((args.batchsize_u,) + shape_x, need_grad=True)
    r = noise / (F.sum(noise ** 2, [1, 2, 3], keepdims=True)) ** 0.5
    r.persistent = True
    y2 = forward(xu + args.xi_for_vat * r, test=False)
    y3 = forward(xu + args.eps_for_vat * r, test=False)
    loss_k = F.mean(distance(y1, y2))
    loss_u = F.mean(distance(y1, y3))

    # Net for evaluating validation data
    xv = nn.Variable((args.batchsize_v,) + shape_x, need_grad=False)
    hv = forward(xv, test=True)
    tv = nn.Variable((args.batchsize_v, 1), need_grad=False)
    err = F.mean(F.top_n_error(hv, tv, n=1))

    # Create solver
    solver = S.Adam(args.learning_rate)
    solver.set_parameters(nn.get_parameters())

    # Monitor training and validation stats.
    import nnabla.monitor as M
    monitor = M.Monitor(args.model_save_path)
    monitor_verr = M.MonitorSeries("Test error", monitor, interval=240)
    monitor_time = M.MonitorTimeElapsed("Elapsed time", monitor, interval=240)

    contents = save_nnp({'x': xv}, {'y': hv}, 1)
    save.save(os.path.join(args.model_save_path,
                           'result_epoch0.nnp'), contents)

    # Training Loop.
    t0 = time.time()

    for i in range(args.max_iter):

        # Validation Test
        if i % args.val_interval == 0:
            valid_error = calc_validation_error(
                di_v, xv, tv, err, args.val_iter)
            monitor_verr.add(i, valid_error)

        #################################
        ## Training by Labeled Data #####
        #################################

        # forward, backward and update
        xl.d, tl.d = di_l.next()
        xl.d = xl.d / 255
        solver.zero_grad()
        loss_l.forward(clear_no_need_grad=True)
        loss_l.backward(clear_buffer=True)
        solver.weight_decay(args.weight_decay)
        solver.update()

        #################################
        ## Training by Unlabeled Data ###
        #################################

        # Calculate y without noise, only once.
        xu.d, _ = di_u.next()
        xu.d = xu.d / 255
        yu.forward(clear_buffer=True)

        ##### Calculate Adversarial Noise #####
        # Do power method iteration
        noise.d = np.random.normal(size=xu.shape).astype(np.float32)
        for k in range(args.n_iter_for_power_method):
            r.grad.zero()
            loss_k.forward(clear_no_need_grad=True)
            loss_k.backward(clear_buffer=True)
            noise.data.copy_from(r.grad)

        ##### Calculate loss for unlabeled data #####
        # forward, backward and update
        solver.zero_grad()
        loss_u.forward(clear_no_need_grad=True)
        loss_u.backward(clear_buffer=True)
        solver.weight_decay(args.weight_decay)
        solver.update()

        ##### Learning rate update #####
        if i % args.iter_per_epoch == 0:
            solver.set_learning_rate(
                solver.learning_rate() * args.learning_rate_decay)
        monitor_time.add(i)

    # Evaluate the final model by the error rate with validation dataset
    valid_error = calc_validation_error(di_v, xv, tv, err, args.val_iter)
    monitor_verr.add(i, valid_error)
    monitor_time.add(i)

    # Save the model.
    parameter_file = os.path.join(
        args.model_save_path, 'params_%06d.h5' % args.max_iter)
    nn.save_parameters(parameter_file)

    contents = save_nnp({'x': xv}, {'y': hv}, 1)
    save.save(os.path.join(args.model_save_path,
                           'result.nnp'), contents)


if __name__ == '__main__':
    main()
