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
import nnabla.solver as S
from nnabla.logger import logger

import numpy as np
import time
import os


def split_dataset(x, t, n_labeled, n_class):
    """
    Function for spliting a labeled dataset to labeled and unlabeled dataset
    """

    # Create datasets regarding to each class
    xs_l = [0] * n_class
    xs_u = [0] * n_class
    ts_l = [0] * n_class
    ts_u = [0] * n_class
    # To get designated number of labels
    n_labels_per_class = n_labeled / n_class
    for i in range(n_class):

        # i the class datum
        xi = x[(i == t)[:, 0]]
        ti = t[(i == t)[:, 0]]

        # shuffle
        idx = np.random.permutation(xi.shape[0])

        # former datum are used for labeled set(xs_l, ys_l)
        xs_l[i] = xi[idx[:n_labels_per_class], :]
        ts_l[i] = ti[idx[:n_labels_per_class], :]

        # latter datum are used for unlabeled set(xs_u, ys_u)
        xs_u[i] = xi[idx[n_labels_per_class:], :]
        ts_u[i] = ti[idx[n_labels_per_class:], :]

    # connect datasets
    x_l = np.vstack(xs_l)
    t_l = np.vstack(ts_l)
    x_u = np.vstack(xs_u)
    t_u = np.vstack(ts_u)

    return x_l, t_l, x_u, t_u


class DataIterator(object):
    """
    Data iterator for creating minibatches

    """

    def __init__(self, batch_size, xs,
                 shuffle=True, rng=None, preprocess=None):
        """
        Initialization

        Args:
            batch_size: size of minibatch
            xs: datasets list of array, like [datum, labels]
            shuffle: True/False
            rng: random state
            preprocess: preprocess function

        """
        if rng is None:
            rng = np.random.RandomState(313)
        if not isinstance(rng, np.random.RandomState):
            rng = np.random.RandomState(rng)
        self.rng = rng
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.xs = xs
        self.preprocess = preprocess
        self._reset()
        self.current = 0

    def next(self, batch_size=None):
        """
        Creating minibatch

        Args:
            batch_size: size of minibatch

        Returns:
            list of N-D array
        """

        if batch_size is None:
            batch_size = self.batch_size

        # Calculate end_index of minibatch
        end = min(self.current + batch_size, self.idxs.size)

        # Calculate batch_size in this epoch
        actual_batch_size = end - self.current

        # Temporal minibatch
        x = [xm[self.idxs[self.current:end]] for xm in self.xs]

        # Temporal index
        self.current = end

        # If index reachs to the end then reset index and data shuffle
        if self.current == self.idxs.size:
            self._reset()

        # If temporal batchsize is smaller than batch_size, then append datum
        if actual_batch_size < batch_size:
            a = self.next(batch_size - actual_batch_size)
            x = [np.vstack([xm, am]) for xm, am in zip(x, a)]

        # If preprocess is not None then process the pre-process
        if self.preprocess is None:
            return x
        else:
            return self.preprocess(x)

    def _reset(self):
        # If shuffle is True then do dataset shuffle
        if self.shuffle:
            self.idxs = self.rng.permutation(self.xs[0].shape[0])
        else:
            self.idxs = np.arange(self.xs[0].shape[0])
        self.current = 0


def categorical_error(p, t):
    """
    Calculate error rate

    Args:
        p: logit (float values of log(p(y|x)))
        t: label (int)

    Returns:
        error rate
    """
    k = p.argmax(1)
    return (k != t.flat).mean()


def calc_validation_error(di_v, xv, tv, pv, val_iter):
    """
    Calculate validation error rate

    Args:
        di_v; validation dataset
        xv: N-D array
        tv: N array
        pv: N-D array
        val_iter: numver of iteration

    Returns:
        error rate
    """
    ve = 0.0
    for j in range(val_iter):
        xv.d, tv.d = di_v.next()
        pv.forward(clear_buffer=True)
        ve += categorical_error(pv.d, tv.d)
    return ve / val_iter


def mlp_net(x, n_h, n_y, test=False):
    """
    Function for building multi-layer-perceptron with batch_normalization

    Args:
        x(`~nnabla.Variable`): N-D array 
        n_h(int): number of units in an intermediate layer
        n_y(int): number of classes
        test: operation type train=True, test=False

    Returns:
        ~nnabla.Variable: log(p(y|x))
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


def vat(x, r, eps, predict, distance):
    """
    Function for calculate LDS Loss, e.g. KL(p(y|x)||KL(p(y|x+n)

    Args:
        x(`~nnabla.Variable`): N-D array 
        r(`~nnabla.Variable`): N-D array of randn/grad
        eps(`~nnabla.Variable`): Scaling factor, xi for power iteration, epsilon for loss 
        predict: pointer of feed-forward-net building function
        distance: pointer of distance function e.g. KL(p(y|x)||KL(p(y|x+n)

    Returns:
        ~nnabla.Variable: LDS loss (KL(p(y|x)||KL(p(y|x+n))
    """
    # Calculate log(p(y|x))
    y = predict(x)

    # For stoping the backprop from this path.
    y1 = y.unlinked()

    # Calculate log(p(y|x+n))
    y2 = predict(x + eps * r)

    # Calculate kl(p(y|x)||p(y|x+n))
    loss = distance(y1, y2)
    loss = F.mean(loss)

    # Returns loss and y
    # y is returned for avoiding duplicated calculation
    return loss, y


def get_direction(d):
    """
    Vector normalization to get vector direction
    """
    shape = d.shape
    d = d.reshape((shape[0], np.prod(shape[1:])))
    d = d / np.sqrt(np.sum(d**2, axis=1)).reshape((shape[0], 1))
    d = d.reshape(shape).astype(np.float32)
    return d


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
    parser.add_argument("--device-id", "-d", type=int, default=0)
    parser.add_argument("--model-save-path", "-o",
                        type=str, default="tmp.monitor.vat")
    parser.add_argument('--context', '-c', type=str,
                        default=None, help="Extension path. ex) cpu, cuda.cudnn.")
    return parser.parse_args()


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
    *       Calculate Cross Entropy Loss 
    *     by Unlabeled Data
    *       Estimate Adversarial Direction
    *       Calculate LDS Loss
    """

    args = get_args()

    # Get context.
    from nnabla.contrib.context import extension_context
    extension_module = args.context
    if args.context is None:
        extension_module = 'cpu'
    logger.info("Running in %s" % extension_module)
    ctx = extension_context(extension_module, device_id=args.device_id)
    nn.set_default_context(ctx)

    shape_x = (1, 28, 28)
    n_h = args.n_units
    n_y = args.n_class

    # Load MNist Dataset
    from mnist_data import MnistDataSource
    with MnistDataSource(train=True) as d:
        x_t = d.images
        t_t = d.labels
    with MnistDataSource(train=False) as d:
        x_v = d.images
        t_v = d.labels
    x_t = np.array(x_t / 256.0).astype(np.float32)
    x_t, t_t = x_t[:args.n_train], t_t[:args.n_train]
    x_v, t_v = x_v[:args.n_valid], t_v[:args.n_valid]

    # Create Semi-supervised Datasets
    x_l, t_l, x_u, _ = split_dataset(x_t, t_t, args.n_labeled, args.n_class)
    x_u = np.r_[x_l, x_u]
    x_v = np.array(x_v / 256.0).astype(np.float32)

    # Create DataIterators for datasets of labeled, unlabeled and validation
    di_l = DataIterator(args.batchsize_l, [x_l, t_l])
    di_u = DataIterator(args.batchsize_u, [x_u])
    di_v = DataIterator(args.batchsize_v, [x_v, t_v])

    # Create networks
    # feed-forward-net building function
    def forward(x, test=False):
        return mlp_net(x, n_h, n_y, test)

    # Net for learning labeled data
    xl = nn.Variable((args.batchsize_l,) + shape_x, need_grad=False)
    hl = forward(xl, test=False)
    tl = nn.Variable((args.batchsize_l, 1), need_grad=False)
    loss_l = F.mean(F.softmax_cross_entropy(hl, tl))

    # Net for learning unlabeled data
    xu = nn.Variable((args.batchsize_u,) + shape_x, need_grad=False)
    r = nn.Variable((args.batchsize_u,) + shape_x, need_grad=True)
    eps = nn.Variable((args.batchsize_u,) + shape_x, need_grad=False)
    loss_u, yu = vat(xu, r, eps, forward, distance)

    # Net for evaluating valiation data
    xv = nn.Variable((args.batchsize_v,) + shape_x, need_grad=False)
    hv = forward(xv, test=True)
    tv = nn.Variable((args.batchsize_v, 1), need_grad=False)

    # Create solver
    solver = S.Adam(args.learning_rate)
    solver.set_parameters(nn.get_parameters())

    # Monitor trainig and validation stats.
    import nnabla.monitor as M
    monitor = M.Monitor(args.model_save_path)
    monitor_verr = M.MonitorSeries("Test error", monitor, interval=240)
    monitor_time = M.MonitorTimeElapsed("Elapsed time", monitor, interval=240)

    # Training Loop.
    t0 = time.time()

    for i in range(args.max_iter):

        # Validation Test
        if i % args.val_interval == 0:
            n_error = calc_validation_error(
                di_v, xv, tv, hv, args.val_iter)
            monitor_verr.add(i, n_error)

        #################################
        ## Training by Labeled Data #####
        #################################

        # input minibatch of labeled data into variables
        xl.d, tl.d = di_l.next()

        # initialize gradients
        solver.zero_grad()

        # forward, backward and update
        loss_l.forward(clear_no_need_grad=True)
        loss_l.backward(clear_buffer=True)
        solver.weight_decay(args.weight_decay)
        solver.update()

        #################################
        ## Training by Unlabeled Data ###
        #################################

        # input minibatch of unlabeled data into variables
        xu.d, = di_u.next()

        ##### Calculate Adversarial Noise #####

        # Sample random noise
        n = np.random.normal(size=xu.shape).astype(np.float32)

        # Normalize noise vector and input to variable
        r.d = get_direction(n)

        # Set xi, the power-method scaling parameter.
        eps.data.fill(args.xi_for_vat)

        # Calculate y without noise, only once.
        yu.forward(clear_buffer=True)

        # Do power method iteration
        for k in range(args.n_iter_for_power_method):
            # Initialize gradient to receive value
            r.grad.zero()

            # forward, backward, without update
            loss_u.forward(clear_no_need_grad=True)
            loss_u.backward(clear_buffer=True)

            # Normalize gradinet vector and input to variable
            r.d = get_direction(r.g)

        ##### Calculate loss for unlabeled data #####

        # Clear remained gradients
        solver.zero_grad()

        # Set epsilon, the adversarial noise scaling parameter.
        eps.data.fill(args.eps_for_vat)

        # forward, backward and update
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
    valid_error = calc_validation_error(di_v, xv, tv, hv, args.val_iter)
    monitor_verr.add(i, valid_error)
    monitor_time.add(i)

    # Save the model.
    nn.save_parameters(
        os.path.join(args.model_save_path, 'params_%06d.h5' % args.max_iter))


if __name__ == '__main__':
    main()
