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
import nnabla.logger as logger
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solver as S
import nnabla.utils.save as save

from args import get_args
from mnist_data import data_iterator_mnist

import os


def categorical_error(pred, label):
    """
    Compute categorical error given score vectors and labels as
    numpy.ndarray.
    """
    pred_label = pred.argmax(1)
    return (pred_label != label.flat).mean()


# Binary Connect Model
def mnist_binary_connect_lenet_prediction(image, test=False):
    """
    Construct LeNet for MNIST (BinaryNet version).
    """
    with nn.parameter_scope("conv1"):
        c1 = PF.binary_connect_convolution(image, 16, (5, 5))
        c1 = PF.batch_normalization(c1, batch_stat=not test)
        c1 = F.elu(F.average_pooling(c1, (2, 2)))
    with nn.parameter_scope("conv2"):
        c2 = PF.binary_connect_convolution(c1, 16, (5, 5))
        c2 = PF.batch_normalization(c2, batch_stat=not test)
        c2 = F.elu(F.average_pooling(c2, (2, 2)))
    with nn.parameter_scope("fc3"):
        c3 = PF.binary_connect_affine(c2, 50)
        c3 = PF.batch_normalization(c3, batch_stat=not test)
        c3 = F.elu(c3)
    with nn.parameter_scope("fc4"):
        c4 = PF.binary_connect_affine(c3, 10)
        c4 = PF.batch_normalization(c4, batch_stat=not test)
    return c4


def mnist_binary_connect_resnet_prediction(image, test=False):
    """
    Construct ResNet for MNIST (BinaryNet version).
    """
    def bn(x):
        return PF.batch_normalization(x, batch_stat=not test)

    def res_unit(x, scope):
        C = x.shape[1]
        with nn.parameter_scope(scope):
            with nn.parameter_scope('conv1'):
                h = F.elu(bn(PF.binary_connect_convolution(
                    x, C / 2, (1, 1), with_bias=False)))
            with nn.parameter_scope('conv2'):
                h = F.elu(
                    bn(PF.binary_connect_convolution(h, C / 2, (3, 3), pad=(1, 1), with_bias=False)))
            with nn.parameter_scope('conv3'):
                h = bn(PF.binary_connect_convolution(
                    h, C, (1, 1), with_bias=False))
        return F.elu(x + h)
    # Conv1 --> 64 x 32 x 32
    with nn.parameter_scope("conv1"):
        c1 = F.elu(
            bn(PF.binary_connect_convolution(image, 64, (3, 3), pad=(3, 3), with_bias=False)))
    # Conv2 --> 64 x 16 x 16
    c2 = F.max_pooling(res_unit(c1, "conv2"), (2, 2))
    # Conv3 --> 64 x 8 x 8
    c3 = F.max_pooling(res_unit(c2, "conv3"), (2, 2))
    # Conv4 --> 64 x 8 x 8
    c4 = res_unit(c3, "conv4")
    # Conv5 --> 64 x 4 x 4
    c5 = F.max_pooling(res_unit(c4, "conv5"), (2, 2))
    # Conv5 --> 64 x 4 x 4
    c6 = res_unit(c5, "conv6")
    pl = F.average_pooling(c6, (4, 4))
    with nn.parameter_scope("classifier"):
        y = bn(PF.binary_connect_affine(pl, 10))
    return y

# Binary Net Model


def mnist_binary_net_lenet_prediction(image, test=False):
    """
    Construct LeNet for MNIST (BinaryNet version).
    """
    with nn.parameter_scope("conv1"):
        c1 = PF.binary_connect_convolution(image, 16, (5, 5))
        c1 = PF.batch_normalization(c1, batch_stat=not test)
        c1 = F.binary_tanh(F.average_pooling(c1, (2, 2)))
    with nn.parameter_scope("conv2"):
        c2 = PF.binary_connect_convolution(c1, 16, (5, 5))
        c2 = PF.batch_normalization(c2, batch_stat=not test)
        c2 = F.binary_tanh(F.average_pooling(c2, (2, 2)))
    with nn.parameter_scope("fc3"):
        c3 = PF.binary_connect_affine(c2, 50)
        c3 = PF.batch_normalization(c3, batch_stat=not test)
        c3 = F.binary_tanh(c3)
    with nn.parameter_scope("fc4"):
        c4 = PF.binary_connect_affine(c3, 10)
        c4 = PF.batch_normalization(c4, batch_stat=not test)
    return c4


def mnist_binary_net_resnet_prediction(image, test=False):
    """
    Construct ResNet for MNIST (BinaryNet version).
    """
    def bn(x):
        return PF.batch_normalization(x, batch_stat=not test)

    def res_unit(x, scope):
        C = x.shape[1]
        with nn.parameter_scope(scope):
            with nn.parameter_scope('conv1'):
                h = F.binary_tanh(bn(PF.binary_connect_convolution(
                    x, C / 2, (1, 1), with_bias=False)))
            with nn.parameter_scope('conv2'):
                h = F.binary_tanh(
                    bn(PF.binary_connect_convolution(h, C / 2, (3, 3), pad=(1, 1), with_bias=False)))
            with nn.parameter_scope('conv3'):
                h = bn(PF.binary_connect_convolution(
                    h, C, (1, 1), with_bias=False))
        return F.binary_tanh(x + h)
    # Conv1 --> 64 x 32 x 32
    with nn.parameter_scope("conv1"):
        c1 = F.binary_tanh(
            bn(PF.binary_connect_convolution(image, 64, (3, 3), pad=(3, 3), with_bias=False)))
    # Conv2 --> 64 x 16 x 16
    c2 = F.max_pooling(res_unit(c1, "conv2"), (2, 2))
    # Conv3 --> 64 x 8 x 8
    c3 = F.max_pooling(res_unit(c2, "conv3"), (2, 2))
    # Conv4 --> 64 x 8 x 8
    c4 = res_unit(c3, "conv4")
    # Conv5 --> 64 x 4 x 4
    c5 = F.max_pooling(res_unit(c4, "conv5"), (2, 2))
    # Conv5 --> 64 x 4 x 4
    c6 = res_unit(c5, "conv6")
    pl = F.average_pooling(c6, (4, 4))
    with nn.parameter_scope("classifier"):
        y = bn(PF.binary_connect_affine(pl, 10))
    return y

# Binary Weight Model


def mnist_binary_weight_lenet_prediction(image, test=False):
    """
    Construct LeNet for MNIST (Binary Weight Network version).
    """
    with nn.parameter_scope("conv1"):
        c1 = PF.binary_weight_convolution(image, 16, (5, 5))
        c1 = F.elu(F.average_pooling(c1, (2, 2)))
    with nn.parameter_scope("conv2"):
        c2 = PF.binary_weight_convolution(c1, 16, (5, 5))
        c2 = F.elu(F.average_pooling(c2, (2, 2)))
    with nn.parameter_scope("fc3"):
        c3 = F.elu(PF.binary_weight_affine(c2, 50))
    with nn.parameter_scope("fc4"):
        c4 = PF.binary_weight_affine(c3, 10)
    return c4


def mnist_binary_weight_resnet_prediction(image, test=False):
    """
    Construct ResNet for MNIST (Binary Weight Network version).
    """
    def bn(x):
        return PF.batch_normalization(x, batch_stat=not test)

    def res_unit(x, scope):
        C = x.shape[1]
        with nn.parameter_scope(scope):
            with nn.parameter_scope('conv1'):
                h = F.elu(bn(PF.binary_weight_convolution(
                    x, C / 2, (1, 1), with_bias=False)))
            with nn.parameter_scope('conv2'):
                h = F.elu(
                    bn(PF.binary_weight_convolution(h, C / 2, (3, 3), pad=(1, 1), with_bias=False)))
            with nn.parameter_scope('conv3'):
                h = bn(PF.binary_weight_convolution(
                    h, C, (1, 1), with_bias=False))
        return F.elu(x + h)
    # Conv1 --> 64 x 32 x 32
    with nn.parameter_scope("conv1"):
        c1 = F.elu(
            bn(PF.binary_weight_convolution(image, 64, (3, 3), pad=(3, 3), with_bias=False)))
    # Conv2 --> 64 x 16 x 16
    c2 = F.max_pooling(res_unit(c1, "conv2"), (2, 2))
    # Conv3 --> 64 x 8 x 8
    c3 = F.max_pooling(res_unit(c2, "conv3"), (2, 2))
    # Conv4 --> 64 x 8 x 8
    c4 = res_unit(c3, "conv4")
    # Conv5 --> 64 x 4 x 4
    c5 = F.max_pooling(res_unit(c4, "conv5"), (2, 2))
    # Conv5 --> 64 x 4 x 4
    c6 = res_unit(c5, "conv6")
    pl = F.average_pooling(c6, (4, 4))
    with nn.parameter_scope("classifier"):
        y = PF.binary_weight_affine(pl, 10)
    return y


def train():
    """
    Main script.

    Steps:

    * Parse command line arguments.
    * Specify a context for computation.
    * Initialize DataIterator for MNIST.
    * Construct a computation graph for training and validation.
    * Initialize a solver and set parameter variables to it.
    * Create monitor instances for saving and displaying training stats.
    * Training loop
      * Computate error rate for validation data (periodically)
      * Get a next minibatch.
      * Set parameter gradients zero
      * Execute forwardprop on the training graph.
      * Execute backprop.
      * Solver updates parameters by using gradients computed by backprop.
      * Compute training error
    """
    args = get_args(monitor_path='tmp.monitor.bnn')

    # Get context.
    from nnabla.contrib.context import extension_context
    extension_module = args.context
    if args.context is None:
        extension_module = 'cpu'
    logger.info("Running in %s" % extension_module)
    ctx = extension_context(extension_module, device_id=args.device_id)
    nn.set_default_context(ctx)

    # Initialize DataIterator for MNIST.
    data = data_iterator_mnist(args.batch_size, True)
    vdata = data_iterator_mnist(args.batch_size, False)

    # Create CNN network for both training and testing.
    mnist_cnn_prediction = mnist_binary_connect_lenet_prediction
    if args.net == 'bincon':
        mnist_cnn_prediction = mnist_binary_connect_lenet_prediction
    elif args.net == 'binnet':
        mnist_cnn_prediction = mnist_binary_net_lenet_prediction
    elif args.net == 'bwn':
        mnist_cnn_prediction = mnist_binary_weight_lenet_prediction
    elif args.net == 'bincon_resnet':
        mnist_cnn_prediction = mnist_binary_connect_resnet_prediction
    elif args.net == 'binnet_resnet':
        mnist_cnn_prediction = mnist_binary_net_resnet_prediction
    elif args.net == 'bwn_resnet':
        mnist_cnn_prediction = mnist_binary_weight_resnet_prediction

    # TRAIN
    # Create input variables.
    image = nn.Variable([args.batch_size, 1, 28, 28])
    label = nn.Variable([args.batch_size, 1])
    # Create predition graph.
    pred = mnist_cnn_prediction(image / 255, test=False)
    pred.persistent = True
    # Create loss function.
    loss = F.mean(F.softmax_cross_entropy(pred, label))

    # TEST
    # Create input variables.
    vimage = nn.Variable([args.batch_size, 1, 28, 28])
    vlabel = nn.Variable([args.batch_size, 1])
    # Create predition graph.
    vpred = mnist_cnn_prediction(vimage / 255, test=True)

    # Create Solver.
    solver = S.Adam(args.learning_rate)
    solver.set_parameters(nn.get_parameters())

    # Create monitor.
    import nnabla.monitor as M
    monitor = M.Monitor(args.monitor_path)
    monitor_loss = M.MonitorSeries("Training loss", monitor, interval=10)
    monitor_err = M.MonitorSeries("Training error", monitor, interval=10)
    monitor_time = M.MonitorTimeElapsed("Training time", monitor, interval=100)
    monitor_verr = M.MonitorSeries("Test error", monitor, interval=10)

    # Training loop.
    for i in range(args.max_iter):
        if i % args.val_interval == 0:
            # Validation
            ve = 0.0
            for j in range(args.val_iter):
                vimage.d, vlabel.d = vdata.next()
                vpred.forward(clear_buffer=True)
                ve += categorical_error(vpred.d, vlabel.d)
            monitor_verr.add(i, ve / args.val_iter)
        if i % args.model_save_interval == 0:
            nn.save_parameters(os.path.join(
                args.model_save_path, 'params_%06d.h5' % i))
        # Training forward
        image.d, label.d = data.next()
        solver.zero_grad()
        loss.forward(clear_no_need_grad=True)
        # Training backward & update
        loss.backward(clear_buffer=True)
        solver.weight_decay(args.weight_decay)
        solver.update()
        # Monitor
        e = categorical_error(pred.d, label.d)
        monitor_loss.add(i, loss.d.copy())
        monitor_err.add(i, e)
        monitor_time.add(i)

    parameter_file = os.path.join(
        args.model_save_path, 'params_%06d.h5' % args.max_iter)
    structure_file = os.path.join(
        args.model_save_path, 'params_%06d.nntxt' % args.max_iter)
    nnp_file = os.path.join(
        args.model_save_path, '{}_{:06}.nnp'.format(args.net, args.max_iter))
    runtime_contents = {
        'networks': [
            {'name': 'Validation',
             'batch_size': args.batch_size,
             'variable': vpred}],
        'executors': [
            {'name': 'Runtime',
             'network': 'Validation',
             'variables': ['x', 'y']}]}

    nn.save_parameters(parameter_file)
    save.save(structure_file, runtime_contents)
    save.save(nnp_file, runtime_contents)


if __name__ == '__main__':
    train()
