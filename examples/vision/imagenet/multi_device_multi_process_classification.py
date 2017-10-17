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
import nnabla.communicators as C
import nnabla.logger as logger
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S

from args import get_args
from tiny_imagenet_data import data_iterator_tiny_imagenet
from imagenet_data import data_iterator_imagenet
import model_resnet

import os
from collections import namedtuple

def get_model(args, num_classes, test=False, tiny=False):
    """
    Create computation graph and variables.

    Args:

        tiny: Tiny ImageNet mode if True.
    """
    data_size = 320
    nn_in_size = 224
    if tiny:
        data_size = 64
        nn_in_size = 56
    image = nn.Variable([args.batch_size, 3, data_size, data_size])
    label = nn.Variable([args.batch_size, 1])
    pimage = image_preprocess(image, nn_in_size, test)
    pred, hidden = model_resnet.resnet_imagenet(
        pimage, num_classes, args.num_layers, args.shortcut_type, test=test, tiny=tiny)
    loss = F.mean(F.softmax_cross_entropy(pred, label))
    Model = namedtuple('Model', ['image', 'label', 'pred', 'loss', 'hidden'])
    return Model(image, label, pred, loss, hidden)


def image_preprocess(image, img_size=224, test=False):
    h, w = image.shape[2:]
    image = image / 255
    if test:
        hs = (h - img_size) / 2
        ws = (w - img_size) / 2
        he = (h + img_size) / 2
        we = (w + img_size) / 2
        image = F.slice(image, (0,ws,hs), (3,we,he), (1, 1, 1))
    else:
        size = min(h, w)
        min_size = img_size * 1.1
        max_size = min_size * 2
        min_scale = min_size / size
        max_scale = max_size / size
        image = F.image_augmentation(image, (3, img_size, img_size), pad=(0,0), min_scale=min_scale, max_scale=max_scale, angle=0.5, aspect_ratio=1.3, distortion=0.2, flip_lr=True, flip_ud=False, brightness=0.0, brightness_each=True, contrast=1.1, contrast_center=0.5, contrast_each=True, noise=0.0)
    image = image - 0.5
    return image

def train():
    """
    Main script.

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
      * Inplace allreduce (THIS IS THE MAIN difference from a single device training)
      * Solver updates parameters by using gradients computed by backprop.
      * Compute training error

    """

    args = get_args()
    n_train_samples = 1282167

    """
    # Get context.
    from nnabla.contrib.context import extension_context
    extension_module = args.context
    if args.context is None:
        extension_module = 'cpu'
    logger.info("Running in %s" % extension_module)
    ctx = extension_context(extension_module, device_id=args.device_id)
    nn.set_default_context(ctx)
    """

    # Communicator and Context
    extension_module = "cuda.cudnn"
    ctx = extension_context(extension_module)
    comm = C.MultiProcessDataParalellCommunicator(ctx)
    comm.init()
    n_devices = comm.size
    mpi_rank = comm.rank
    device_id = mpi_rank
    ctx = extension_context(extension_module, device_id=device_id)

    # Dataset
    # We use Tiny ImageNet from Stanford CS231N class.
    # https://tiny-imagenet.herokuapp.com/
    # Tiny ImageNet consists of 200 categories, each category has 500 images
    # in training set. The image size is 64x64. To adapt ResNet into 64x64
    # image inputs, the input image size of ResNet is set as 56x56, and
    # the stride in the first conv and the first max pooling are removed.
    if args.tiny_mode:
        data = data_iterator_tiny_imagenet(args.batch_size, 'train')
        vdata = data_iterator_tiny_imagenet(args.batch_size, 'val')
        num_classes = 200
    else:
        data = data_iterator_imagenet(args.batch_size, args.train_cachefile_dir)
        vdata = data_iterator_imagenet(args.batch_size, args.val_cachefile_dir)
        num_classes = 1000
    t_model = get_model(
        args, num_classes, test=False, tiny=args.tiny_mode)
    t_model.pred.persistent = True  # Not clearing buffer of pred in backward
    t_pred2 = t_model.pred.unlinked()
    t_e = F.mean(F.top_n_error(t_pred2, t_model.label))
    v_model = get_model(
        args, num_classes, test=True, tiny=args.tiny_mode)
    v_model.pred.persistent = True  # Not clearing buffer of pred in forward
    v_pred2 = v_model.pred.unlinked()
    v_e = F.mean(F.top_n_error(v_pred2, v_model.label))

    # add parameters to communicator
    comm.add_context_and_parameters((ctx, nn.get_parameters()))

    # Create Solver.
    solver = S.Momentum(args.learning_rate, 0.9)
    solver.set_parameters(nn.get_parameters())
    base_lr = args.learning_rate
    warmup_iter = int(1. * n_train_samples /
                      args.batch_size / n_devices) * args.warmup_epoch
    warmup_slope = 1. * n_devices / warmup_iter

    # Create monitor.
    import nnabla.monitor as M
    monitor = M.Monitor(args.monitor_path)
    monitor_loss = M.MonitorSeries("Training loss", monitor, interval=10)
    monitor_err = M.MonitorSeries("Training error", monitor, interval=10)
    monitor_vloss = M.MonitorSeries("Validation loss", monitor, interval=10)
    monitor_verr = M.MonitorSeries("Validation error", monitor, interval=10)
    monitor_time = M.MonitorTimeElapsed("Training time", monitor, interval=10)
    monitor_vtime = M.MonitorTimeElapsed("Validation time", monitor, interval=10)

    # Training loop.
    for i in range(args.max_iter):
        # Save parameters
        if i % args.model_save_interval == 0 and mpi_rank == 0:
            nn.save_parameters(os.path.join(
                args.model_save_path, 'param_%06d.h5' % i))

        # Validation
        if i % args.val_interval == 0 and mpi_rank == 0:

            # Clear all intermediate memory to save memory.
            # t_model.loss.clear_recursive()

            l = 0.0
            e = 0.0
            for j in range(args.val_iter):
                images, labels = vdata.next()
                # The data loading of the next mini-batch from files is overlapped
                # with the computation of forward and backward propagation in your CUDA device
                # because the CUDA stream runs asynchronously and the synchronization happens
                # when the loss variable is referred by the `.d` accessor.
                if i !=  0:
                    l += v_model.loss.d
                    v_e.forward(clear_buffer=True)
                    e += v_e.d
                v_model.image.d = images
                v_model.label.d = labels
                v_model.image.data.cast(np.uint8, ctx)
                v_model.label.data.cast(np.int32, ctx)
                v_model.loss.forward(clear_buffer=True)
            monitor_vloss.add(i, l / args.val_iter)
            monitor_verr.add(i, e / args.val_iter)
            monitor_vtime.add(i)

            # Clear all intermediate memory to save memory.
            # v_model.loss.clear_recursive()

        # Training
        l = 0.0
        e = 0.0
        solver.zero_grad()

        # Gradient accumulation loop
        for j in range(args.accum_grad):
            images, labels = data.next()
            # The data loading of the next mini-batch from files is overlapped
            # with the computation of forward and backward propagation in your CUDA device
            # because the CUDA stream runs asynchronously and the synchronization happens
            # when the loss variable is referred by the `.d` accessor.
            if i !=  0:
                l += t_model.loss.d
                t_e.forward(clear_buffer=True)
                e += t_e.d
            t_model.image.d = images
            t_model.label.d = labels
            t_model.image.data.cast(np.uint8, ctx)
            t_model.label.data.cast(np.int32, ctx)
            t_model.loss.forward(clear_no_need_grad=True)
            t_model.loss.backward(clear_buffer=True)  # Accumulating gradients
        comm.allreduce(division=True)
        solver.weight_decay(args.weight_decay)
        solver.update()
        # Linear Warmup
        if i < warmup_iter:
            lr = base_lr * n_devices * warmup_slope * i
            solver.set_learning_rate(lr)
        else:
            lr = base_lr * n_devices
            solver.set_learning_rate(lr)
        monitor_loss.add(i, l / args.accum_grad)
        monitor_err.add(i, e / args.accum_grad)
        monitor_time.add(i)

        # Learning rate decay at scheduled iter
        if i in args.learning_rate_decay_at:
            solver.set_learning_rate(solver.learning_rate() * 0.1)
    if mpi_rank == 0:
        nn.save_parameters(os.path.join(args.model_save_path, 'param_%06d.h5' % args.max_iter))


if __name__ == '__main__':
    train()
