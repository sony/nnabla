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
    pimage = image_preprocess(image, nn_in_size, data_size, test)
    pred, hidden = model_resnet.resnet_imagenet(
        pimage, num_classes, args.num_layers, args.shortcut_type, test=test, tiny=tiny)
    loss = F.mean(F.softmax_cross_entropy(pred, label))
    Model = namedtuple('Model', ['image', 'label', 'pred', 'loss', 'hidden'])
    return Model(image, label, pred, loss, hidden)


def image_preprocess(image, img_size=224, data_size=320, test=False):
    h, w = image.shape[2:]
    image = image / 255.0
    if test:
        _img_size = data_size * 0.875  # Ratio of size is 87.5%
        hs = (h - _img_size) / 2
        ws = (w - _img_size) / 2
        he = (h + _img_size) / 2
        we = (w + _img_size) / 2
        image = F.slice(image, (0, ws, hs), (3, we, he), (1, 1, 1))
        image = F.image_augmentation(
            image, (3, img_size, img_size), min_scale=0.8, max_scale=0.8)
    else:
        size = min(h, w)
        min_size = img_size * 1.1
        max_size = min_size * 2
        min_scale = min_size / size
        max_scale = max_size / size
        image = F.image_augmentation(image, (3, img_size, img_size), pad=(0, 0), min_scale=min_scale, max_scale=max_scale, angle=0.5, aspect_ratio=1.3,
                                     distortion=0.2, flip_lr=True, flip_ud=False, brightness=0.0, brightness_each=True, contrast=1.1, contrast_center=0.5, contrast_each=True, noise=0.0)
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
    if args.tiny_mode:
        n_train_samples = 100000
    else:
        n_train_samples = 1282167

    # Communicator and Context
    from nnabla.ext_utils import get_extension_context
    extension_module = "cudnn"
    ctx = get_extension_context(extension_module, type_config=args.type_config)
    comm = C.MultiProcessDataParalellCommunicator(ctx)
    comm.init()
    n_devices = comm.size
    mpi_rank = comm.rank
    device_id = mpi_rank
    ctx.device_id = str(device_id)
    nn.set_default_context(ctx)

    # workarond to start with the same parameters.
    rng = np.random.RandomState(device_id)
    if args.tiny_mode:
        # We use Tiny ImageNet from Stanford CS231N class.
        # (Tiny ImageNet, https://tiny-imagenet.herokuapp.com/)
        # Tiny ImageNet consists of 200 categories, each category has 500 images
        # in training set. The image size is 64x64. To adapt ResNet into 64x64
        # image inputs, the input image size of ResNet is set as 56x56, and
        # the stride in the first conv and the first max pooling are removed.
        # Please check README.
        data = data_iterator_tiny_imagenet(args.batch_size, 'train')
        vdata = data_iterator_tiny_imagenet(args.batch_size, 'val')
        num_classes = 200
    else:
        # We use ImageNet.
        # (ImageNet, https://imagenet.herokuapp.com/)
        # ImageNet consists of 1000 categories, each category has 1280 images
        # in training set. The image size is various. To adapt ResNet into
        # 320x320 image inputs, the input image size of ResNet is set as
        # 224x224. We need to get tar file and create cache file(320x320 images).
        # Please check README.
        data = data_iterator_imagenet(
            args.batch_size, args.train_cachefile_dir, rng=rng)
        vdata = data_iterator_imagenet(args.batch_size, args.val_cachefile_dir)
        vdata = vdata.slice(num_of_slices=n_devices, slice_pos=device_id)
        num_classes = 1000
    # Workaround to start with the same initialized weights for all workers.
    np.random.seed(313)
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

    # Add parameters to communicator.
    comm.add_context_and_parameters((ctx, nn.get_parameters()))

    # Create Solver.
    solver = S.Momentum(args.learning_rate, 0.9)
    solver.set_parameters(nn.get_parameters())

    # Setting warmup.
    base_lr = args.learning_rate / n_devices
    warmup_iter = int(1. * n_train_samples /
                      args.batch_size / args.accum_grad / n_devices) * args.warmup_epoch
    warmup_slope = base_lr * (n_devices - 1) / warmup_iter
    solver.set_learning_rate(base_lr)

    # Create monitor.
    import nnabla.monitor as M
    monitor = M.Monitor(args.monitor_path)
    monitor_loss = M.MonitorSeries("Training loss", monitor, interval=10)
    monitor_err = M.MonitorSeries("Training error", monitor, interval=10)
    monitor_vloss = M.MonitorSeries("Validation loss", monitor, interval=1)
    monitor_verr = M.MonitorSeries("Validation error", monitor, interval=1)
    monitor_time = M.MonitorTimeElapsed("Training time", monitor, interval=10)
    monitor_vtime = M.MonitorTimeElapsed(
        "Validation time", monitor, interval=1)

    # Training loop.
    vl = nn.Variable()
    ve = nn.Variable()
    for i in range(int(args.max_iter / n_devices)):
        # Save parameters
        if i % (args.model_save_interval // n_devices) == 0 and device_id == 0:
            nn.save_parameters(os.path.join(
                args.model_save_path, 'param_%06d.h5' % i))

        # Validation
        if i % (args.val_interval // n_devices) == 0 and i != 0:
            ve_local = 0.
            vl_local = 0.
            val_iter_local = args.val_iter // n_devices
            for j in range(val_iter_local):
                images, labels = vdata.next()
                v_model.image.d = images
                v_model.label.d = labels
                v_model.image.data.cast(np.uint8, ctx)
                v_model.label.data.cast(np.int32, ctx)
                v_model.loss.forward(clear_buffer=True)
                v_e.forward(clear_buffer=True)
                vl_local += v_model.loss.d.copy()
                ve_local += v_e.d.copy()
            vl_local /= val_iter_local
            vl.d = vl_local
            comm.all_reduce(vl.data, division=True, inplace=True)
            ve_local /= val_iter_local
            ve.d = ve_local
            comm.all_reduce(ve.data, division=True, inplace=True)

            if device_id == 0:
                monitor_vloss.add(i * n_devices, vl.d.copy())
                monitor_verr.add(i * n_devices, ve.d.copy())
                monitor_vtime.add(i * n_devices)

        # Training
        l = 0.0
        e = 0.0
        solver.zero_grad()

        def accumulate_error(l, e, t_model, t_e):
            l += t_model.loss.d
            e += t_e.d
            return l, e

        # Gradient accumulation loop
        for j in range(args.accum_grad):
            images, labels = data.next()
            if j != 0:
                # Update e and l according to previous results of forward
                # propagation.
                # The update of last iteration is performed
                # after solver update to avoid unnecessary CUDA synchronization.
                # This is performed after data.next() in order to overlap
                # the data loading and graph execution.
                # TODO: Move this to the bottom of the loop when prefetch
                # data loader is available.
                l, e = accumulate_error(l, e, t_model, t_e)
            t_model.image.d = images
            t_model.label.d = labels
            t_model.image.data.cast(np.uint8, ctx)
            t_model.label.data.cast(np.int32, ctx)
            t_model.loss.forward(clear_no_need_grad=True)
            t_model.loss.backward(clear_buffer=True)  # Accumulating gradients
            t_e.forward(clear_buffer=True)

        # AllReduce
        params = [x.grad for x in nn.get_parameters().values()]
        comm.all_reduce(params, division=False, inplace=False)

        # Update
        solver.weight_decay(args.weight_decay)
        solver.update()

        # Accumulate errors after solver update
        l, e = accumulate_error(l, e, t_model, t_e)

        # Linear Warmup
        if i <= warmup_iter:
            lr = base_lr + warmup_slope * i
            solver.set_learning_rate(lr)

        # Synchronize by averaging the weights over devices using allreduce
        if (i+1) % args.sync_weight_every_itr == 0:
            weights = [x.data for x in nn.get_parameters().values()]
            comm.all_reduce(weights, division=True, inplace=True)

        if device_id == 0:
            monitor_loss.add(i * n_devices, l / args.accum_grad)
            monitor_err.add(i * n_devices, e / args.accum_grad)
            monitor_time.add(i * n_devices)

        # Learning rate decay at scheduled iter
        if i * n_devices in args.learning_rate_decay_at:
            solver.set_learning_rate(solver.learning_rate() * 0.1)

    if device_id == 0:
        nn.save_parameters(os.path.join(
            args.model_save_path,
            'param_%06d.h5' % (args.max_iter / n_devices)))


if __name__ == '__main__':
    """
    Call this script with `mpirun` or `mpiexec`
    $ mpirun -n 4 python multi_device_multi_process_classification.py -b 32 -a 2 -L 50 -l 0.1 -i 2000000 -v 20004 -j 1563 -s 20004 -D 600000,1200000,1800000 -T "The path of training cachefile" -V "The path of validation cache file"
    """
    train()
