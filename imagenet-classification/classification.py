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
import nnabla.logger as logger
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S

from args import get_args
from tiny_imagenet_data import data_iterator_tiny_imagenet
import model_resnet

import os
from collections import namedtuple


def categorical_error(pred, label):
    """
    Compute categorical error given score vectors and labels as
    numpy.ndarray.
    """
    pred_label = pred.argmax(1)
    return (pred_label != label.flat).mean()


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
    pimage = image_preprocess(image, nn_in_size)
    pred, hidden = model_resnet.resnet_imagenet(
        pimage, num_classes, args.num_layers, args.shortcut_type, test=test, tiny=tiny)
    loss = F.mean(F.softmax_cross_entropy(pred, label))
    Model = namedtuple('Model', ['image', 'label', 'pred', 'loss', 'hidden'])
    return Model(image, label, pred, loss, hidden)


def image_preprocess(image, img_size=224):
    h, w = image.shape[2:]
    size = min(h, w)
    min_size = img_size * 1.1
    max_size = min_size * 2
    min_scale = min_size / size
    max_scale = max_size / size
    image = F.image_augmentation(image, (3, img_size, img_size),
                                 min_scale=min_scale, max_scale=max_scale,
                                 angle=0.5, aspect_ratio=1.3, distortion=0.2,
                                 flip_lr=True, brightness=25.5,
                                 brightness_each=True,
                                 contrast=1.1, contrast_center=128.0,
                                 contrast_each=True, noise=25.5)
    image = image - 128
    return image


def train():
    """
    Main script.
    """

    args = get_args()

    # Get context.
    from nnabla.ext_utils import get_extension_context
    extension_module = args.context
    if args.context is None:
        extension_module = 'cpu'
    logger.info("Running in %s" % extension_module)
    ctx = get_extension_context(
        extension_module, device_id=args.device_id, type_config=args.type_config)
    nn.set_default_context(ctx)

    # Dataset
    # We use Tiny ImageNet from Stanford CS231N class.
    # https://tiny-imagenet.herokuapp.com/
    # Tiny ImageNet consists of 200 categories, each category has 500 images
    # in training set. The image size is 64x64. To adapt ResNet into 64x64
    # image inputs, the input image size of ResNet is set as 56x56, and
    # the stride in the first conv and the first max pooling are removed.
    data = data_iterator_tiny_imagenet(args.batch_size, 'train')
    vdata = data_iterator_tiny_imagenet(args.batch_size, 'val')

    num_classes = 200
    tiny = True  # TODO: Switch ILSVRC2012 dataset and TinyImageNet.
    t_model = get_model(
        args, num_classes, test=False, tiny=tiny)
    t_model.pred.persistent = True  # Not clearing buffer of pred in backward
    v_model = get_model(
        args, num_classes, test=True, tiny=tiny)
    v_model.pred.persistent = True  # Not clearing buffer of pred in forward

    # Create Solver.
    solver = S.Momentum(args.learning_rate, 0.9)
    solver.set_parameters(nn.get_parameters())

    # Create monitor.
    import nnabla.monitor as M
    monitor = M.Monitor(args.monitor_path)
    monitor_loss = M.MonitorSeries("Training loss", monitor, interval=10)
    monitor_err = M.MonitorSeries("Training error", monitor, interval=10)
    monitor_vloss = M.MonitorSeries("Validation loss", monitor, interval=10)
    monitor_verr = M.MonitorSeries("Validation error", monitor, interval=10)
    monitor_time = M.MonitorTimeElapsed("Training time", monitor, interval=10)

    # Training loop.
    for i in range(args.max_iter):
        # Save parameters
        if i % args.model_save_interval == 0:
            nn.save_parameters(os.path.join(
                args.model_save_path, 'param_%06d.h5' % i))

        # Validation
        if i % args.val_interval == 0:

            # Clear all intermediate memory to save memory.
            # t_model.loss.clear_recursive()

            l = 0.0
            e = 0.0
            for j in range(args.val_iter):
                images, labels = vdata.next()
                v_model.image.d = images
                v_model.label.d = labels
                v_model.image.data.cast(np.uint8, ctx)
                v_model.label.data.cast(np.int32, ctx)
                v_model.loss.forward(clear_buffer=True)
                l += v_model.loss.d
                e += categorical_error(v_model.pred.d, v_model.label.d)
            monitor_vloss.add(i, l / args.val_iter)
            monitor_verr.add(i, e / args.val_iter)

            # Clear all intermediate memory to save memory.
            # v_model.loss.clear_recursive()

        # Training
        l = 0.0
        e = 0.0
        solver.zero_grad()

        # Gradient accumulation loop
        for j in range(args.accum_grad):
            images, labels = data.next()
            t_model.image.d = images
            t_model.label.d = labels
            t_model.image.data.cast(np.uint8, ctx)
            t_model.label.data.cast(np.int32, ctx)
            t_model.loss.forward(clear_no_need_grad=True)
            t_model.loss.backward(clear_buffer=True)  # Accumulating gradients
            l += t_model.loss.d
            e += categorical_error(t_model.pred.d, t_model.label.d)
        solver.weight_decay(args.weight_decay)
        solver.update()
        monitor_loss.add(i, l / args.accum_grad)
        monitor_err.add(i, e / args.accum_grad)
        monitor_time.add(i)

        # Learning rate decay at scheduled iter
        if i in args.learning_rate_decay_at:
            solver.set_learning_rate(solver.learning_rate() * 0.1)
    nn.save_parameters(os.path.join(args.model_save_path,
                                    'param_%06d.h5' % args.max_iter))


if __name__ == '__main__':
    train()
