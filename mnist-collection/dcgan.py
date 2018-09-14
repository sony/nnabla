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

import numpy as np

import nnabla as nn
import nnabla.logger as logger
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import nnabla.utils.save as save

from args import get_args
from mnist_data import data_iterator_mnist

import os


def generator(z, maxh=256, test=False, output_hidden=False):
    """
    Building generator network which takes (B, Z, 1, 1) inputs and generates
    (B, 1, 28, 28) outputs.
    """
    # Define shortcut functions
    def bn(x):
        # Batch normalization
        return PF.batch_normalization(x, batch_stat=not test)

    def upsample2(x, c):
        # Twice upsampling with deconvolution.
        return PF.deconvolution(x, c, kernel=(4, 4), pad=(1, 1), stride=(2, 2), with_bias=False)

    assert maxh / 4 > 0
    with nn.parameter_scope("gen"):
        # (Z, 1, 1) --> (256, 4, 4)
        with nn.parameter_scope("deconv1"):
            d1 = F.elu(bn(PF.deconvolution(z, maxh, (4, 4), with_bias=False)))
        # (256, 4, 4) --> (128, 8, 8)
        with nn.parameter_scope("deconv2"):
            d2 = F.elu(bn(upsample2(d1, maxh / 2)))
        # (128, 8, 8) --> (64, 16, 16)
        with nn.parameter_scope("deconv3"):
            d3 = F.elu(bn(upsample2(d2, maxh / 4)))
        # (64, 16, 16) --> (32, 28, 28)
        with nn.parameter_scope("deconv4"):
            # Convolution with kernel=4, pad=3 and stride=2 transforms a 28 x 28 map
            # to a 16 x 16 map. Deconvolution with those parameters behaves like an
            # inverse operation, i.e. maps 16 x 16 to 28 x 28.
            d4 = F.elu(bn(PF.deconvolution(
                d3, maxh / 8, (4, 4), pad=(3, 3), stride=(2, 2), with_bias=False)))
        # (32, 28, 28) --> (1, 28, 28)
        with nn.parameter_scope("conv5"):
            x = F.tanh(PF.convolution(d4, 1, (3, 3), pad=(1, 1)))
    if output_hidden:
        return x, [d1, d2, d3, d4]
    return x


def discriminator(x, maxh=256, test=False, output_hidden=False):
    """
    Building discriminator network which maps a (B, 1, 28, 28) input to
    a (B, 1).
    """
    # Define shortcut functions
    def bn(xx):
        # Batch normalization
        return PF.batch_normalization(xx, batch_stat=not test)

    def downsample2(xx, c):
        return PF.convolution(xx, c, (3, 3), pad=(1, 1), stride=(2, 2), with_bias=False)

    assert maxh / 8 > 0
    with nn.parameter_scope("dis"):
        # (1, 28, 28) --> (32, 16, 16)
        with nn.parameter_scope("conv1"):
            c1 = F.elu(bn(PF.convolution(x, maxh / 8,
                                         (3, 3), pad=(3, 3), stride=(2, 2), with_bias=False)))
        # (32, 16, 16) --> (64, 8, 8)
        with nn.parameter_scope("conv2"):
            c2 = F.elu(bn(downsample2(c1, maxh / 4)))
        # (64, 8, 8) --> (128, 4, 4)
        with nn.parameter_scope("conv3"):
            c3 = F.elu(bn(downsample2(c2, maxh / 2)))
        # (128, 4, 4) --> (256, 4, 4)
        with nn.parameter_scope("conv4"):
            c4 = bn(PF.convolution(c3, maxh, (3, 3),
                                   pad=(1, 1), with_bias=False))
        # (256, 4, 4) --> (1,)
        with nn.parameter_scope("fc1"):
            f = PF.affine(c4, 1)
    if output_hidden:
        return f, [c1, c2, c3, c4]
    return f


def train(args):
    """
    Main script.
    """

    # Get context.
    from nnabla.ext_utils import get_extension_context
    logger.info("Running in %s" % args.context)
    ctx = get_extension_context(
        args.context, device_id=args.device_id, type_config=args.type_config)
    nn.set_default_context(ctx)

    # Create CNN network for both training and testing.
    # TRAIN

    # Fake path
    z = nn.Variable([args.batch_size, 100, 1, 1])
    fake = generator(z)
    fake.persistent = True  # Not to clear at backward
    pred_fake = discriminator(fake)
    loss_gen = F.mean(F.sigmoid_cross_entropy(
        pred_fake, F.constant(1, pred_fake.shape)))
    fake_dis = fake.get_unlinked_variable(need_grad=True)
    fake_dis.need_grad = True  # TODO: Workaround until v1.0.2
    pred_fake_dis = discriminator(fake_dis)
    loss_dis = F.mean(F.sigmoid_cross_entropy(
        pred_fake_dis, F.constant(0, pred_fake_dis.shape)))

    # Real path
    x = nn.Variable([args.batch_size, 1, 28, 28])
    pred_real = discriminator(x)
    loss_dis += F.mean(F.sigmoid_cross_entropy(pred_real,
                                               F.constant(1, pred_real.shape)))

    # Create Solver.
    solver_gen = S.Adam(args.learning_rate, beta1=0.5)
    solver_dis = S.Adam(args.learning_rate, beta1=0.5)
    with nn.parameter_scope("gen"):
        solver_gen.set_parameters(nn.get_parameters())
    with nn.parameter_scope("dis"):
        solver_dis.set_parameters(nn.get_parameters())

    # Create monitor.
    import nnabla.monitor as M
    monitor = M.Monitor(args.monitor_path)
    monitor_loss_gen = M.MonitorSeries("Generator loss", monitor, interval=10)
    monitor_loss_dis = M.MonitorSeries(
        "Discriminator loss", monitor, interval=10)
    monitor_time = M.MonitorTimeElapsed("Time", monitor, interval=100)
    monitor_fake = M.MonitorImageTile(
        "Fake images", monitor, normalize_method=lambda x: x + 1 / 2.)

    data = data_iterator_mnist(args.batch_size, True)
    # Training loop.
    for i in range(args.max_iter):
        if i % args.model_save_interval == 0:
            with nn.parameter_scope("gen"):
                nn.save_parameters(os.path.join(
                    args.model_save_path, "generator_param_%06d.h5" % i))
            with nn.parameter_scope("dis"):
                nn.save_parameters(os.path.join(
                    args.model_save_path, "discriminator_param_%06d.h5" % i))

        # Training forward
        image, _ = data.next()
        x.d = image / 255. - 0.5  # [0, 255] to [-1, 1]
        z.d = np.random.randn(*z.shape)

        # Generator update.
        solver_gen.zero_grad()
        loss_gen.forward(clear_no_need_grad=True)
        loss_gen.backward(clear_buffer=True)
        solver_gen.weight_decay(args.weight_decay)
        solver_gen.update()
        monitor_fake.add(i, fake)
        monitor_loss_gen.add(i, loss_gen.d.copy())

        # Discriminator update.
        solver_dis.zero_grad()
        loss_dis.forward(clear_no_need_grad=True)
        loss_dis.backward(clear_buffer=True)
        solver_dis.weight_decay(args.weight_decay)
        solver_dis.update()
        monitor_loss_dis.add(i, loss_dis.d.copy())
        monitor_time.add(i)

    with nn.parameter_scope("gen"):
        nn.save_parameters(os.path.join(
            args.model_save_path, "generator_param_%06d.h5" % i))
    with nn.parameter_scope("dis"):
        nn.save_parameters(os.path.join(
            args.model_save_path, "discriminator_param_%06d.h5" % i))


if __name__ == '__main__':
    monitor_path = 'tmp.monitor.dcgan'
    args = get_args(monitor_path=monitor_path, model_save_path=monitor_path,
                    max_iter=20000, learning_rate=0.0002, batch_size=64,
                    weight_decay=0.0001)
    train(args)
