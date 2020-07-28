# Copyright (c) 2020 Sony Corporation. All Rights Reserved.
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
from functools import partial


# --------------------
# Network for Decoder
# --------------------

def netG_decoder(x, test=False):
    # x: (1, 15, 64, 64) -> c0: (1, 15, 128, 128)
    with nn.parameter_scope('ReluDeconvBN1'):
        c0 = PF.batch_normalization(PF.deconvolution(F.relu(x), 15, (4, 4), pad=(
            1, 1), stride=(2, 2), with_bias=False), batch_stat=not test)

    # c0: (1, 15, 128, 128) -> c1: (1, 15, 256, 256)
    with nn.parameter_scope('ReluDeconvBN2'):
        c1 = F.tanh(PF.deconvolution(F.relu(c0), 15,
                                     (4, 4), pad=(1, 1), stride=(2, 2)))

    # c1: (1, 15, 256, 256) -> down_0: (1, 64, 128, 128)
    with nn.parameter_scope('down0'):
        down_0 = PF.convolution(c1, 64, (4, 4), pad=(
            1, 1), stride=(2, 2), with_bias=False)

    # down_0: (1, 64, 128, 128) -> down_1: (1, 128, 64, 64)
    with nn.parameter_scope('down1'):
        down_1 = PF.batch_normalization(PF.convolution(F.leaky_relu(down_0, alpha=0.2), 128, (4, 4), pad=(
            1, 1), stride=(2, 2), with_bias=False), batch_stat=not test)

    # down_1: (1, 128, 64, 64) -> down_2: (1, 256, 32, 32)
    with nn.parameter_scope('down2'):
        down_2 = PF.batch_normalization(PF.convolution(F.leaky_relu(down_1, alpha=0.2), 256, (4, 4), pad=(
            1, 1), stride=(2, 2), with_bias=False), batch_stat=not test)

    # down_2: (1, 256, 32, 32) -> down_3: (1, 512, 16, 16)
    with nn.parameter_scope('down3'):
        down_3 = PF.batch_normalization(PF.convolution(F.leaky_relu(down_2, alpha=0.2), 512, (4, 4), pad=(
            1, 1), stride=(2, 2), with_bias=False), batch_stat=not test)

    # down_3: (1, 512, 16, 16) -> down_4: (1, 512, 8, 8)
    with nn.parameter_scope('down4'):
        down_4 = PF.batch_normalization(PF.convolution(F.leaky_relu(down_3, alpha=0.2), 512, (4, 4), pad=(
            1, 1), stride=(2, 2), with_bias=False), batch_stat=not test)

    # down_4: (1, 512, 8, 8) -> down_5: (1, 512, 4, 4)
    with nn.parameter_scope('down5'):
        down_5 = PF.batch_normalization(PF.convolution(F.leaky_relu(down_4, alpha=0.2), 512, (4, 4), pad=(
            1, 1), stride=(2, 2), with_bias=False), batch_stat=not test)

    # down_5: (1, 512, 4, 4) -> down_6: (1, 512, 2, 2)
    with nn.parameter_scope('down6'):
        down_6 = PF.batch_normalization(PF.convolution(F.leaky_relu(down_5, alpha=0.2), 512, (4, 4), pad=(
            1, 1), stride=(2, 2), with_bias=False), batch_stat=not test)

    # down_6: (1, 512, 2, 2) -> down_7: (1, 512, 1, 1)
    with nn.parameter_scope('down7'):
        down_7 = PF.convolution(F.leaky_relu(down_6, alpha=0.2), 512, (4, 4), pad=(
            1, 1), stride=(2, 2), with_bias=False)

    # down_7: (1, 512, 1, 1) -> up_0: (1, 512, 2, 2)
    with nn.parameter_scope('up0'):
        up_0 = PF.batch_normalization(PF.deconvolution(F.relu(down_7), 512, (4, 4), pad=(
            1, 1), stride=(2, 2), with_bias=False), batch_stat=not test)

    # down_6: (1, 512, 2, 2) + up_0: (1, 512, 2, 2) -> up_1: (1, 512, 4, 4)
    with nn.parameter_scope('up1'):
        up_1 = PF.batch_normalization(PF.deconvolution(F.relu(F.concatenate(
            down_6, up_0, axis=1)), 512, (4, 4), pad=(1, 1), stride=(2, 2), with_bias=False), batch_stat=not test)
    if not test:
        up_1 = F.dropout(up_1, 0.5)

    # down_5: (1, 512, 4, 4) + up_1: (1, 512, 4, 4)-> up_2: (1, 512, 8, 8)
    with nn.parameter_scope('up2'):
        up_2 = PF.batch_normalization(PF.deconvolution(F.relu(F.concatenate(
            down_5, up_1, axis=1)), 512, (4, 4), pad=(1, 1), stride=(2, 2), with_bias=False), batch_stat=not test)
    if not test:
        up_2 = F.dropout(up_2, 0.5)

    # down_4: (1, 512, 8, 8) + up_2: (1, 512, 8, 8) -> up_3: (1, 512, 16, 16)
    with nn.parameter_scope('up3'):
        up_3 = PF.batch_normalization(PF.deconvolution(F.relu(F.concatenate(
            down_4, up_2, axis=1)), 512, (4, 4), pad=(1, 1), stride=(2, 2), with_bias=False), batch_stat=not test)
    if not test:
        up_3 = F.dropout(up_3, 0.5)

    # down_3: (1, 512, 16, 16) + up_3: (1, 512, 16, 16) -> up_4: (1, 256, 32, 32)
    with nn.parameter_scope('up4'):
        up_4 = PF.batch_normalization(PF.deconvolution(F.relu(F.concatenate(
            down_3, up_3, axis=1)), 256, (4, 4), pad=(1, 1), stride=(2, 2), with_bias=False), batch_stat=not test)

    # down_2: (1, 256, 32, 32) + up_4: (1, 256, 32, 32) -> up_5: (1, 128, 64, 64)
    with nn.parameter_scope('up5'):
        up_5 = PF.batch_normalization(PF.deconvolution(F.relu(F.concatenate(
            down_2, up_4, axis=1)), 128, (4, 4), pad=(1, 1), stride=(2, 2), with_bias=False), batch_stat=not test)

    # down_1: (1, 128, 64, 64) + up_5: (1, 128, 64, 64) -> up_6: (1, 64, 128, 128)
    with nn.parameter_scope('up6'):
        up_6 = PF.batch_normalization(PF.deconvolution(F.relu(F.concatenate(
            down_1, up_5, axis=1)), 64, (4, 4), pad=(1, 1), stride=(2, 2), with_bias=False), batch_stat=not test)

    # down_0: (1, 64, 128, 128) + up_6: (1, 64, 128, 128) -> output: (1, 3, 256, 256)
    with nn.parameter_scope('up7'):
        output = F.tanh(PF.deconvolution(F.relu(F.concatenate(
            down_0, up_6, axis=1)), 3, (4, 4), pad=(1, 1), stride=(2, 2)))

    return output


def netD_decoder(x, test=False):
    # x: (1, 18, 256, 256)
    kw = (4, 4)
    pad = (1, 1)
    stride = (2, 2)
    # (1, 18, 256, 256) -> (1, 64, 128, 128)
    with nn.parameter_scope('conv0'):
        c0 = F.leaky_relu(PF.convolution(
            x, 64, kw, pad=pad, stride=stride), alpha=0.2)

    # (1, 64, 128, 128) -> (1, 128, 64, 64)
    with nn.parameter_scope('conv1'):
        c1 = F.leaky_relu(PF.batch_normalization(PF.convolution(
            c0, 128, kw, pad=pad, stride=stride, with_bias=False), batch_stat=not test), alpha=0.2)

    # (1, 128, 64, 64) -> (1, 256, 32, 32)
    with nn.parameter_scope('conv2'):
        c2 = F.leaky_relu(PF.batch_normalization(PF.convolution(
            c1, 256, kw, pad=pad, stride=stride, with_bias=False), batch_stat=not test), alpha=0.2)

    # (1, 256, 32, 32) -> (1, 512, 31, 31)
    with nn.parameter_scope('conv3'):
        c3 = F.leaky_relu(PF.batch_normalization(PF.convolution(
            c2, 512, kw, pad=pad, stride=(1, 1), with_bias=False), batch_stat=not test), alpha=0.2)

    # (1, 512, 31, 31) -> (1, 1, 30, 30)
    with nn.parameter_scope('conv4'):
        c4 = PF.convolution(c3, 1, kw, pad=pad, stride=(1, 1))
        c4 = F.sigmoid(c4)

    return c4

# ------------------------------
# Network for Transformer
# ------------------------------


def convblock(x, n=64, k=(3, 3), s=(2, 2), p=(1, 1), test=False, norm_type="batch_norm"):
    x = PF.convolution(x, n, k, pad=p, stride=s, with_bias=False)
    if norm_type == "instance_norm":
        x = PF.instance_normalization(x, eps=1e-05)
    else:
        x = PF.batch_normalization(x, batch_stat=not test)
    x = F.relu(x)
    return x


def deconvblock(x, n=64, k=(3, 3), s=(2, 2), p=(1, 1), test=False, norm_type="batch_norm"):
    x = PF.deconvolution(x, n, kernel=(3, 3), pad=(1, 1), stride=(
        2, 2), output_padding=(1, 1), with_bias=False)
    if norm_type == "instance_norm":
        x = PF.instance_normalization(x, eps=1e-05)
    else:
        x = PF.batch_normalization(x, batch_stat=not test)
    x = F.relu(x)
    return x


def resblock(x, n=256, test=False, norm_type="batch_norm"):
    r = x
    r = F.pad(r, (1, 1, 1, 1), 'reflect')
    with nn.parameter_scope('block1'):
        r = PF.convolution(r, n, (3, 3), with_bias=False)
        if norm_type == "instance_norm":
            r = PF.instance_normalization(r, eps=1e-05)
        else:
            r = PF.batch_normalization(r, batch_stat=not test)
        r = F.relu(r)

    r = F.pad(r, (1, 1, 1, 1), 'reflect')
    with nn.parameter_scope('block2'):
        r = PF.convolution(r, n, (3, 3), with_bias=False)
        if norm_type == "instance_norm":
            r = PF.instance_normalization(r, eps=1e-05)
        else:
            r = PF.batch_normalization(r, batch_stat=not test)
    return x + r


def netG_transformer(x, test=False, norm_type="batch_norm"):
    # x: (1, 15, 64, 64) -> x: (1, 64, 64, 64)
    x = F.pad(x, (3, 3, 3, 3), 'reflect')
    with nn.parameter_scope('conv1'):
        x = convblock(x, n=64, k=(7, 7), s=(1, 1), p=(
            0, 0), test=test, norm_type=norm_type)

    # x: (1, 64, 64, 64) -> x: (1, 128, 32, 32)
    with nn.parameter_scope('conv2'):
        x = convblock(x, n=64*2, k=(3, 3), s=(2, 2), p=(1, 1),
                      test=test, norm_type=norm_type)

    # x: (1, 128, 32, 32) -> x: (1, 256, 16, 16)
    with nn.parameter_scope('conv3'):
        x = convblock(x, n=64*4, k=(3, 3), s=(2, 2), p=(1, 1),
                      test=test, norm_type=norm_type)

    # x: (1, 256, 16, 16) -> x: (1, 256, 16, 16)
    for i in range(9):
        with nn.parameter_scope(f'res{i + 1}'):
            x = resblock(x, n=64*4, test=test, norm_type=norm_type)

    # x: (1, 256, 16, 16) -> x: (1, 128, 32, 32)
    with nn.parameter_scope('deconv1'):
        x = deconvblock(x, n=64*2, k=(4, 4), s=(2, 2),
                        p=(1, 1), test=test, norm_type=norm_type)

    # x: (1, 128, 32, 32) -> x: (1,  64, 64, 64)
    with nn.parameter_scope('deconv2'):
        x = deconvblock(x, n=64, k=(4, 4), s=(2, 2), p=(
            1, 1), test=test, norm_type=norm_type)

    # x: (1, 64, 64, 64) -> x: (1, 15, 64, 64)
    x = F.pad(x, (3, 3, 3, 3), 'reflect')
    with nn.parameter_scope('deconv3'):
        x = PF.convolution(x, 15, kernel=(7, 7), with_bias=True)
        x = F.sigmoid(x)

    return x


def netD_transformer(x, test=False):
    kw = (4, 4)
    pad = (1, 1)
    stride = (2, 2)

    # (1, 15,  64,  64) -> (1, 64,  32, 32)
    with nn.parameter_scope('conv0'):
        c0 = F.leaky_relu(PF.convolution(
            x, 64, kw, pad=pad, stride=stride), alpha=0.2)

    # (1, 64,  32,  32) -> (1, 128, 16, 16)
    with nn.parameter_scope('conv1'):
        c1 = F.leaky_relu(PF.batch_normalization(PF.convolution(
            c0, 128, kw, pad=pad, stride=stride, with_bias=False), batch_stat=not test), alpha=0.2)

    # (1, 128, 16, 16) -> (1, 256, 8, 8)
    with nn.parameter_scope('conv2'):
        c2 = F.leaky_relu(PF.batch_normalization(PF.convolution(
            c1, 256, kw, pad=pad, stride=stride, with_bias=False), batch_stat=not test), alpha=0.2)

    # (1, 256, 8, 8) -> (1, 512, 7, 7)
    with nn.parameter_scope('conv3'):
        c3 = F.leaky_relu(PF.batch_normalization(PF.convolution(
            c2, 512, kw, pad=pad, stride=(1, 1), with_bias=False), batch_stat=not test), alpha=0.2)

    # (1, 512, 7, 7) -> (1, 1, 6, 6)
    with nn.parameter_scope('conv4'):
        c4 = PF.convolution(c3, 1, kw, pad=pad, stride=(1, 1))

    return c4


# -------------------------------------
# Align Network (used for Transformer)
# -------------------------------------

def align_resnet(x, channel_basic=16, test=False, fix_parameters=False):

    def resblock_align(x, channel, stride=(1, 1), test=False, downsample=False, fix_parameters=False):
        residual = x
        with nn.parameter_scope('conv1'):
            h = PF.convolution(x, channel, kernel=(3, 3), stride=stride, pad=(
                1, 1), with_bias=False, fix_parameters=fix_parameters)
        with nn.parameter_scope('bn1'):
            h = PF.batch_normalization(
                h, batch_stat=not test, fix_parameters=fix_parameters)
        h = F.relu(h, inplace=True)

        with nn.parameter_scope('conv2'):
            h = PF.convolution(h, channel, kernel=(3, 3), stride=(1, 1), pad=(
                1, 1), with_bias=False, fix_parameters=fix_parameters)
        with nn.parameter_scope('bn2'):
            h = PF.batch_normalization(
                h, batch_stat=not test, fix_parameters=fix_parameters)

        if downsample:
            with nn.parameter_scope('downsample'):
                residual = PF.convolution(x, channel, kernel=(
                    1, 1), stride=stride, with_bias=False, fix_parameters=fix_parameters)
                residual = PF.batch_normalization(
                    residual, batch_stat=not test, fix_parameters=fix_parameters)

        out = h + residual
        out = F.relu(out, inplace=True)

        return out

    with nn.parameter_scope('layer0'):
        h = PF.convolution(x, 3, kernel=(3, 3), stride=(1, 1), pad=(
            1, 1), with_bias=True, fix_parameters=fix_parameters)
    with nn.parameter_scope('layer1'):
        h = PF.convolution(h, 16, kernel=(7, 7), stride=(2, 2), pad=(
            3, 3), with_bias=False, fix_parameters=fix_parameters)
    with nn.parameter_scope('layer2'):
        h = PF.batch_normalization(
            h, batch_stat=not test, fix_parameters=fix_parameters)
    h = F.relu(h, inplace=True)
    h = F.max_pooling(h, kernel=(3, 3), stride=(2, 2), pad=(1, 1))

    use_downsample = False
    stride = (1, 1)
    for i in range(5, 9):
        with nn.parameter_scope(f'layer{i}_0'):
            h = resblock_align(h, channel_basic * (2**(i-5)), stride=stride,
                               test=False, downsample=use_downsample, fix_parameters=fix_parameters)

        with nn.parameter_scope(f'layer{i}_1'):
            h = resblock_align(h, channel_basic * (2**(i-5)),
                               stride=(1, 1), test=False, fix_parameters=fix_parameters)

        use_downsample = True
        stride = (2, 2)

    with nn.parameter_scope('mlp1'):
        h = F.relu(PF.affine(h, 128, with_bias=True,
                             fix_parameters=fix_parameters), inplace=True)
    with nn.parameter_scope('mlp3'):
        h = F.relu(PF.affine(h, 128, with_bias=True,
                             fix_parameters=fix_parameters), inplace=True)
    with nn.parameter_scope('mlp5'):
        h = PF.affine(h, 212, with_bias=True, fix_parameters=fix_parameters)

    return h


# ------------------------------
# Network for Encoder
# ------------------------------

def resblock_hg(x, in_channels, bottleneck, out_channels, batch_stat=True):
    # (bn --> relu --> conv) * 3
    with nn.parameter_scope('bn1'):
        h = PF.batch_normalization(x, batch_stat=batch_stat)
    h = F.relu(h, True)
    with nn.parameter_scope('conv1'):
        h = PF.convolution(h, bottleneck, kernel=(1, 1))

    with nn.parameter_scope('bn2'):
        h = PF.batch_normalization(h, batch_stat=batch_stat)
    h = F.relu(h, True)
    with nn.parameter_scope('conv2'):
        h = PF.convolution(h, bottleneck, kernel=(3, 3), pad=(1, 1))

    with nn.parameter_scope('bn3'):
        h = PF.batch_normalization(h, batch_stat=batch_stat)
    h = F.relu(h, True)
    with nn.parameter_scope('conv3'):
        h = PF.convolution(h, out_channels, kernel=(1, 1))

    if in_channels != out_channels:
        with nn.parameter_scope('downsample'):
            x = PF.convolution(x, out_channels, kernel=(1, 1))

    return x + h


def hourglass(x, planes, batch_stat=True):
    depth = 4  # hard-coded
    ResBlk = partial(resblock_hg,
                     in_channels=planes,
                     bottleneck=planes//2,
                     out_channels=planes,
                     batch_stat=batch_stat)  # set True

    ops = [[ResBlk, ResBlk, ResBlk, ResBlk],
           [ResBlk, ResBlk, ResBlk],
           [ResBlk, ResBlk, ResBlk],
           [ResBlk, ResBlk, ResBlk]]

    def hg_module(n, x):
        with nn.parameter_scope(f"{n - 1}.0.0"):
            up1 = ops[n - 1][0](x)
        low1 = F.max_pooling(x, kernel=(2, 2), stride=(2, 2))
        with nn.parameter_scope(f"{n - 1}.1.0"):
            low1 = ops[n - 1][1](low1)

        if n > 1:
            low2 = hg_module(n - 1, low1)
        else:
            with nn.parameter_scope(f"{n - 1}.3.0"):
                low2 = ops[n - 1][3](low1)
        with nn.parameter_scope(f"{n - 1}.2.0"):
            low3 = ops[n - 1][2](low2)

        up2 = F.interpolate(low3, scale=(2, 2), mode="nearest")

        out = up1 + up2
        return out

    return hg_module(depth, x)


def fc(x, planes, batch_stat=True):
    h = PF.convolution(x, planes, kernel=(1, 1))
    h = PF.batch_normalization(h, batch_stat=batch_stat)
    h = F.relu(h, True)
    return h


def stacked_hourglass_net(x,
                          batch_stat=True,
                          planes=64,
                          output_nc=15,
                          num_stacks=2,
                          activation='none'):
    with nn.parameter_scope('conv1'):
        x = PF.convolution(x, planes, kernel=(7, 7), pad=(3, 3), stride=(2, 2))
    with nn.parameter_scope('bn1'):
        x = PF.batch_normalization(x, batch_stat=batch_stat)
    x = F.relu(x, True)

    with nn.parameter_scope('layer1'):
        x = resblock_hg(x, planes, planes, planes*2, batch_stat=batch_stat)

    x = F.max_pooling(x, kernel=(2, 2), stride=(2, 2))

    with nn.parameter_scope('layer2'):
        x = resblock_hg(x, planes*2, planes*2, planes*4, batch_stat=batch_stat)

    with nn.parameter_scope('layer3'):
        x = resblock_hg(x, planes*4, planes*2, planes*4, batch_stat=batch_stat)

    planes = planes * 4
    scores = []
    for i in range(1, num_stacks):
        # applied only once
        with nn.parameter_scope(f'hourglass{i-1}'):
            y = hourglass(x, planes, batch_stat=batch_stat)
        with nn.parameter_scope('res0'):
            y = resblock_hg(y, planes, planes//2, planes,
                            batch_stat=batch_stat)
        with nn.parameter_scope('fc0'):
            y = fc(y, planes, batch_stat=batch_stat)  # True

        score = PF.convolution(y, output_nc, kernel=(1, 1), name='score0')
        score.persistent = True
        scores.append(score)

        fc_ = PF.convolution(y, planes, kernel=(1, 1), name='fc_')
        score_ = PF.convolution(score, planes, kernel=(1, 1), name='score_')
        x = x + fc_ + score_

    with nn.parameter_scope('hourglass1'):
        y = hourglass(x, planes, batch_stat=batch_stat)
    with nn.parameter_scope('res1'):
        y = resblock_hg(y, planes, planes//2, planes, batch_stat=batch_stat)
    with nn.parameter_scope('fc1'):
        y = fc(y, planes, batch_stat=batch_stat)  # mistakenly set as True

    score = PF.convolution(y, output_nc, kernel=(1, 1), name='score1')
    score.persistent = True
    scores.append(score)

    return scores
