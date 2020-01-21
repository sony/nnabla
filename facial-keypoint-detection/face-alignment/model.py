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
import nnabla.parametric_functions as PF
import nnabla.functions as F


def conv_block(x, in_planes, out_planes, test=True):
    residual = x
    out1 = PF.batch_normalization(x, batch_stat=not test, name='bn1')
    out1 = F.relu(out1, True)
    out1 = PF.convolution(out1, int(out_planes / 2), kernel=(3, 3),
                          stride=(1, 1), pad=(1, 1), name='conv1', with_bias=False)

    out2 = PF.batch_normalization(out1, batch_stat=not test, name='bn2')
    out2 = F.relu(out2, True)
    out2 = PF.convolution(out2, int(out_planes / 4), kernel=(3, 3),
                          stride=(1, 1), pad=(1, 1), name='conv2', with_bias=False)

    out3 = PF.batch_normalization(out2, batch_stat=not test, name='bn3')
    out3 = F.relu(out3, True)
    out3 = PF.convolution(out3, int(out_planes / 4), kernel=(3, 3),
                          stride=(1, 1), pad=(1, 1), name='conv3', with_bias=False)

    out3 = F.concatenate(out1, out2, out3, axis=1)

    if in_planes != out_planes:
        residual = PF.batch_normalization(
            residual, batch_stat=not test, name='downsample/0')
        residual = F.relu(residual, True)
        residual = PF.convolution(residual, out_planes, kernel=(
            1, 1), stride=(1, 1), name='downsample/2', with_bias=False)
    out3 += residual
    return out3


def hour_glass(inp, depth, num_features):
    # Upper branch
    up1 = inp
    with nn.parameter_scope('b1_' + str(depth)):
        up1 = conv_block(up1, num_features, num_features)

    # Lower branch
    low1 = F.average_pooling(inp, (2, 2), stride=(2, 2))
    with nn.parameter_scope('b2_' + str(depth)):
        low1 = conv_block(low1, num_features, num_features)

    if depth > 1:
        low2 = hour_glass(low1, depth - 1, num_features)
    else:
        low2 = low1
        with nn.parameter_scope('b2_plus_' + str(depth)):
            low2 = conv_block(low2, num_features, num_features)

    low3 = low2
    with nn.parameter_scope('b3_' + str(depth)):
        low3 = conv_block(low3, num_features, num_features)
    up2 = F.interpolate(low3, scale=(2, 2), mode='nearest')
    return up1 + up2


def fan(x, num_modules=1, test=True):
    x = PF.convolution(x, 64, kernel=(7, 7), stride=(2, 2),
                       pad=(3, 3), name='conv1')
    x = PF.batch_normalization(x, batch_stat=not test, name='bn1')
    x = F.relu(x, True)
    with nn.parameter_scope('conv2'):
        x = conv_block(x, 64, 128)
    x = F.average_pooling(x, (2, 2), stride=(2, 2))
    with nn.parameter_scope('conv3'):
        x = conv_block(x, 128, 128)
    with nn.parameter_scope('conv4'):
        x = conv_block(x, 128, 256)
    previous = x

    outputs = []
    for i in range(num_modules):
        with nn.parameter_scope('m' + str(i)):
            hg = hour_glass(previous, 4, 256)
        ll = hg
        with nn.parameter_scope('top_m_' + str(i)):
            ll = conv_block(ll, 256, 256)
        ll = PF.convolution(ll, 256, kernel=(1, 1), stride=(
            1, 1), pad=(0, 0), name='conv_last' + str(i))
        ll = PF.batch_normalization(
            ll, batch_stat=not test, name='bn_end' + str(i))
        ll = F.relu(ll, True)

        # Predict heatmaps
        tmp_out = PF.convolution(ll, 68, kernel=(
            1, 1), stride=(1, 1), pad=(0, 0), name='l' + str(i))
        outputs.append(tmp_out)

        if i < num_modules - 1:
            ll = PF.convolution(ll, 256, kernel=(1, 1), stride=(
                1, 1), pad=(0, 0), name='bl' + str(i))
            tmp_out_ = PF.convolution(tmp_out, 256, kernel=(
                1, 1), stride=(1, 1), pad=(0, 0), name='al' + str(i))
            previous = previous + ll + tmp_out_
    return outputs


def bottleneck(x, planes, stride=1, downsample=None, test=True):
    residual = x
    out = PF.convolution(x, planes, kernel=(
        1, 1), name='conv1', with_bias=False)
    out = PF.batch_normalization(out, batch_stat=not test, name='bn1')
    out = F.relu(out, True)
    out = PF.convolution(out, planes, kernel=(3, 3), stride=(
        stride, stride), pad=(1, 1), name='conv2', with_bias=False)
    out = PF.batch_normalization(out, batch_stat=not test, name='bn2')
    out = F.relu(out, True)
    out = PF.convolution(out, planes * 4, kernel=(1, 1),
                         name='conv3', with_bias=False)
    out = PF.batch_normalization(out, batch_stat=not test, name='bn3')

    if downsample is not None:
        residual = downsample
    out += residual
    out = F.relu(out, True)
    return out


def create_layer(x, inplanes, planes, blocks, stride=1, test=True):
    downsample = None
    dict = {64: 'layer1', 128: 'layer2', 256: 'layer3', 512: 'layer4'}
    with nn.parameter_scope(dict[planes]):
        with nn.parameter_scope('0'):
            if stride != 1 or inplanes != planes * 4:
                downsample = PF.convolution(x, planes * 4, kernel=(1, 1), stride=(stride, stride),
                                            name='downsample/0',
                                            with_bias=False)
                downsample = PF.batch_normalization(
                    downsample, batch_stat=not test, name='downsample/1')
            layers = bottleneck(x, planes, stride, downsample)
        for i in range(1, blocks):
            with nn.parameter_scope(str(i)):
                layers = bottleneck(layers, planes)
    return layers


def resnet_depth(x, layers=[3, 8, 36, 3], num_classes=68, test=True):
    inplanes = 64
    x = PF.convolution(x, 64, kernel=(7, 7), stride=(2, 2),
                       pad=(3, 3), name='conv1', with_bias=False)
    x = PF.batch_normalization(x, batch_stat=not test, name='bn1')
    x = F.relu(x, True)
    x = F.max_pooling(x, kernel=(3, 3), stride=(2, 2), pad=(1, 1))
    x = create_layer(x, inplanes, 64, layers[0])
    x = create_layer(x, inplanes, 128, layers[1], stride=2)
    x = create_layer(x, inplanes, 256, layers[2], stride=2)
    x = create_layer(x, inplanes, 512, layers[3], stride=2)
    x = F.average_pooling(x, kernel=(7, 7))
    x = F.reshape(x, (x.shape[0], -1))
    x = PF.affine(x, num_classes, name='fc')
    return x
