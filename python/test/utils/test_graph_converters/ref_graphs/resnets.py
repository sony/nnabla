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

from __future__ import absolute_import

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF

from .helper import create_scale_bias, get_channel_axes


# Small Channel First ResNet
def cf_resblock(x, maps, kernel=(3, 3), pad=(1, 1), stride=(1, 1),
                test=False, channel_last=False, name='cf-convblock'):
    axes = get_channel_axes(channel_last)
    h = x
    with nn.parameter_scope(name):
        h = PF.convolution(h, maps, kernel=kernel, pad=pad, stride=stride,
                           channel_last=channel_last, with_bias=False)
        h = PF.batch_normalization(h, axes=axes, batch_stat=not test)
    return F.relu(h + x)


def small_cf_resnet(image, test=False, channel_last=False):
    axes = get_channel_axes(channel_last)
    h = image
    h /= 255.0
    h = PF.convolution(h, 16, kernel=(3, 3), pad=(1, 1), channel_last=channel_last,
                       with_bias=False, name='first-cf-conv')
    h = PF.batch_normalization(
        h, axes=axes, batch_stat=not test, name='first-cf-bn')
    h = F.relu(h)
    h = F.max_pooling(h, (2, 2), channel_last=channel_last)
    h = cf_resblock(h, maps=16, test=test,
                    channel_last=channel_last, name='cf-cb1')
    h = cf_resblock(h, maps=16, test=test,
                    channel_last=channel_last, name='cf-cb2')
    h = cf_resblock(h, maps=16, test=test,
                    channel_last=channel_last, name='cf-cb3')
    h = cf_resblock(h, maps=16, test=test,
                    channel_last=channel_last, name='cf-cb4')
    h = F.average_pooling(h, (2, 2), channel_last=channel_last)
    pred = PF.affine(h, 10, name='cf-fc')
    return pred


# Small Channel Last ResNet
def cl_resblock(x, maps, kernel=(3, 3), pad=(1, 1), stride=(1, 1), test=False, name='cl_convblock'):
    h = x
    with nn.parameter_scope(name):
        h = PF.convolution(h, maps, kernel=kernel, pad=pad, stride=stride,
                           channel_last=True, with_bias=False)
        h = PF.batch_normalization(h, axes=[3], batch_stat=not test)
    return F.relu(h + x)


def small_cl_resnet(image, test=False):
    h = image
    h /= 255.0
    h = PF.convolution(h, 16, kernel=(3, 3), pad=(1, 1), channel_last=True,
                       with_bias=False, name='first-cl-conv')
    h = PF.batch_normalization(
        h, axes=[3], batch_stat=not test, name='first-cl-bn')
    h = F.relu(h)
    h = F.max_pooling(h, (2, 2), channel_last=True)
    h = cl_resblock(h, maps=16, test=test, name='cl-cb1')
    h = cl_resblock(h, maps=16, test=test, name='cl-cb2')
    h = cl_resblock(h, maps=16, test=test, name='cl-cb3')
    h = cl_resblock(h, maps=16, test=test, name='cl-cb4')
    h = F.average_pooling(h, (2, 2), channel_last=True)
    pred = PF.affine(h, 10, name='cl-fc')
    return pred


# BatchNormalization Self-folding Small ResNet
def bn_self_folding_resblock(x, i, maps, kernel=(3, 3), pad=(1, 1),
                             stride=(1, 1), channel_last=False, name='convblock'):
    h = x
    with nn.parameter_scope(name):
        h = PF.convolution(h, maps, kernel=kernel, pad=pad, stride=stride,
                           channel_last=channel_last, with_bias=False)
        axes = get_channel_axes(channel_last)
        a, b = create_scale_bias(1, h.shape, axes=axes)
        h = a * h + b
    return F.relu(h + x)


def small_bn_self_folding_resnet(image, channel_last=False, name='bn-self-folding-graph-ref'):
    h = image
    h /= 255.0
    h = PF.convolution(h, 16, kernel=(3, 3), pad=(1, 1), channel_last=channel_last,
                       with_bias=False, name='first-conv')
    axes = get_channel_axes(channel_last)
    a, b = create_scale_bias(1, h.shape, axes=axes)
    h = a * h + b
    h = F.relu(h)
    h = F.max_pooling(h, (2, 2), channel_last=channel_last)
    h = bn_self_folding_resblock(
        h, 2, maps=16, channel_last=channel_last, name='cb1')
    h = bn_self_folding_resblock(
        h, 3, maps=16, channel_last=channel_last, name='cb2')
    h = bn_self_folding_resblock(
        h, 4, maps=16, channel_last=channel_last, name='cb3')
    h = bn_self_folding_resblock(
        h, 5, maps=16, channel_last=channel_last, name='cb4')
    h = F.average_pooling(h, (2, 2), channel_last=channel_last)
    pred = PF.affine(h, 10, name='fc')
    return pred


# BatchNormalization Small ResNet
def bn_resblock(x, maps, kernel=(3, 3), pad=(1, 1), stride=(1, 1),
                test=False, w_bias=False, channel_last=False, name='convblock'):
    axes = get_channel_axes(channel_last)
    h = x
    with nn.parameter_scope(name):
        h = PF.convolution(h, maps, kernel=kernel, pad=pad, stride=stride,
                           channel_last=channel_last, with_bias=w_bias)
        h = PF.batch_normalization(h, axes=axes, batch_stat=not test)
    return F.relu(h + x)


def small_bn_resnet(image, test=False, w_bias=False, channel_last=False, name='bn-graph-ref'):
    axes = get_channel_axes(channel_last)

    h = image
    h /= 255.0
    h = PF.convolution(h, 16, kernel=(3, 3), pad=(1, 1), channel_last=channel_last,
                       with_bias=w_bias, name='first-conv')
    h = PF.batch_normalization(
        h, axes=axes, batch_stat=not test, name='first-bn')
    h = F.relu(h)
    h = F.max_pooling(h, (2, 2), channel_last=channel_last)
    h = bn_resblock(h, maps=16, test=test, w_bias=w_bias,
                    channel_last=channel_last, name='cb1')
    h = bn_resblock(h, maps=16, test=test, w_bias=w_bias,
                    channel_last=channel_last, name='cb2')
    h = bn_resblock(h, maps=16, test=test, w_bias=w_bias,
                    channel_last=channel_last, name='cb3')
    h = bn_resblock(h, maps=16, test=test, w_bias=w_bias,
                    channel_last=channel_last, name='cb4')
    h = F.average_pooling(h, (2, 2), channel_last=channel_last)
    pred = PF.affine(h, 10, name='fc')
    return pred


# BatchNormalization Small ResNet Opposite
def bn_opp_resblock(x, maps, kernel=(3, 3), pad=(1, 1), stride=(1, 1),
                    test=False, channel_last=False, name='convblock'):
    axes = get_channel_axes(channel_last)
    with nn.parameter_scope(name):
        h = PF.batch_normalization(x, axes=axes, batch_stat=not test)
        z = h
        h = PF.convolution(h, maps, kernel=kernel, pad=pad, stride=stride,
                           channel_last=channel_last, with_bias=True)
    return F.relu(z + h)


def small_bn_opp_resnet(image, test=False, w_bias=False, channel_last=False, name='bn-graph-ref'):
    axes = get_channel_axes(channel_last)

    h = image
    h /= 255.0
    h = PF.batch_normalization(
        h, axes=axes, batch_stat=not test, name='first-bn')
    h = PF.convolution(h, 16, kernel=(3, 3), pad=(1, 1), channel_last=channel_last,
                       with_bias=w_bias, name='first-conv')
    h = F.relu(h)
    h = F.max_pooling(h, (2, 2), channel_last=channel_last)
    h = bn_opp_resblock(h, maps=16, test=test,
                        channel_last=channel_last, name='cb1')
    h = bn_opp_resblock(h, maps=16, test=test,
                        channel_last=channel_last, name='cb2')
    h = bn_opp_resblock(h, maps=16, test=test,
                        channel_last=channel_last, name='cb3')
    h = bn_opp_resblock(h, maps=16, test=test,
                        channel_last=channel_last, name='cb4')
    h = F.average_pooling(h, (2, 2), channel_last=channel_last)
    pred = PF.affine(h, 10, name='fc')
    return pred


# BatchNormalization Folding Small ResNet
def bn_folding_resblock(x, maps, kernel=(3, 3), pad=(1, 1), stride=(1, 1),
                        test=False, channel_last=False, name='convblock'):
    h = x
    with nn.parameter_scope(name):
        h = PF.convolution(h, maps, kernel=kernel, pad=pad, stride=stride,
                           channel_last=channel_last, with_bias=True)
    return F.relu(h + x)


def small_bn_folding_resnet(image, test=False, channel_last=False, name='bn-graph-ref'):
    h = image
    h /= 255.0
    h = PF.convolution(h, 16, kernel=(3, 3), pad=(1, 1), channel_last=channel_last,
                       with_bias=True, name='first-conv')
    h = F.relu(h)
    h = F.max_pooling(h, (2, 2), channel_last=channel_last)
    h = bn_folding_resblock(h, maps=16, test=test,
                            channel_last=channel_last, name='cb1')
    h = bn_folding_resblock(h, maps=16, test=test,
                            channel_last=channel_last, name='cb2')
    h = bn_folding_resblock(h, maps=16, test=test,
                            channel_last=channel_last, name='cb3')
    h = bn_folding_resblock(h, maps=16, test=test,
                            channel_last=channel_last, name='cb4')
    h = F.average_pooling(h, (2, 2), channel_last=channel_last)
    pred = PF.affine(h, 10, name='fc')
    return pred


# FusedBatchNormalization Small ResNet
def fbn_resblock(x, maps, kernel=(3, 3), pad=(1, 1), stride=(1, 1), test=False, name='fbn-convblock'):
    with nn.parameter_scope(name):
        h = PF.convolution(x, maps, kernel=kernel, pad=pad,
                           stride=stride, with_bias=False)
        h = PF.fused_batch_normalization(h, x, batch_stat=not test)
    return h


def small_fbn_resnet(image, test=False, name='fbn-graph-ref'):
    h = image
    h /= 255.0
    h = PF.convolution(h, 16, kernel=(3, 3), pad=(1, 1),
                       with_bias=False, name='first-fbn-conv')
    h = PF.batch_normalization(h, batch_stat=not test, name='first-fbn')
    h = F.relu(h)
    h = F.max_pooling(h, (2, 2))
    h = fbn_resblock(h, maps=16, test=test, name='fbn-cb1')
    h = fbn_resblock(h, maps=16, test=test, name='fbn-cb2')
    h = fbn_resblock(h, maps=16, test=test, name='fbn-cb3')
    h = fbn_resblock(h, maps=16, test=test, name='fbn-cb4')
    h = F.average_pooling(h, (2, 2))
    pred = PF.affine(h, 10, name='fbn-fc')
    return pred


# BatchNormalization Small ResNet removed functions
def bn_rm_resblock(x, maps, kernel=(3, 3), pad=(1, 1), stride=(1, 1),
                   test=False, w_bias=False, name='convblock'):
    h = x
    with nn.parameter_scope(name):
        h = PF.convolution(h, maps, kernel=kernel, pad=pad,
                           stride=stride, with_bias=w_bias)
    return F.relu(h + x)


def small_bn_rm_resnet(image, test=False, w_bias=False, name='bn-rm-graph-ref'):
    h = image
    h = PF.convolution(h, 16, kernel=(3, 3), pad=(1, 1),
                       with_bias=w_bias, name='first-conv')
    h = F.relu(h)
    h = F.max_pooling(h, (2, 2))
    h = bn_rm_resblock(h, maps=16, test=test, w_bias=w_bias, name='cb1')
    h = bn_rm_resblock(h, maps=16, test=test, w_bias=w_bias, name='cb2')
    h = bn_rm_resblock(h, maps=16, test=test, w_bias=w_bias, name='cb3')
    h = bn_rm_resblock(h, maps=16, test=test, w_bias=w_bias, name='cb4')
    h = F.average_pooling(h, (2, 2))
    pred = PF.affine(h, 10, name='bn-rm-fc')
    return pred


# BatchNormalization Small ResNet with batch_stat False
def bsf_resblock(x, maps, kernel=(3, 3), pad=(1, 1), stride=(1, 1),
                 test=False, w_bias=False, name='convblock'):
    with nn.parameter_scope(name):
        h = PF.convolution(x, maps, kernel=kernel, pad=pad,
                           stride=stride, with_bias=w_bias)
        z = h
        h = PF.batch_normalization(h, batch_stat=False)
    return F.relu(h + z)


def small_bsf_resnet(image, w_bias=False, name='bn-graph-ref'):
    h = image
    h /= 255.0
    h = PF.convolution(h, 16, kernel=(3, 3), pad=(1, 1),
                       with_bias=w_bias, name='first-conv')
    h = PF.batch_normalization(h, batch_stat=False, name='first-bn-bsf')
    h = F.relu(h)
    h = F.max_pooling(h, (2, 2))
    h = bsf_resblock(h, maps=16, test=False, w_bias=w_bias, name='cb1')
    h = bsf_resblock(h, maps=16, test=False, w_bias=w_bias, name='cb2')
    h = bsf_resblock(h, maps=16, test=False, w_bias=w_bias, name='cb3')
    h = bsf_resblock(h, maps=16, test=False, w_bias=w_bias, name='cb4')
    h = F.average_pooling(h, (2, 2))
    pred = PF.affine(h, 10, name='fc')
    return pred


# Small BatchNormalization Multiple Inputs/Outputs ResNet
def multiple_inputs_outputs_resblock(x, maps, kernel=(3, 3), pad=(1, 1), stride=(1, 1),
                                     w_bias=False, test=False, name='mo-convblock'):
    h = x
    with nn.parameter_scope(name):
        h = PF.convolution(h, maps, kernel=kernel, pad=pad,
                           stride=stride, with_bias=w_bias)
        h = PF.batch_normalization(h, axes=[1], batch_stat=not test)
    return F.relu(h + x)


def small_multiple_inputs_outputs_resnet(images, test=False, w_bias=False):
    # Branches
    outputs = []
    for i, image in enumerate(images):
        h = image
        h /= 255.0
        h = PF.convolution(h, 16, kernel=(3, 3), pad=(1, 1),
                           with_bias=w_bias, name='first-mo-conv-{}'.format(i))
        h = PF.batch_normalization(
            h, axes=[1], batch_stat=not test, name='first-mo-bn-{}'.format(i))
        h = F.relu(h)
        h = F.max_pooling(h, (2, 2))
        outputs.append(h)
    # Merge branches
    z = sum(outputs)

    h = multiple_inputs_outputs_resblock(
        z, maps=16, w_bias=w_bias, test=test, name='mo-cb1')
    h = F.average_pooling(h, (2, 2))
    pred1 = PF.affine(h, 10, name='mo-fc1')

    h = multiple_inputs_outputs_resblock(
        z, maps=16, w_bias=w_bias, test=test, name='mo-cb2')
    h = F.average_pooling(h, (2, 2))
    pred2 = PF.affine(h, 10, name='mo-fc2')
    return [pred1, pred2]


# Small BatchNormalization Folding Multiple Inputs/Outputs ResNet
def multiple_inputs_outputs_bn_folding_resblock(x, maps, kernel=(3, 3), pad=(1, 1),
                                                stride=(1, 1), test=False, name='mo-convblock'):
    h = x
    with nn.parameter_scope(name):
        h = PF.convolution(h, maps, kernel=kernel, pad=pad,
                           stride=stride, with_bias=True)
    return F.relu(h + x)


def small_multiple_inputs_outputs_bn_folding_resnet(images, test=False):
    # Branches
    outputs = []
    for i, image in enumerate(images):
        h = image
        h /= 255.0
        h = PF.convolution(h, 16, kernel=(3, 3), pad=(1, 1),
                           with_bias=True, name='first-mo-conv-{}'.format(i))
        h = F.relu(h)
        h = F.max_pooling(h, (2, 2))
        outputs.append(h)
    # Merge branches
    z = sum(outputs)

    h = multiple_inputs_outputs_bn_folding_resblock(
        z, maps=16, test=test, name='mo-cb1')
    h = F.average_pooling(h, (2, 2))
    pred1 = PF.affine(h, 10, name='mo-fc1')

    h = multiple_inputs_outputs_bn_folding_resblock(
        z, maps=16, test=test, name='mo-cb2')
    h = F.average_pooling(h, (2, 2))
    pred2 = PF.affine(h, 10, name='mo-fc2')
    return [pred1, pred2]


# ChannelLast BatchNormalization Small ResNet
def clbn_resblock(x, maps, kernel=(3, 3), pad=(1, 1), stride=(1, 1), test=False, bias_w=False, name='clbn-convblock'):
    h = x
    with nn.parameter_scope(name):
        h = PF.convolution(x, maps, kernel=kernel, pad=pad, stride=stride,
                           channel_last=True, with_bias=bias_w)
        z = h
        h = PF.batch_normalization(h, axes=[3], batch_stat=not test)
    return F.relu(h + z)


def small_clbn_resnet(image, test=False, w_bias=False):
    h = image
    h /= 255.0
    h = PF.convolution(h, 16, kernel=(3, 3), pad=(1, 1), channel_last=True,
                       with_bias=w_bias, name='first-clbn-conv')
    h = PF.batch_normalization(
        h, axes=[3], batch_stat=not test, name='first-clbn-bn')
    h = F.relu(h)
    h = F.max_pooling(h, (2, 2), channel_last=True)
    h = clbn_resblock(h, maps=16, test=test, bias_w=w_bias, name='clbn-cb1')
    h = clbn_resblock(h, maps=16, test=test, bias_w=w_bias, name='clbn-cb2')
    h = clbn_resblock(h, maps=16, test=test, bias_w=w_bias, name='clbn-cb3')
    h = clbn_resblock(h, maps=16, test=test, bias_w=w_bias, name='clbn-cb4')
    h = F.average_pooling(h, (2, 2), channel_last=True)
    pred = PF.affine(h, 10, name='clbn-fc')
    return pred


# ChannelLast BatchNormalization Folding Small ResNet
def clbn_folding_resblock(x, maps, kernel=(3, 3), pad=(1, 1), stride=(1, 1), test=False, name='clbn-convblock'):
    h = x
    with nn.parameter_scope(name):
        h = PF.convolution(x, maps, kernel=kernel, pad=pad, stride=stride,
                           channel_last=True, with_bias=True)
        z = h
    return F.relu(h + z)


def small_clbn_folding_resnet(image, test=False):
    h = image
    h /= 255.0
    h = PF.convolution(h, 16, kernel=(3, 3), pad=(1, 1), channel_last=True,
                       with_bias=True, name='first-clbn-conv')
    h = F.relu(h)
    h = F.max_pooling(h, (2, 2), channel_last=True)
    h = clbn_folding_resblock(h, maps=16, test=test, name='clbn-cb1')
    h = clbn_folding_resblock(h, maps=16, test=test, name='clbn-cb2')
    h = clbn_folding_resblock(h, maps=16, test=test, name='clbn-cb3')
    h = clbn_folding_resblock(h, maps=16, test=test, name='clbn-cb4')
    h = F.average_pooling(h, (2, 2), channel_last=True)
    pred = PF.affine(h, 10, name='clbn-fc')
    return pred


# ChannelLast BatchNormalization Self-folding Small ResNet
def clbn_self_folding_resblock(x, i, maps, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name='convblock'):
    h = x
    with nn.parameter_scope(name):
        h = PF.convolution(h, maps, kernel=kernel, pad=pad, stride=stride,
                           channel_last=True, with_bias=False)
        a, b = create_scale_bias(i, h.shape[3], axes=[3])
        h = a * h + b
    return F.relu(h + x)


def small_clbn_self_folding_resnet(image, name='clbn-self-folding-graph-ref'):
    h = image
    h /= 255.0
    h = PF.convolution(h, 16, kernel=(3, 3), pad=(1, 1), channel_last=True,
                       with_bias=False, name='first-conv')
    a, b = create_scale_bias(1, h.shape[3], axes=[3])
    h = a * h + b
    h = F.relu(h)
    h = F.max_pooling(h, (2, 2), channel_last=True)
    h = clbn_self_folding_resblock(h, 2, maps=16, name='cb1')
    h = clbn_self_folding_resblock(h, 3, maps=16, name='cb2')
    h = clbn_self_folding_resblock(h, 4, maps=16, name='cb3')
    h = clbn_self_folding_resblock(h, 5, maps=16, name='cb4')
    h = F.average_pooling(h, (2, 2), channel_last=True)
    pred = PF.affine(h, 10, name='fc')
    return pred


# Small Identity ResNet
def id_resblock(x, maps, kernel=(3, 3), pad=(1, 1), stride=(1, 1), test=False, name='id-convblock'):
    h = x
    with nn.parameter_scope(name):
        h = PF.convolution(h, maps, kernel=kernel, pad=pad,
                           stride=stride, with_bias=False)
        h = PF.batch_normalization(h, axes=[1], batch_stat=not test)
    return F.relu(h + x)


def small_id_resnet(image, test=False):
    h = image
    h /= 255.0
    h = PF.convolution(h, 16, kernel=(3, 3), pad=(1, 1),
                       with_bias=False, name='first-id-conv')
    h = PF.batch_normalization(
        h, axes=[1], batch_stat=not test, name='first-id-bn')
    h = F.relu(h)
    h = F.max_pooling(h, (2, 2))
    h = id_resblock(h, maps=16, test=test, name='id-cb1')
    h = id_resblock(h, maps=16, test=test, name='id-cb2')
    h = id_resblock(h, maps=16, test=test, name='id-cb3')
    h = id_resblock(h, maps=16, test=test, name='id-cb4')
    h = F.average_pooling(h, (2, 2))
    pred = PF.affine(h, 10, name='id-fc')
    return pred
