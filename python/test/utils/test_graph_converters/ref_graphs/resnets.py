# Copyright 2018,2019,2020,2021 Sony Corporation.
# Copyright 2021,2022 Sony Group Corporation.
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

from .helper import (create_scale_bias,
                     create_conv_weight_bias,
                     create_affine_weight_bias,
                     get_channel_axes)


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
        axes = get_channel_axes(h, channel_last)
        a, b = create_scale_bias(1, h.shape, axes=axes)
        h = h * a + b
    return F.relu(h + x)


def small_bn_self_folding_resnet(image, channel_last=False, name='bn-self-folding-graph-ref'):
    h = image
    h /= 255.0
    h = PF.convolution(h, 16, kernel=(3, 3), pad=(1, 1), channel_last=channel_last,
                       with_bias=False, name='first-conv')
    axes = get_channel_axes(h, channel_last)
    a, b = create_scale_bias(1, h.shape, axes=axes)
    h = h * a + b
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
                test=False, w_bias=False, channel_last=False, name='convblock', dims=2):
    h = x
    kernel = (3,) * dims
    pad = (1,) * dims
    stride = (1,) * dims
    with nn.parameter_scope(name):
        h = PF.convolution(h, maps, kernel=kernel, pad=pad, stride=stride,
                           channel_last=channel_last, with_bias=w_bias)
        axes = get_channel_axes(h, channel_last, dims)
        h = PF.batch_normalization(h, axes=axes, batch_stat=not test)
    return F.relu(h + x)


def small_bn_resnet(image, test=False, w_bias=False, channel_last=False, name='bn-graph-ref', dims=2):
    kernel = (3,) * dims
    pool_kernel = (2,) * dims
    pad = (1,) * dims

    h = image
    h /= 255.0
    h = PF.convolution(h, 16, kernel=kernel, pad=pad, channel_last=channel_last,
                       with_bias=w_bias, name='first-conv')
    axes = get_channel_axes(h, channel_last, dims)
    h = PF.batch_normalization(
        h, axes=axes, batch_stat=not test, name='first-bn')
    h = F.relu(h)
    h = F.max_pooling(
        h, pool_kernel, channel_last=channel_last) if dims > 1 else h
    h = bn_resblock(h, maps=16, test=test, w_bias=w_bias,
                    channel_last=channel_last, name='cb1', dims=dims)
    h = bn_resblock(h, maps=16, test=test, w_bias=w_bias,
                    channel_last=channel_last, name='cb2', dims=dims)
    h = bn_resblock(h, maps=16, test=test, w_bias=w_bias,
                    channel_last=channel_last, name='cb3', dims=dims)
    h = bn_resblock(h, maps=16, test=test, w_bias=w_bias,
                    channel_last=channel_last, name='cb4', dims=dims)
    h = F.average_pooling(
        h, pool_kernel, channel_last=channel_last) if dims > 1 else h
    pred = PF.affine(h, 10, name='fc')
    return pred


# Small deconv network
def small_bn_dcn(image, test=False, w_bias=False, channel_last=False, name='small-bn-dcn', dims=2):
    h = image
    h /= 255.0
    kernel = (3,) * dims
    pool_kernel = (2,) * dims
    pad = (1,) * dims
    axes = get_channel_axes(channel_last, dims)
    with nn.parameter_scope('dcn-deconv1') as scope:
        h = PF.deconvolution(h, 16, kernel=kernel, pad=pad, channel_last=channel_last,
                             with_bias=w_bias, name='deconv1')
        h = PF.batch_normalization(
            h, axes=axes, batch_stat=not test, name='bn1')
        h = F.relu(h)
        h = F.max_pooling(
            h, pool_kernel, channel_last=channel_last) if dims > 1 else h
    with nn.parameter_scope('dcn-deconv2') as scope:
        h = PF.deconvolution(h, 16, kernel=kernel, pad=pad, channel_last=channel_last,
                             with_bias=w_bias, name='deconv2')
        h = PF.batch_normalization(
            h, axes=axes, batch_stat=not test, name='bn2')
        h = F.relu(h)
        h = F.max_pooling(
            h, pool_kernel, channel_last=channel_last) if dims > 1 else h
    with nn.parameter_scope('dcn-deconv3') as scope:
        h = PF.deconvolution(h, 16, kernel=kernel, pad=pad, channel_last=channel_last,
                             with_bias=w_bias, name='deconv3')
        h = PF.batch_normalization(
            h, axes=axes, batch_stat=not test, name='bn3')
        h = F.relu(h)
        h = F.max_pooling(
            h, pool_kernel, channel_last=channel_last) if dims > 1 else h
    pred = PF.affine(h, 10, name='dcn-fc')
    return pred


# Small deconv network
def small_dcn(image, test=False, w_bias=True, channel_last=False, name='small-dcn', dims=2):
    h = image
    h /= 255.0
    kernel = (3,) * dims
    pool_kernel = (2,) * dims
    pad = (1,) * dims
    with nn.parameter_scope('deconv1') as scope:
        h = PF.deconvolution(h, 16, kernel=kernel, pad=pad, channel_last=channel_last,
                             with_bias=w_bias, name='deconv1')
        h = F.relu(h)
        h = F.max_pooling(
            h, pool_kernel, channel_last=channel_last) if dims > 1 else h
    with nn.parameter_scope('deconv2') as scope:
        h = PF.deconvolution(h, 16, kernel=kernel, pad=pad, channel_last=channel_last,
                             with_bias=w_bias, name='deconv2')
        h = F.relu(h)
        h = F.max_pooling(
            h, pool_kernel, channel_last=channel_last) if dims > 1 else h
    with nn.parameter_scope('deconv3') as scope:
        h = PF.deconvolution(h, 16, kernel=kernel, pad=pad, channel_last=channel_last,
                             with_bias=w_bias, name='deconv3')
        h = F.relu(h)
        h = F.max_pooling(
            h, pool_kernel, channel_last=channel_last) if dims > 1 else h
    pred = PF.affine(h, 10, name='fc')
    return pred


# Small deconv network bn opp
def small_bn_opp_dcn(image, test=False, w_bias=False, channel_last=False, name='small-bn-opp-dcn', dims=2):
    h = image
    h /= 255.0
    kernel = (3,) * dims
    pool_kernel = (2,) * dims
    pad = (1,) * dims
    axes = get_channel_axes(channel_last, dims)
    with nn.parameter_scope('dcn-deconv1') as scope:
        h = PF.deconvolution(h, 16, kernel=kernel, pad=pad, channel_last=channel_last,
                             with_bias=w_bias, name='deconv1')
        h = F.relu(h)
        h = PF.batch_normalization(
            h, axes=axes, batch_stat=not test, name='bn1')
        h = PF.deconvolution(h, 16, kernel=kernel, pad=pad, channel_last=channel_last,
                             with_bias=w_bias, name='deconv2')
        h = F.max_pooling(
            h, pool_kernel, channel_last=channel_last) if dims > 1 else h
    with nn.parameter_scope('dcn-deconv2') as scope:
        h = PF.deconvolution(h, 16, kernel=kernel, pad=pad, channel_last=channel_last,
                             with_bias=w_bias, name='deconv3')
        h = F.relu(h)
        h = PF.batch_normalization(
            h, axes=axes, batch_stat=not test, name='bn2')
        h = PF.deconvolution(h, 16, kernel=kernel, pad=pad, channel_last=channel_last,
                             with_bias=w_bias, name='deconv4')
        h = F.max_pooling(
            h, pool_kernel, channel_last=channel_last) if dims > 1 else h
    with nn.parameter_scope('dcn-deconv3') as scope:
        h = PF.deconvolution(h, 16, kernel=kernel, pad=pad, channel_last=channel_last,
                             with_bias=w_bias, name='deconv5')
        h = F.relu(h)
        h = PF.batch_normalization(
            h, axes=axes, batch_stat=not test, name='bn3')
        h = PF.deconvolution(h, 16, kernel=kernel, pad=pad, channel_last=channel_last,
                             with_bias=w_bias, name='last-conv')
        h = F.max_pooling(
            h, pool_kernel, channel_last=channel_last) if dims > 1 else h
    pred = PF.affine(h, 10, name='dcn-fc')
    return pred


def small_opp_dcn(image, test=False, w_bias=True, channel_last=False, name='small-opp-dcn', dims=2):
    h = image
    h /= 255.0
    kernel = (3,) * dims
    pool_kernel = (2,) * dims
    pad = (1,) * dims
    with nn.parameter_scope('deconv1') as scope:
        h = PF.deconvolution(h, 16, kernel=kernel, pad=pad, channel_last=channel_last,
                             with_bias=w_bias, name='deconv1')
        h = F.relu(h)
        h = PF.deconvolution(h, 16, kernel=kernel, pad=pad, channel_last=channel_last,
                             with_bias=w_bias, name='deconv2')
        h = F.max_pooling(
            h, pool_kernel, channel_last=channel_last) if dims > 1 else h
    with nn.parameter_scope('deconv2') as scope:
        h = PF.deconvolution(h, 16, kernel=kernel, pad=pad, channel_last=channel_last,
                             with_bias=w_bias, name='deconv3')
        h = F.relu(h)
        h = PF.deconvolution(h, 16, kernel=kernel, pad=pad, channel_last=channel_last,
                             with_bias=w_bias, name='deconv4')
        h = F.max_pooling(
            h, pool_kernel, channel_last=channel_last) if dims > 1 else h
    with nn.parameter_scope('deconv3') as scope:
        h = PF.deconvolution(h, 16, kernel=kernel, pad=pad, channel_last=channel_last,
                             with_bias=w_bias, name='deconv5')
        h = F.relu(h)
        h = PF.deconvolution(h, 16, kernel=kernel, pad=pad, channel_last=channel_last,
                             with_bias=w_bias, name='last-conv')
        h = F.max_pooling(
            h, pool_kernel, channel_last=channel_last) if dims > 1 else h
    pred = PF.affine(h, 10, name='fc')
    return pred


# BatchNormalization Small ResNet Opposite
def bn_opp_resblock(x, maps, kernel=(3, 3), pad=(1, 1), stride=(1, 1),
                    test=False, channel_last=False, name='convblock', dims=2):
    axes = get_channel_axes(x, channel_last, dims)
    kernel = (3,) * dims
    pad = (1,) * dims
    stride = (1,) * dims

    with nn.parameter_scope(name):
        h = PF.batch_normalization(x, axes=axes, batch_stat=not test)
        z = h
        h = PF.convolution(h, maps, kernel=kernel, pad=pad, stride=stride,
                           channel_last=channel_last, with_bias=True)
    return F.relu(z + h)


def small_bn_opp_resnet(image, test=False, w_bias=False, channel_last=False, name='bn-graph-ref', dims=2):
    kernel = (3,) * dims
    pool_kernel = (2,) * dims
    pad = (1,) * dims

    h = image
    h /= 255.0
    axes = get_channel_axes(h, channel_last, dims)
    h = PF.batch_normalization(
        h, axes=axes, batch_stat=not test, name='first-bn')
    h = PF.convolution(h, 16, kernel=kernel, pad=pad, channel_last=channel_last,
                       with_bias=True, name='first-conv')
    h = F.relu(h)
    h = F.max_pooling(
        h, pool_kernel, channel_last=channel_last) if dims > 1 else h
    h = bn_opp_resblock(h, maps=16, test=test,
                        channel_last=channel_last, name='cb1', dims=dims)
    h = bn_opp_resblock(h, maps=16, test=test,
                        channel_last=channel_last, name='cb2', dims=dims)
    h = bn_opp_resblock(h, maps=16, test=test,
                        channel_last=channel_last, name='cb3', dims=dims)
    h = bn_opp_resblock(h, maps=16, test=test,
                        channel_last=channel_last, name='cb4', dims=dims)
    h = F.average_pooling(
        h, pool_kernel, channel_last=channel_last) if dims > 1 else h
    pred = PF.affine(h, 10, name='fc')
    return pred


# BatchNormalization Folding Small ResNet
def bn_folding_resblock(x, maps, kernel=(3, 3), pad=(1, 1), stride=(1, 1),
                        channel_last=False, name='convblock', dims=2):
    h = x
    kernel = (3,) * dims
    pad = (1,) * dims
    stride = (1,) * dims
    with nn.parameter_scope(name):
        h = PF.convolution(h, maps, kernel=kernel, pad=pad, stride=stride,
                           channel_last=channel_last, with_bias=True)
    return F.relu(h + x)


def small_bn_folding_resnet(image, channel_last=False, name='bn-graph-ref', dims=2):
    h = image
    h /= 255.0
    kernel = (3,) * dims
    pool_kernel = (2,) * dims
    pad = (1,) * dims
    h = PF.convolution(h, 16, kernel=kernel, pad=pad, channel_last=channel_last,
                       with_bias=True, name='first-conv')
    h = F.relu(h)
    h = F.max_pooling(
        h, pool_kernel, channel_last=channel_last) if dims > 1 else h
    h = bn_folding_resblock(h, maps=16,
                            channel_last=channel_last, name='cb1', dims=dims)
    h = bn_folding_resblock(h, maps=16,
                            channel_last=channel_last, name='cb2', dims=dims)
    h = bn_folding_resblock(h, maps=16,
                            channel_last=channel_last, name='cb3', dims=dims)
    h = bn_folding_resblock(h, maps=16,
                            channel_last=channel_last, name='cb4', dims=dims)
    h = F.average_pooling(
        h, pool_kernel, channel_last=channel_last) if dims > 1 else h
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
        h = PF.batch_normalization(h, batch_stat=False)
    return F.relu(h + x)


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


def reset_the_weight_value(inputs, output_axis, threshold):
    x, w = inputs[:2]
    shape = w.shape
    from functools import reduce
    items = reduce(lambda x, y: x * y, shape)
    upbound = (threshold / (items / shape[output_axis])) ** 0.5

    # some channels less than the upbound, some greater than the upbound
    slice0, slice1 = None, None
    if output_axis == 0:
        slice0 = '[:shape[0]//2,...]'
        slice1 = '[shape[0]//2:,...]'
    if output_axis == 1:
        slice0 = '[:,:shape[1]//2,...]'
        slice1 = '[:,shape[1]//2:,...]'
    if output_axis == -1:
        slice0 = '[...,:shape[-1]//2]'
        slice1 = '[...,shape[-1]//2:]'
    exec('w.d{} = upbound * 0.9'.format(slice0))
    exec('w.d{} = upbound * 2'.format(slice1))


def net_for_pruning(image, threshold, with_bias=False, channel_last=False, name_scope='net1'):
    with nn.parameter_scope(name_scope):
        h = image
        h /= 255.0
        h = PF.convolution(h, 16, kernel=(3, 3), pad=(1, 1),
                           with_bias=with_bias, channel_last=channel_last, name='conv')
        inputs = h.parent.inputs
        axis = 0
        # force reset the weight value
        reset_the_weight_value(inputs, axis, threshold)

        h = PF.deconvolution(h, 16, kernel=(
            3, 3), with_bias=with_bias, channel_last=channel_last, name='deconv')
        inputs = h.parent.inputs
        axis = -1 if channel_last else 1
        # force reset the weight value
        reset_the_weight_value(inputs, axis, threshold)

        pred = PF.affine(h, 10, name='fc')
        inputs = pred.parent.inputs
        axis = 1
        # force reset the weight value
        reset_the_weight_value(inputs, axis, threshold)

        return pred


def depthwise_net_for_pruning(image, threshold, with_bias=False, channel_last=False, name_scope='net1'):
    with nn.parameter_scope(name_scope):
        h = image
        h /= 255.0
        h = PF.convolution(h, 16, kernel=(3, 3), pad=(1, 1),
                           with_bias=False, channel_last=channel_last, name='conv')
        inputs = h.parent.inputs
        axis = 0
        # force reset the weight value
        reset_the_weight_value(inputs, axis, threshold)

        h = PF.depthwise_convolution(h, kernel=(
            3, 3), with_bias=with_bias, name='depthwise_conv')
        inputs = h.parent.inputs
        axis = 0
        # force reset the weight value
        reset_the_weight_value(inputs, axis, threshold)

        h = PF.depthwise_deconvolution(h, kernel=(
            3, 3), with_bias=with_bias, name='depthwise_deconv')
        inputs = h.parent.inputs
        axis = 0
        # force reset the weight value
        reset_the_weight_value(inputs, axis, threshold)

        pred = PF.affine(h, 10, name='fc')
        inputs = pred.parent.inputs
        axis = 1
        # force reset the weight value
        reset_the_weight_value(inputs, axis, threshold)

        return pred


# NonQNN to Recording Small ResNet
def nonqnn_to_recording_resblock(x, cfg, maps, kernel=(3, 3), pad=(1, 1), stride=(1, 1),
                                 test=False, channel_last=False, bn_self_folding=False,
                                 record_layers=(), name='convblock'):
    recorder = cfg.recorder_activation
    recorder_weight = cfg.recorder_weight
    axes = get_channel_axes(x, channel_last)

    with nn.parameter_scope(name):
        h = x

    with nn.parameter_scope('{}-conv'.format(name)):
        h = recorder()(h, axes=axes)
        hr1 = h
        w, b = create_conv_weight_bias(h, maps=maps, kernel=kernel,
                                       channel_last=channel_last, name=name)
        w = recorder_weight()(w, axes=axes, name='w')
        b = recorder_weight()(b, axes=axes, name='b')
        h = F.convolution(h, w, b, pad=pad, stride=stride,
                          channel_last=channel_last)
    if bn_self_folding:
        a, b = create_scale_bias(1, h.shape, axes=axes)

        with nn.parameter_scope('{}-Mul2'.format(name)):
            if not record_layers:
                hr2 = recorder()(h, axes=axes)
                h = hr2 * recorder()(a, axes=axes)
            else:
                h = h * a

        with nn.parameter_scope('{}-Add2'.format(name)):
            if not record_layers:
                hr3 = recorder()(h, axes=axes)
                h = hr3 + recorder()(b, axes=axes)
            else:
                h = h + b
    with nn.parameter_scope('{}/ReLU-Add2'.format(name)):
        if not record_layers:
            hr2 = recorder()(h, axes=axes)
        else:
            hr2 = h
        h = hr2 + hr1

    with nn.parameter_scope('{}-ReLU'.format(name)):
        if not record_layers:
            h = recorder()(h, axes=axes)
        else:
            h = h
    return F.relu(h)


def small_nonqnn_to_recording_resnet(image, config, test=False, channel_last=False,
                                     bn_self_folding=False, record_layers=(), name='bn-graph-ref'):
    recorder = config.recorder_activation
    recorder_weight = config.recorder_weight
    axes = get_channel_axes(image, channel_last)

    h = image
    func_id = 0
    with nn.parameter_scope('MulScale-{}'.format(func_id)):
        if not record_layers:
            h = recorder()(h, axes=axes)
        h /= 255.0
        func_id += 1
    with nn.parameter_scope('first-conv'):
        h = recorder()(h, axes=axes)
        w, b = create_conv_weight_bias(h, 16, kernel=(3, 3),
                                       channel_last=channel_last, name=name)
        w = recorder_weight()(w, axes=axes, name='w')
        b = recorder_weight()(b, axes=axes, name='b')
        h = F.convolution(h, w, b, pad=(1, 1), stride=(1, 1),
                          channel_last=channel_last)
    if bn_self_folding:
        a, b = create_scale_bias(1, h.shape, axes=axes)
        with nn.parameter_scope('Mul2-{}'.format(func_id)):
            if not record_layers:
                h = recorder()(h, axes=axes)
                h = h * recorder()(a, axes=axes)
            else:
                h = h * a
            func_id += 1
        with nn.parameter_scope('Add2-{}'.format(func_id)):
            if not record_layers:
                h = recorder()(h, axes=axes)
                h = h + recorder()(b, axes=axes)
            else:
                h = h + b
            func_id += 1

    with nn.parameter_scope('ReLU-{}'.format(func_id)):
        if not record_layers:
            h = recorder()(h, axes=axes)
        h = F.relu(h)
        func_id += 1
    with nn.parameter_scope('MaxPooling-{}'.format(func_id)):
        if not record_layers:
            h = recorder()(h, axes=axes)
        h = F.max_pooling(h, (2, 2), channel_last=channel_last)
        func_id += 1
    h = nonqnn_to_recording_resblock(h, config, maps=16, test=test,
                                     channel_last=channel_last, bn_self_folding=bn_self_folding,
                                     record_layers=record_layers, name='cb1')
    h = nonqnn_to_recording_resblock(h, config, maps=16, test=test,
                                     channel_last=channel_last, bn_self_folding=bn_self_folding,
                                     record_layers=record_layers, name='cb2')
    h = nonqnn_to_recording_resblock(h, config, maps=16, test=test,
                                     channel_last=channel_last, bn_self_folding=bn_self_folding,
                                     record_layers=record_layers, name='cb3')
    h = nonqnn_to_recording_resblock(h, config, maps=16, test=test,
                                     channel_last=channel_last, bn_self_folding=bn_self_folding,
                                     record_layers=record_layers, name='cb4')
    with nn.parameter_scope('AveragePooling-16'):
        if not record_layers:
            h = recorder()(h, axes=axes)
        h = F.average_pooling(h, (2, 2), channel_last=channel_last)
    with nn.parameter_scope('fc'):
        h = recorder()(h, axes=axes)
        w, b = create_affine_weight_bias(h, 10, name=name)
        w = recorder_weight()(w, axes=axes, name='w')
        b = recorder_weight()(b, axes=axes, name='b')
        pred = F.affine(h, w, b)
        return pred


# BatchNormalization Small ResNet
# For Recording to Training
def small_bn_r2t_resnet(image, test=False, w_bias=False, channel_last=False,
                        name='bn-graph-ref'):
    h = image
    h = PF.convolution(h, 16, kernel=(3, 3), pad=(1, 1), channel_last=channel_last,
                       with_bias=w_bias, name='first-conv')
    axes = get_channel_axes(h, channel_last)
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


def get_fake_quantization_parameter(shape, name):
    scale = nn.parameter.get_parameter_or_create(
        'scale-{}'.format(name), shape)
    scale.d = 1e-4
    zero_point = nn.parameter.get_parameter_or_create(
        'zeropoint-{}'.format(name), shape)
    zero_point.d = 0
    return scale, zero_point


# Recording to Training Small ResNet
def recording_to_training_resblock(x, cfg, maps, sx, zpx, kernel=(3, 3),
                                   pad=(1, 1), stride=(1, 1), test=False,
                                   channel_last=False, name='convblock'):
    axes = get_channel_axes(x, channel_last)
    rm = cfg.round_mode
    nr = cfg.narrow_range
    dt = cfg.dtype
    sw, zpw, sb, zpb = None, None, None, None
    get_scale_zeropoint = cfg.recorder_activation.get_scale_zeropoint

    h = x
    h1 = h
    with nn.parameter_scope(name):
        h1 = F.dequantize_linear(h, sx, zpx)

    with nn.parameter_scope('{}-conv'.format(name)):
        # Q->x->D
        h = h1
        w, b = create_conv_weight_bias(h, 16, kernel=kernel,
                                       channel_last=channel_last, name=name)

        # Q->w->D
        sw, zpw = get_scale_zeropoint(w, axes=axes, narrow_range=nr, name='sw')
        if sw is None:
            shape = [1] * w.ndim
            name = '{}-conv'.format(name) + '/' + 'sw'
            sw, zpw = get_fake_quantization_parameter(shape, name)
        w = F.quantize_linear(w, sw, zpw, rm, nr, dt)
        w = F.dequantize_linear(w, sw, zpw)

        # Q->b->D
        sb = sx.reshape([1]) * sw.reshape([1])
        sb = nn.Variable.from_numpy_array(sb.d)
        zpb = zpx.reshape([1])
        zpb = nn.Variable.from_numpy_array(zpb.d)
        b = F.quantize_linear(b, sb, zpb, rm, nr, dt)
        b = F.dequantize_linear(b, sb, zpb)

        # D->F->Q
        h = F.convolution(h, w, b, pad=pad, stride=stride,
                          channel_last=channel_last)
        sx, zpx = get_scale_zeropoint(h, axes=axes, narrow_range=nr, name='s')
        if sx is None:
            shape = [1] * h.ndim
            name = '{}-conv'.format(name) + '/' + 's'
            sx, zpx = get_fake_quantization_parameter(shape, name)
        h = F.quantize_linear(h, sx, zpx, rm, nr, dt)

    with nn.parameter_scope('{}-Add2'.format(name)):
        # Q->x->D
        h = F.dequantize_linear(h, sx, zpx)
        h = h1 + h
        # D->F->Q
        sx, zpx = get_scale_zeropoint(h, axes=axes, narrow_range=nr, name='s')
        if sx is None:
            shape = [1] * h.ndim
            name = '{}-Add2'.format(name) + '/' + 's'
            sx, zpx = get_fake_quantization_parameter(shape, name)
        h = F.quantize_linear(h, sx, zpx, rm, nr, dt)

    with nn.parameter_scope('{}-ReLU'.format(name)):
        # D->F->Q
        h = F.dequantize_linear(h, sx, zpx)
        h = F.relu(h)
        sx, zpx = get_scale_zeropoint(h, axes=axes, narrow_range=nr, name='s')
        if sx is None:
            shape = [1] * h.ndim
            name = '{}-ReLU'.format(name) + '/' + 's'
            sx, zpx = get_fake_quantization_parameter(shape, name)
        h = F.quantize_linear(h, sx, zpx, rm, nr, dt)

    return h, sx, zpx


def small_recording_to_training_resnet(image, config, test=False, channel_last=False,
                                       name='bn-graph-ref'):
    axes = get_channel_axes(image, channel_last)
    rm = config.round_mode
    nr = config.narrow_range
    dt = config.dtype
    sx, zpx, sw, zpw, sb, zpb = None, None, None, None, None, None
    get_scale_zeropoint = config.recorder_activation.get_scale_zeropoint

    h = image

    # x->Q
    sx, zpx = get_scale_zeropoint(h, axes=axes, narrow_range=nr, name='s')
    if sx is None:
        shape = [1] * h.ndim
        name = 's'
        sx, zpx = get_fake_quantization_parameter(shape, name)

    h = F.quantize_linear(h, sx, zpx, rm, nr, dt)

    with nn.parameter_scope('first-conv'):
        # Q->x->D
        h = F.dequantize_linear(h, sx, zpx)
        w, b = create_conv_weight_bias(h, 16, kernel=(3, 3),
                                       channel_last=channel_last, name=name)
        # Q->w->D
        sw, zpw = get_scale_zeropoint(w, axes=axes, narrow_range=nr, name='sw')
        if sw is None:
            shape = [1] * w.ndim
            name = 'sw'
            sw, zpw = get_fake_quantization_parameter(shape, name)

        w = F.quantize_linear(w, sw, zpw, rm, nr, dt)
        w = F.dequantize_linear(w, sw, zpw)

        # Q->b->D
        sb = sx.reshape([1]) * sw.reshape([1])
        sb = nn.Variable.from_numpy_array(sb.d)
        zpb = zpx.reshape([1])
        zpb = nn.Variable.from_numpy_array(zpb.d)
        b = F.quantize_linear(b, sb, zpb, rm, nr, dt)
        b = F.dequantize_linear(b, sb, zpb)

        # D->F->Q
        h = F.convolution(h, w, b, pad=(1, 1), stride=(1, 1),
                          channel_last=channel_last)
        sx, zpx = get_scale_zeropoint(h, axes=axes, narrow_range=nr, name='s')
        if sx is None:
            shape = [1] * h.ndim
            name = 's'
            sx, zpx = get_fake_quantization_parameter(shape, name)
        h = F.quantize_linear(h, sx, zpx, rm, nr, dt)
    with nn.parameter_scope('ReLU-2'):
        # D->F->Q
        h = F.dequantize_linear(h, sx, zpx)
        h = F.relu(h)
        sx, zpx = get_scale_zeropoint(h, axes=axes, narrow_range=nr, name='s')
        if sx is None:
            shape = [1] * h.ndim
            name = 'ReLU-2/s'
            sx, zpx = get_fake_quantization_parameter(shape, name)
        h = F.quantize_linear(h, sx, zpx, rm, nr, dt)
    with nn.parameter_scope('MaxPooling-3'):
        # D->F->Q
        h = F.dequantize_linear(h, sx, zpx)
        h = F.max_pooling(h, (2, 2), channel_last=channel_last)
        sx, zpx = get_scale_zeropoint(h, axes=axes, narrow_range=nr, name='s')
        if sx is None:
            shape = [1] * h.ndim
            name = 'MaxPooling-3/s'
            sx, zpx = get_fake_quantization_parameter(shape, name)
        h = F.quantize_linear(h, sx, zpx, rm, nr, dt)
    h, sx, zpx = recording_to_training_resblock(h, config, 16, sx, zpx, test=test,
                                                channel_last=channel_last, name='cb1')
    h, sx, zpx = recording_to_training_resblock(h, config, 16, sx, zpx, test=test,
                                                channel_last=channel_last, name='cb2')
    h, sx, zpx = recording_to_training_resblock(h, config, 16, sx, zpx, test=test,
                                                channel_last=channel_last, name='cb3')
    h, sx, zpx = recording_to_training_resblock(h, config, 16, sx, zpx, test=test,
                                                channel_last=channel_last, name='cb4')

    with nn.parameter_scope('AveragePooling-16'):
        # D->F->Q
        h = F.dequantize_linear(h, sx, zpx)
        h = F.average_pooling(h, (2, 2), channel_last=channel_last)
        sx, zpx = get_scale_zeropoint(h, axes=axes, narrow_range=nr, name='s')
        if sx is None:
            shape = [1] * h.ndim
            name = 'AveragePooling-16/s'
            sx, zpx = get_fake_quantization_parameter(shape, name)
        h = F.quantize_linear(h, sx, zpx, rm, nr, dt)
    with nn.parameter_scope('fc'):
        # Q->x->D
        h = F.dequantize_linear(h, sx, zpx)
        w, b = create_affine_weight_bias(h, 10, name=name)

        # Q->w->D
        sw, zpw = get_scale_zeropoint(w, axes=axes, narrow_range=nr, name='w')
        if sw is None:
            shape = [1] * w.ndim
            name = 'fc/sw'
            sw, zpw = get_fake_quantization_parameter(shape, name)
        w = F.quantize_linear(w, sw, zpw, rm, nr, dt)
        w = F.dequantize_linear(w, sw, zpw)

        # Q->b->D
        sb = sx.reshape([1]) * sw.reshape([1])
        sb = nn.Variable.from_numpy_array(sb.d)
        zpb = zpx.reshape([1])
        zpb = nn.Variable.from_numpy_array(zpb.d)
        b = F.quantize_linear(b, sb, zpb, rm, nr, dt)
        b = F.dequantize_linear(b, sb, zpb)

        # D->F->Q
        pred = F.affine(h, w, b)
        return pred


# NonQNN to Specific Recording Position Small ResNet (Convolution, Affine)
def nonqnn_to_specific_recording_pos_resblock(x, cfg, maps, kernel=(3, 3), pad=(1, 1), stride=(1, 1),
                                              test=False, channel_last=False,
                                              bn_self_folding=False, record_layers=(), name='convblock'):
    recorder = cfg.recorder_activation
    recorder_weight = cfg.recorder_weight
    axes = get_channel_axes(x, channel_last)

    with nn.parameter_scope(name):
        h = x

    with nn.parameter_scope('{}-conv'.format(name)):
        h = recorder()(h, axes=axes)
        hr1 = h
        w, b = create_conv_weight_bias(h, maps=maps, kernel=kernel,
                                       channel_last=channel_last, name=name)
        w = recorder_weight()(w, axes=axes, name='w')
        b = recorder_weight()(b, axes=axes, name='b')
        h = F.convolution(h, w, b, pad=pad, stride=stride,
                          channel_last=channel_last)
        h = recorder()(h, axes=axes)

        h = h + hr1

    return F.relu(h)


def small_nonqnn_to_specific_recording_pos_resnet(image, config, test=False, channel_last=False,
                                                  bn_self_folding=False, record_layers=(),
                                                  name='bn-graph-ref'):
    recorder = config.recorder_activation
    recorder_weight = config.recorder_weight
    axes = get_channel_axes(image, channel_last)

    h = image
    h /= 255.0

    with nn.parameter_scope('first-conv'):
        h = recorder()(h, axes=axes)
        w, b = create_conv_weight_bias(h, 16, kernel=(3, 3),
                                       channel_last=channel_last, name=name)
        w = recorder_weight()(w, axes=axes, name='w')
        b = recorder_weight()(b, axes=axes, name='b')
        h = F.convolution(h, w, b, pad=(1, 1), stride=(1, 1),
                          channel_last=channel_last)
        h = recorder()(h, axes=axes)
    h = F.relu(h)
    h = F.max_pooling(h, (2, 2), channel_last=channel_last)
    h = nonqnn_to_specific_recording_pos_resblock(h, config, maps=16, test=test,
                                                  channel_last=channel_last, name='cb1')
    h = nonqnn_to_specific_recording_pos_resblock(h, config, maps=16, test=test,
                                                  channel_last=channel_last, name='cb2')
    h = nonqnn_to_specific_recording_pos_resblock(h, config, maps=16, test=test,
                                                  channel_last=channel_last, name='cb3')
    h = nonqnn_to_specific_recording_pos_resblock(h, config, maps=16, test=test,
                                                  channel_last=channel_last, name='cb4')
    h = F.average_pooling(h, (2, 2), channel_last=channel_last)

    with nn.parameter_scope('fc'):
        h = recorder()(h, axes=axes)
        w, b = create_affine_weight_bias(h, 10, name=name)
        w = recorder_weight()(w, axes=axes, name='w')
        b = recorder_weight()(b, axes=axes, name='b')
        h = F.affine(h, w, b)
    with nn.parameter_scope('fc-rec'):
        pred = recorder()(h, axes=axes)

    return pred


def small_bn_fcn(image, test=False, w_bias=False, channel_last=False, name='bn-graph-ref'):
    h = image
    h /= 255.0
    h = PF.convolution(h, 16, kernel=(3, 3), pad=(1, 1), channel_last=channel_last,
                       with_bias=w_bias, name='first-conv')
    axes = get_channel_axes(h, channel_last)
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
    pred = PF.convolution(h, 16, kernel=(3, 3), pad=(1, 1), channel_last=channel_last,
                          with_bias=w_bias, name='last-conv')
    return pred


def small_nonqnn_to_recording_skip_conv_fcn(image, config, test=False, channel_last=False,
                                            bn_self_folding=False, record_layers=(),
                                            name='bn-graph-ref'):
    recorder = config.recorder_activation
    recorder_weight = config.recorder_weight
    axes = get_channel_axes(image, channel_last)

    h = image
    with nn.parameter_scope('MulScale-0'):
        h = recorder()(h, axes=axes)
        h /= 255.0
    with nn.parameter_scope('first-conv'):  # skip recording first conv
        w, b = create_conv_weight_bias(h, 16, kernel=(3, 3),
                                       channel_last=channel_last, name=name)
        h = F.convolution(h, w, b, pad=(1, 1), stride=(1, 1),
                          channel_last=channel_last)
    with nn.parameter_scope('ReLU-2'):
        h = recorder()(h, axes=axes)
        h = F.relu(h)
    with nn.parameter_scope('MaxPooling-3'):
        h = recorder()(h, axes=axes)
        h = F.max_pooling(h, (2, 2), channel_last=channel_last)
    h = nonqnn_to_recording_resblock(h, config, maps=16, test=test,
                                     channel_last=channel_last, name='cb1')
    h = nonqnn_to_recording_resblock(h, config, maps=16, test=test,
                                     channel_last=channel_last, name='cb2')
    h = nonqnn_to_recording_resblock(h, config, maps=16, test=test,
                                     channel_last=channel_last, name='cb3')
    h = nonqnn_to_recording_resblock(h, config, maps=16, test=test,
                                     channel_last=channel_last, name='cb4')
    with nn.parameter_scope('AveragePooling-16'):
        h = recorder()(h, axes=axes)
        h = F.average_pooling(h, (2, 2), channel_last=channel_last)
    with nn.parameter_scope('last-conv'):  # skip recording last conv
        w, b = create_conv_weight_bias(h, 16, kernel=(3, 3),
                                       channel_last=channel_last, name=name)
        pred = F.convolution(h, w, b, pad=(1, 1), stride=(1, 1),
                             channel_last=channel_last)
        return pred


def small_bn_multi_fc_resnet(image, test=False, w_bias=False, channel_last=False, name='bn-graph-ref'):
    h = image
    h /= 255.0
    h = PF.convolution(h, 16, kernel=(3, 3), pad=(1, 1), channel_last=channel_last,
                       with_bias=w_bias, name='first-conv')
    axes = get_channel_axes(h, channel_last)
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
    h = PF.affine(h, 256, name='fc1')
    h = PF.affine(h, 128, name='fc2')
    pred = PF.affine(h, 10, name='fc3')
    return pred


def small_nonqnn_to_recording_skip_affine_resnet(image, config, test=False, channel_last=False,
                                                 bn_self_folding=False, record_layers=(),
                                                 name='bn-graph-ref'):
    recorder = config.recorder_activation
    recorder_weight = config.recorder_weight
    axes = get_channel_axes(image, channel_last)

    h = image
    with nn.parameter_scope('MulScale-0'):
        h = recorder()(h, axes=axes)
        h /= 255.0
    with nn.parameter_scope('first-conv'):
        h = recorder()(h, axes=axes)
        w, b = create_conv_weight_bias(h, 16, kernel=(3, 3),
                                       channel_last=channel_last, name=name)
        w = recorder_weight()(w, axes=axes, name='w')
        b = recorder_weight()(b, axes=axes, name='b')
        h = F.convolution(h, w, b, pad=(1, 1), stride=(1, 1),
                          channel_last=channel_last)
    with nn.parameter_scope('ReLU-2'):
        h = recorder()(h, axes=axes)
        h = F.relu(h)
    with nn.parameter_scope('MaxPooling-3'):
        h = recorder()(h, axes=axes)
        h = F.max_pooling(h, (2, 2), channel_last=channel_last)
    h = nonqnn_to_recording_resblock(h, config, maps=16, test=test,
                                     channel_last=channel_last, name='cb1')
    h = nonqnn_to_recording_resblock(h, config, maps=16, test=test,
                                     channel_last=channel_last, name='cb2')
    h = nonqnn_to_recording_resblock(h, config, maps=16, test=test,
                                     channel_last=channel_last, name='cb3')
    h = nonqnn_to_recording_resblock(h, config, maps=16, test=test,
                                     channel_last=channel_last, name='cb4')
    with nn.parameter_scope('AveragePooling-16'):
        h = recorder()(h, axes=axes)
        h = F.average_pooling(h, (2, 2), channel_last=channel_last)
    with nn.parameter_scope('fc1'):  # skip recording first affine
        w, b = create_affine_weight_bias(h, 256, name=name)
        h = F.affine(h, w, b)
    with nn.parameter_scope('fc2'):
        h = recorder()(h, axes=axes)
        w, b = create_affine_weight_bias(h, 128, name=name)
        w = recorder_weight()(w, axes=axes, name='w')
        b = recorder_weight()(b, axes=axes, name='b')
        h = F.affine(h, w, b)
    with nn.parameter_scope('fc3'):  # skip recording last affine
        w, b = create_affine_weight_bias(h, 10, name=name)
        pred = F.affine(h, w, b)

        return pred
