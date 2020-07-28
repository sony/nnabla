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

from .helper import create_scale_bias


# Small Channel First ResNet
def cf_resblock(x, maps, kernel=(3, 3), pad=(1, 1), stride=(1, 1), test=False, name='cf-convblock'):
    h = x
    with nn.parameter_scope(name):
        h = PF.convolution(h, maps, kernel=kernel, pad=pad,
                           stride=stride, with_bias=False)
        h = PF.batch_normalization(h, axes=[1], batch_stat=not test)
    return F.relu(h + x)


def small_cf_resnet(image, test=False):
    h = image
    h /= 255.0
    h = PF.convolution(h, 16, kernel=(3, 3), pad=(1, 1),
                       with_bias=False, name='first-cf-conv')
    h = PF.batch_normalization(
        h, axes=[1], batch_stat=not test, name='first-cf-bn')
    h = F.relu(h)
    h = F.max_pooling(h, (2, 2))
    h = cf_resblock(h, maps=16, test=test, name='cf-cb1')
    h = cf_resblock(h, maps=16, test=test, name='cf-cb2')
    h = cf_resblock(h, maps=16, test=test, name='cf-cb3')
    h = cf_resblock(h, maps=16, test=test, name='cf-cb4')
    h = F.average_pooling(h, (2, 2))
    pred = PF.affine(h, 10, name='cf-fc')
    return pred


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
def bn_self_folding_resblock(x, i, maps, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name='convblock'):
    h = x
    with nn.parameter_scope(name):
        h = PF.convolution(h, maps, kernel=kernel, pad=pad,
                           stride=stride, with_bias=False)
        a, b = create_scale_bias(i, h.shape[1])
        h = a * h + b
    return F.relu(h + x)


def small_bn_self_folding_resnet(image, name='bn-self-folding-graph-ref'):
    h = image
    h /= 255.0
    h = PF.convolution(h, 16, kernel=(3, 3), pad=(1, 1),
                       with_bias=False, name='first-conv')
    a, b = create_scale_bias(1, h.shape[1])
    h = a * h + b
    h = F.relu(h)
    h = F.max_pooling(h, (2, 2))
    h = bn_self_folding_resblock(h, 2, maps=16, name='cb1')
    h = bn_self_folding_resblock(h, 3, maps=16, name='cb2')
    h = bn_self_folding_resblock(h, 4, maps=16, name='cb3')
    h = bn_self_folding_resblock(h, 5, maps=16, name='cb4')
    h = F.average_pooling(h, (2, 2))
    pred = PF.affine(h, 10, name='fc')
    return pred


# BatchNormalization Small ResNet
def bn_resblock(x, maps, kernel=(3, 3), pad=(1, 1), stride=(1, 1), test=False, name='convblock'):
    with nn.parameter_scope(name):
        h = PF.convolution(x, maps, kernel=kernel, pad=pad,
                           stride=stride, with_bias=False)
        z = h
        h = PF.batch_normalization(h, batch_stat=not test)
    return F.relu(h + z)


def small_bn_resnet(image, test=False, w_bias=False, name='bn-graph-ref'):
    h = image
    h /= 255.0
    h = PF.convolution(h, 16, kernel=(3, 3), pad=(1, 1),
                       with_bias=w_bias, name='first-conv')
    h = PF.batch_normalization(h, batch_stat=not test, name='first-bn')
    h = F.relu(h)
    h = bn_resblock(h, maps=16, test=test, name='cb1')
    h = bn_resblock(h, maps=16, test=test, name='cb2')
    h = bn_resblock(h, maps=16, test=test, name='cb3')
    h = bn_resblock(h, maps=16, test=test, name='cb4')
    pred = PF.affine(h, 10, name='fc')
    return pred


# BatchNormalization folding Small ResNet
def bn_folding_resblock(x, maps, kernel=(3, 3), pad=(1, 1), stride=(1, 1), test=False, name='convblock'):
    with nn.parameter_scope(name):
        h = PF.convolution(x, maps, kernel=kernel, pad=pad,
                           stride=stride, with_bias=True)
        z = h
    return F.relu(h + z)


def small_bn_folding_resnet(image, test=False, name='bn-graph-ref'):
    h = image
    h /= 255.0
    h = PF.convolution(h, 16, kernel=(3, 3), pad=(1, 1),
                       with_bias=True, name='first-conv')
    h = F.relu(h)
    h = bn_folding_resblock(h, maps=16, test=test, name='cb1')
    h = bn_folding_resblock(h, maps=16, test=test, name='cb2')
    h = bn_folding_resblock(h, maps=16, test=test, name='cb3')
    h = bn_folding_resblock(h, maps=16, test=test, name='cb4')
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
    h = fbn_resblock(h, maps=16, test=test, name='fbn-cb1')
    h = fbn_resblock(h, maps=16, test=test, name='fbn-cb2')
    h = fbn_resblock(h, maps=16, test=test, name='fbn-cb3')
    h = fbn_resblock(h, maps=16, test=test, name='fbn-cb4')
    pred = PF.affine(h, 10, name='fbn-fc')
    return pred


# BatchNormalization Small ResNet removed functions
def bn_rm_resblock(x, maps, kernel=(3, 3), pad=(1, 1), stride=(1, 1), test=False, name='convblock'):
    with nn.parameter_scope(name):
        h = PF.convolution(x, maps, kernel=kernel, pad=pad,
                           stride=stride, with_bias=False)
        z = h
    return F.relu(h + z)


def small_bn_rm_resnet(image, test=False, w_bias=False, name='bn-rm-graph-ref'):
    h = image
    h = PF.convolution(h, 16, kernel=(3, 3), pad=(1, 1),
                       with_bias=w_bias, name='first-conv')
    h = F.relu(h)
    h = bn_rm_resblock(h, maps=16, test=test, name='cb1')
    h = bn_rm_resblock(h, maps=16, test=test, name='cb2')
    h = bn_rm_resblock(h, maps=16, test=test, name='cb3')
    h = bn_rm_resblock(h, maps=16, test=test, name='cb4')
    pred = PF.affine(h, 10, name='bn-rm-fc')
    return pred
