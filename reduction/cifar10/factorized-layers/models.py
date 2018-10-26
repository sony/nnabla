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

from six.moves import range

import os

import nnabla as nn
import nnabla.logger as logger
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import nnabla.utils.save as save
from nnabla.parameter import get_parameter, get_parameter_or_create, set_parameter
import numpy as np


def categorical_error(pred, label):
    """
    Compute categorical error given score vectors and labels as
    numpy.ndarray.
    """
    pred_label = pred.argmax(1)
    return (pred_label != label.flat).mean()


def cifar10_resnet23_prediction(image, maps=64,
                                test=False):
    """
    Construct Resnet23 as reference.
    """
    # Residual Unit
    def res_unit(x, scope_name, dn=False):
        C = x.shape[1]
        with nn.parameter_scope(scope_name):

            # Conv -> BN -> Relu
            with nn.parameter_scope("conv1"):
                h = PF.convolution(x, C // 2, kernel=(1, 1), pad=(0, 0),
                                   with_bias=False)
                h = PF.batch_normalization(h, batch_stat=not test)
                h = F.relu(h)
            # Conv -> BN -> Relu
            with nn.parameter_scope("conv2"):
                h = PF.convolution(h, C // 2, kernel=(3, 3), pad=(1, 1),
                                   with_bias=False)
                h = PF.batch_normalization(h, batch_stat=not test)
                h = F.relu(h)
            # Conv -> BN
            with nn.parameter_scope("conv3"):
                h = PF.convolution(h, C, kernel=(1, 1), pad=(0, 0),
                                   with_bias=False)
                h = PF.batch_normalization(h, batch_stat=not test)
            # Residual -> Relu
            h = F.relu(h + x)

            # Maxpooling
            if dn:
                h = F.max_pooling(h, kernel=(2, 2), stride=(2, 2))

        return h

    ncls = 10

    # Conv -> BN -> Relu
    with nn.parameter_scope("conv1"):
        # Preprocess
        image /= 255.0
        if not test:
            image = F.image_augmentation(image, contrast=1.0,
                                         angle=0.25,
                                         flip_lr=True)
            image.need_grad = False
        h = PF.convolution(image, maps, kernel=(3, 3), pad=(1, 1),
                           with_bias=False)
        h = PF.batch_normalization(h, batch_stat=not test)
        h = F.relu(h)

    h = res_unit(h, "conv2", False)    # -> 32x32
    h = res_unit(h, "conv3", True)     # -> 16x16
    h = res_unit(h, "conv4", False)    # -> 16x16
    h = res_unit(h, "conv5", True)     # -> 8x8
    h = res_unit(h, "conv6", False)    # -> 8x8
    h = res_unit(h, "conv7", True)     # -> 4x4
    h = res_unit(h, "conv8", False)    # -> 4x4
    h = F.average_pooling(h, kernel=(4, 4))  # -> 1x1
    pred = PF.affine(h, ncls)

    return pred


def cifar10_cpd3_factorized_resnet23_prediction(image, maps=64,
                                                test=False,
                                                compression_ratio=0.0,
                                                lambda_reg=0.0):
    """
    Construct Resnet23 with factorized affine and convolution
    """
    # SVD affine
    def svd_affine(x, n_outputs, cr):
        W = get_parameter('affine/W')

        if W is None:
            UV = None
        else:
            UV = W.d
        b = get_parameter('affine/b')
        # compute rank (size of intermediate activations)
        # to obtained desired reduction
        inshape = np.prod(x.shape[1:])
        outshape = np.prod(n_outputs)
        rank = int(np.floor((1-cr)*inshape*outshape/(inshape+outshape)))

        # Initialize bias to existing b in affine if exists
        b_new =  get_parameter('svd_affine/b')
        if b is not None and b_new is None:
            b_new = get_parameter_or_create(
                'svd_affine/b', b.d.shape, need_grad=b.need_grad)
            b_new.d = b.d.copy()
        logger.info("SVD affine created: input_shape = {}; output_shape = {}; compression = {}; rank = {};".format(
            inshape, outshape, cr, rank))

        # create svd_affine initialized from W in current context if it exists
        return PF.svd_affine(x, n_outputs, rank, uv_init=UV)

    # CP convolution
    def cpd3_convolution(x, n_outputs, kernel, pad, with_bias, cr, lambda_reg):
        W = get_parameter('conv/W')

        if W is None:
            OIK = None
        else:
            OIK = W.d
        b = get_parameter('conv/b')
        # compute rank (size of intermediate activations)
        # to obtained desired reduction
        inmaps = x.shape[1]
        outmaps = n_outputs
        Ksize = np.prod(kernel)
        rank = int(np.floor((1-cr)*inmaps*outmaps *
                            Ksize/(inmaps+outmaps+Ksize)))

        # Initialize bias to existing b in affine if exists
        b_new =  get_parameter('cpd3_conv/b')
        if b is not None and b_new is None:
            b_new = get_parameter_or_create(
                'cpd3_conv/b', b.d.shape, need_grad=b.need_grad)
            b_new.d = b.d.copy()
        logger.info("CP convolution created: inmaps = {}; outmaps = {}; compression = {}; rank = {}; lambda_reg = {};".format(
            inmaps, outmaps, cr, rank, lambda_reg))

        # create cpd3_convolution  initialized from W in current context if it exists
        return PF.cpd3_convolution(x, n_outputs, kernel=kernel, r=rank, pad=pad,
                                   with_bias=with_bias, oik_init=OIK,
                                   lambda_reg=lambda_reg)

    # Residual Unit
    def res_unit(x, scope_name, dn=False):
        C = x.shape[1]
        with nn.parameter_scope(scope_name):

            # Conv -> BN -> Relu
            with nn.parameter_scope("conv1"):
                h = PF.convolution(x, C // 2, kernel=(1, 1), pad=(0, 0),
                                   with_bias=False)
                h = PF.batch_normalization(h, batch_stat=not test)
                h = F.relu(h)
            # Conv -> BN -> Relu
            with nn.parameter_scope("conv2"):
                h = cpd3_convolution(h, C // 2, kernel=(3, 3), pad=(1, 1),
                                     with_bias=False, cr=compression_ratio,
                                     lambda_reg=lambda_reg)
                h = PF.batch_normalization(h, batch_stat=not test)
                h = F.relu(h)
            # Conv -> BN
            with nn.parameter_scope("conv3"):
                h = PF.convolution(h, C, kernel=(1, 1), pad=(0, 0),
                                   with_bias=False)
                h = PF.batch_normalization(h, batch_stat=not test)
            # Residual -> Relu
            h = F.relu(h + x)

            # Maxpooling
            if dn:
                h = F.max_pooling(h, kernel=(2, 2), stride=(2, 2))

        return h

    ncls = 10

    # Conv -> BN -> Relu
    with nn.parameter_scope("conv1"):
        # Preprocess
        image /= 255.0
        if not test:
            image = F.image_augmentation(image, contrast=1.0,
                                         angle=0.25,
                                         flip_lr=True)
            image.need_grad = False
        h = cpd3_convolution(image, maps, kernel=(3, 3), pad=(1, 1),
                             with_bias=False, cr=compression_ratio,
                             lambda_reg=lambda_reg)
        h = PF.batch_normalization(h, batch_stat=not test)
        h = F.relu(h)

    h = res_unit(h, "conv2", False)    # -> 32x32
    h = res_unit(h, "conv3", True)     # -> 16x16
    h = res_unit(h, "conv4", False)    # -> 16x16
    h = res_unit(h, "conv5", True)     # -> 8x8
    h = res_unit(h, "conv6", False)    # -> 8x8
    h = res_unit(h, "conv7", True)     # -> 4x4
    h = res_unit(h, "conv8", False)    # -> 4x4
    h = F.average_pooling(h, kernel=(4, 4))  # -> 1x1
    pred = svd_affine(h, ncls, compression_ratio)

    return pred


def cifar10_svd_factorized_resnet23_prediction(image, maps=64,
                                               test=False, compression_ratio=0.0):
    """
    Construct Resnet23 with factorized affine and convolution
    """
    # SVD affine
    def svd_affine(x, n_outputs, cr):
        W = get_parameter('affine/W')

        if W is None:
            UV = None
        else:
            UV = W.d
        b = get_parameter('affine/b')
        # compute rank (size of intermediate activations)
        # to obtained desired reduction
        inshape = np.prod(x.shape[1:])
        outshape = np.prod(n_outputs)
        rank = int(np.floor((1-cr)*inshape*outshape/(inshape+outshape)))

        # Initialize bias to existing b in affine if exists
        if b is not None:
            b_new = get_parameter_or_create(
                'svd_affine/b', b.d.shape, need_grad=b.need_grad)
            b_new.d = b.d.copy()
        logger.info("SVD affine created: input_shape = {}; output_shape = {}; compression = {}; rank = {};".format(
            inshape, outshape, cr, rank))

        # create svd_affine initialized from W in current context if it exists
        return PF.svd_affine(x, n_outputs, rank, uv_init=UV)

    # SVD convolution
    def svd_convolution(x, n_outputs, kernel, pad, with_bias, cr):
        W = get_parameter('conv/W')

        if W is None:
            UV = None
        else:
            UV = W.d
        b = get_parameter('conv/b')
        # compute rank (size of intermediate activations)
        # to obtained desired reduction
        inmaps = x.shape[1]
        outmaps = n_outputs
        Ksize = np.prod(kernel)
        rank = int(np.floor((1-cr)*inmaps*outmaps *
                            Ksize/(inmaps*Ksize+inmaps*outmaps)))

        # Initialize bias to existing b in affine if exists
        if b is not None:
            b_new = get_parameter_or_create(
                'svd_conv/b', b.d.shape, need_grad=b.need_grad)
            b_new.d = b.d.copy()
        logger.info("SVD convolution created: inmaps = {}; outmaps = {}; compression = {}; rank = {};".format(
            inmaps, outmaps, cr, rank))

        # create svd_convolution initialized from W in current context if it exists
        return PF.svd_convolution(x, n_outputs, kernel=kernel, r=rank, pad=pad, with_bias=with_bias, uv_init=UV)

    # Residual Unit
    def res_unit(x, scope_name, dn=False):
        C = x.shape[1]
        with nn.parameter_scope(scope_name):

            # Conv -> BN -> Relu
            with nn.parameter_scope("conv1"):
                h = PF.convolution(x, C // 2, kernel=(1, 1), pad=(0, 0),
                                   with_bias=False)
                h = PF.batch_normalization(h, batch_stat=not test)
                h = F.relu(h)
            # Conv -> BN -> Relu
            with nn.parameter_scope("conv2"):
                h = svd_convolution(h, C // 2, kernel=(3, 3), pad=(1, 1),
                                    with_bias=False, cr=compression_ratio)
                h = PF.batch_normalization(h, batch_stat=not test)
                h = F.relu(h)
            # Conv -> BN
            with nn.parameter_scope("conv3"):
                h = PF.convolution(h, C, kernel=(1, 1), pad=(0, 0),
                                   with_bias=False)
                h = PF.batch_normalization(h, batch_stat=not test)
            # Residual -> Relu
            h = F.relu(h + x)

            # Maxpooling
            if dn:
                h = F.max_pooling(h, kernel=(2, 2), stride=(2, 2))

        return h

    ncls = 10

    # Conv -> BN -> Relu
    with nn.parameter_scope("conv1"):
        # Preprocess
        image /= 255.0
        if not test:
            image = F.image_augmentation(image, contrast=1.0,
                                         angle=0.25,
                                         flip_lr=True)
            image.need_grad = False
        h = svd_convolution(image, maps, kernel=(3, 3), pad=(1, 1),
                            with_bias=False, cr=compression_ratio)
        h = PF.batch_normalization(h, batch_stat=not test)
        h = F.relu(h)

    h = res_unit(h, "conv2", False)    # -> 32x32
    h = res_unit(h, "conv3", True)     # -> 16x16
    h = res_unit(h, "conv4", False)    # -> 16x16
    h = res_unit(h, "conv5", True)     # -> 8x8
    h = res_unit(h, "conv6", False)    # -> 8x8
    h = res_unit(h, "conv7", True)     # -> 4x4
    h = res_unit(h, "conv8", False)    # -> 4x4
    h = F.average_pooling(h, kernel=(4, 4))  # -> 1x1
    pred = svd_affine(h, ncls, compression_ratio)

    return pred
