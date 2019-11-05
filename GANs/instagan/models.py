# Copyright (c) 2019 Sony Corporation. All Rights Reserved.
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

import os
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.initializer as I
import numpy as np

import functools


# Define a resnet block
def resnetblock(x, dim, padding_type, norm_layer, use_dropout, use_bias):
    assert dim == x.shape[1], "The number of input / output channels must match."
    h = x

    p = 0
    if padding_type == 'reflect':
        h = F.pad(h, (1, 1, 1, 1), 'reflect')
    elif padding_type == 'zero':
        p = 1
    else:
        raise NotImplementedError(
            'padding {} is not implemented'.format(padding_type))
    w_init = I.NormalInitializer(sigma=0.02, rng=None)

    h = PF.convolution(h, dim, kernel=(3, 3), pad=(
        p, p), w_init=w_init, with_bias=use_bias, name="1st")
    h = norm_layer(h, name="1st")
    h = F.relu(h, inplace=True)

    if use_dropout:
        h = F.dropout(h, 0.5)

    if padding_type == 'reflect':
        h = F.pad(h, (1, 1, 1, 1), 'reflect')

    h = PF.convolution(h, dim, kernel=(3, 3), pad=(
        p, p), w_init=w_init, with_bias=use_bias, name="2nd")
    h = norm_layer(h, name="2nd")

    out = F.add2(x, h)

    return out


def encode(input_variable, input_nc, n_downsampling, ngf, norm_layer, use_dropout, n_blocks, padding_type, use_bias):
    """
    encoder is used for both image and mask.
    """
    w_init = I.NormalInitializer(sigma=0.02, rng=None)

    x = input_variable
    h = F.pad(x, (3, 3, 3, 3), 'reflect')
    h = PF.convolution(h, ngf, kernel=(7, 7), w_init=w_init,
                       with_bias=use_bias, name="enc_initial_conv")
    h = norm_layer(h, name="enc_initial_norm")
    h = F.relu(h, inplace=True)

    for i in range(n_downsampling):
        with nn.parameter_scope("enc_downsampling_{}".format(i)):
            mult = 2**i
            h = PF.convolution(h, ngf * mult * 2, kernel=(3, 3), stride=(2, 2),
                               pad=(1, 1), w_init=w_init, with_bias=use_bias)
            h = norm_layer(h)
            h = F.relu(h, inplace=True)

    mult = 2**n_downsampling
    for i in range(n_blocks):
        with nn.parameter_scope("resblock_{}".format(i)):
            h = resnetblock(h, ngf * mult, padding_type=padding_type,
                            norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)

    return h


def decode(input_feature, output_nc, n_downsampling, ngf, norm_layer, use_bias):
    h = input_feature
    w_init = I.NormalInitializer(sigma=0.02, rng=None)

    for i in range(n_downsampling):
        with nn.parameter_scope("dec_downsampling_{}".format(i)):
            mult = 2**(n_downsampling - i)
            h = PF.deconvolution(h, int(ngf * mult / 2), kernel=(4, 4),
                                 stride=(2, 2), pad=(1, 1), w_init=w_init, with_bias=use_bias)
            # kernel changed 3 -> 4 to make the output fit to the desired size.
            h = norm_layer(h)
            h = F.relu(h, inplace=True)

    h = F.pad(h, (3, 3, 3, 3), 'reflect')
    h = PF.convolution(h, output_nc, kernel=(
        7, 7), w_init=w_init, with_bias=use_bias, name="dec_last_conv")
    h = F.tanh(h)

    return h


def generator(imgseg, input_nc=3, output_nc=3, ngf=64, norm_layer=functools.partial(PF.instance_normalization, fix_parameters=True), use_dropout=False, n_blocks=9, padding_type='reflect'):

    if type(norm_layer) == functools.partial:
        #use_bias = norm_layer.func == PF.instance_normalization
        use_bias = norm_layer.func != PF.batch_normalization
    else:
        #use_bias = norm_layer == PF.instance_normalization
        use_bias = norm_layer != PF.batch_normalization

    n_downsampling = 2

    input_image = imgseg[:, :3, :, :]
    input_mask = imgseg[:, 3:, :, :]

    with nn.parameter_scope("img"):
        img_feat = encode(input_image, input_nc, n_downsampling, ngf,
                          norm_layer, use_dropout, n_blocks, padding_type, use_bias)

    with nn.parameter_scope("seg"):
        seg_feat = encode(input_mask, input_nc, n_downsampling, ngf,
                          norm_layer, use_dropout, n_blocks, padding_type, use_bias)
    seg_feat_sum = F.sum(seg_feat, axis=0, keepdims=True)

    input_for_image = F.concatenate(img_feat, seg_feat, axis=1)
    input_for_mask = F.concatenate(seg_feat, img_feat, seg_feat_sum, axis=1)
    # input_for_mask = F.concatenate(img_feat, seg_feat, axis=1)

    with nn.parameter_scope("img"):
        generated_img = decode(input_for_image, output_nc,
                               n_downsampling, 2 * ngf, norm_layer, use_bias)
    with nn.parameter_scope("seg"):
        generated_mask = decode(
            input_for_mask, 1, n_downsampling, 3 * ngf, norm_layer, use_bias)

    generated_imgseg = F.concatenate(generated_img, generated_mask, axis=1)

    return generated_imgseg


def feature_extractor(input_variable, input_nc, ndf, n_layers, kw, padw, norm_layer, use_bias):
    w_init = I.NormalInitializer(sigma=0.02, rng=None)
    x = input_variable

    with nn.parameter_scope("feature_extractor_init_conv"):
        def apply_w(w): return PF.spectral_norm(w, dim=0)
        h = PF.convolution(x, ndf, kernel=(kw, kw), stride=(
            2, 2), pad=(padw, padw), w_init=w_init, apply_w=apply_w)
        h = F.leaky_relu(h, alpha=0.2, inplace=True)

    nf_mult = 1
    nf_mult_prev = 1

    for n in range(1, n_layers):
        nf_mult_prev = nf_mult
        nf_mult = min(2**n, 8)
        with nn.parameter_scope("feature_extractor_stage_{}".format(n)):
            def apply_w(w): return PF.spectral_norm(w, dim=0)
            h = PF.convolution(h, ndf * nf_mult, kernel=(kw, kw), stride=(2, 2),
                               pad=(padw, padw), w_init=w_init, with_bias=use_bias, apply_w=apply_w)
            h = norm_layer(h)
            h = F.leaky_relu(h, alpha=0.2, inplace=True)

    return h


def classifier(input_feature, ndf, n_layers, kw, padw, norm_layer, use_sigmoid):
    w_init = I.NormalInitializer(sigma=0.02, rng=None)
    h = input_feature
    nf_mult_prev = min(2**(n_layers - 1), 8)
    nf_mult = min(2**n_layers, 8)

    with nn.parameter_scope("dis_classifier_1"):
        def apply_w(w): return PF.spectral_norm(w, dim=0)
        h = PF.convolution(h, ndf * nf_mult, kernel=(kw, kw), stride=(1, 1),
                           pad=(padw, padw), w_init=w_init, apply_w=apply_w)
        h = norm_layer(h)
        h = F.leaky_relu(h, alpha=0.2, inplace=True)

    # Use spectral normalization
    with nn.parameter_scope("dis_classifier_2"):
        def apply_w(w): return PF.spectral_norm(w, dim=0)
        h = PF.convolution(h, 1, kernel=(kw, kw), stride=(
            1, 1), pad=(padw, padw), w_init=w_init, apply_w=apply_w)

    if use_sigmoid:
        h = F.sigmoid(h)
    return h


def discriminator(imgseg, input_nc=3, ndf=64, n_layers=3, norm_layer=functools.partial(PF.instance_normalization, fix_parameters=True), use_sigmoid=False):

    if type(norm_layer) == functools.partial:
        #use_bias = norm_layer.func == PF.instance_normalization
        use_bias = norm_layer.func != PF.batch_normalization
    else:
        #use_bias = norm_layer == PF.instance_normalization
        use_bias = norm_layer != PF.batch_normalization

    kw = 4
    padw = 1

    input_image = imgseg[:, :3, :, :]
    input_mask = imgseg[:, 3:, :, :]

    # run feature extractor
    with nn.parameter_scope("img"):
        feature_img = feature_extractor(
            input_image, input_nc, ndf, n_layers, kw, padw, norm_layer, use_bias)

    with nn.parameter_scope("seg"):
        feature_seg = feature_extractor(
            input_mask, 1, ndf, n_layers, kw, padw, norm_layer, use_bias)

    # run classifier
    concat_feature = F.concatenate(feature_img, feature_seg, axis=1)
    with nn.parameter_scope("classifier"):
        score = classifier(concat_feature, 2 * ndf, n_layers,
                           kw, padw, norm_layer, use_sigmoid)

    return score
