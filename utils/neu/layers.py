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

import numpy as np

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.initializer as I

from .initializer import w_init
from .losses import get_gan_loss, mae
from .callbacks import spectral_norm_callback

# alias
ps = nn.parameter_scope


#################################################################
# normalizations
#################################################################
def _normalize(x, norm_type, channel_axis=1):
    if norm_type.lower() == "in":
        return F.instance_normalization(x, gamma=None, beta=None, channel_axis=channel_axis)
    elif norm_type.lower() == "bn":
        return F.batch_normalization(x, gamma=None, beta=None, mean=None, variance=None, axes=channel_axis)
    else:
        raise ValueError("unknown norm_type: {}".format(norm_type))


def spade(x, m, hidden_dim=128, kernel=(3, 3), norm_type="in"):
    """
    Spatially-Adaptive Normalization proposed in Semantic Image Synthesis with Spatially-Adaptive Normalization (https://arxiv.org/pdf/1903.07291.pdf).


    Args:
        x (nn.Variable): Input variable for spade layer.
        m (nn.Variable):
            Spatial condition variable like object_id mask segmentation.
            This is for generating adaptive scale(gamma) and adaptice bias(beta) applied after normalization.
        hidden_dim (int): Hidden dims for first convolution applied to m.
        kernel (list of int): Kernel shapes for convolutions.
        norm_type (str) : A type of normalization. ["in", "bn"] are supported now.
    """
    # x: (N, Cx, H, W), m: (N, Cm, H, W)
    assert len(x.shape) == 4 and len(m.shape) == 4

    pad = tuple(i // 2 for i in kernel)
    c_dim = x.shape[1]
    conv_args = dict(kernel=kernel, pad=pad)
    with ps("spatial_adaptive_normalization"):
        normalized = _normalize(x, norm_type)

        m = F.interpolate(m, output_size=x.shape[2:], mode="nearest")

        with ps("shared"):
            actv = F.relu(PF.convolution(
                m, hidden_dim, w_init=w_init(m, hidden_dim), **conv_args))

        with ps("gamma"):
            gamma = PF.convolution(
                actv, c_dim, w_init=w_init(actv, c_dim), **conv_args)

        with ps("beta"):
            beta = PF.convolution(
                actv, c_dim, w_init=w_init(actv, c_dim), **conv_args)

    return normalized * gamma + beta


def rescale_values(x, input_min=-1, input_max=1, output_min=0, output_max=255):
    """
    Rescale the range of values of `x` from [input_min, input_max] -> [output_min, output_max].
    """

    assert input_max > input_min
    assert output_max > output_min

    return (x - input_min) * (output_max - output_min) / (input_max - input_min) + output_min


##############################################
# GAN Generator/Discriminator
##############################################

class PatchGAN(object):
    def __init__(self, n_layers=4, base_ndf=64, n_scales=2,
                 use_sigmoid=False, use_spectral_normalization=True):
        """
        PatchGAN discriminator.

        Args:
            n_layers:
        :param base_ndf:
        :param n_scales:
        :param use_sigmoid:
        """

        self.n_layers = n_layers
        self.base_ndf = base_ndf
        self.n_scales = n_scales
        self.use_sigmoid = use_sigmoid

        self.conv_opts = dict(w_init=I.NormalInitializer(0.02))
        if use_spectral_normalization:
            self.conv_opts["apply_w"] = spectral_norm_callback(dim=0)

    def instance_norm_lrelu(self, x, alpha=0.2):
        norm = PF.instance_normalization(x, no_scale=True, no_bias=True)
        return F.leaky_relu(norm, alpha=alpha, inplace=True)

    def pad_conv(self, x, fdim, stride):
        h = PF.convolution(x, fdim, (4, 4), stride=stride,
                           pad=(2, 2), **self.conv_opts)

        return h

    @staticmethod
    def get_loss(real_out, real_feats, fake_out, fake_feats,
                 use_fm=True, fm_lambda=10., gan_loss_type="ls"):
        """
        Get losses from patchGAN outputs.
        If use_fm is False, F.constant(0) is returned as g_feat.

        Args:
            real_out, real_feats, fake_out, fake_feats (nn.Variable): Variables returned from PatchGAN.__call__().
            use_fm (bool): Use feature matching loss or not.
            fm_lambda (float): Coefficient applied for feature matching loss.
            gan_loss_type (str): GAN loss type. One of ["ls", "hinge"].

        usage:
            real = nn.Variable(...)
            fake = generator(...)

            d = PatchGAN(n_layers=4, n_scales=2)
            real_out, real_feats = d(real)
            fake_out, fake_feats = d(fake)
            g_gan, g_feat, d_real, d_fake = d.get_loss(real_out, real_feats, fake_out, fake_feats,
                                                       use_fm=True, fm_lambda=10., gan_loss_type="ls")

        """

        g_gan = 0
        g_feat = 0 if use_fm else F.constant(0)
        d_real = 0
        d_fake = 0

        gan_loss = get_gan_loss(gan_loss_type)

        n_disc = len(real_out)

        for disc_id in real_out.keys():
            r_out = real_out[disc_id]
            f_out = fake_out[disc_id]

            # define GAN loss
            _d_real, _d_fake, _g_gan = gan_loss(r_out, f_out)

            d_real += _d_real
            d_fake += _d_fake
            g_gan += _g_gan

            # feature matching
            if use_fm:
                assert r_out.shape == f_out.shape
                r_feats = real_feats[disc_id]
                f_feats = fake_feats[disc_id]

                for layer_id, r_feat in r_feats.items():
                    g_feat += mae(r_feat,
                                  f_feats[layer_id]) * fm_lambda / n_disc

        return g_gan, g_feat, d_real, d_fake

    def __call__(self, x):
        outs = {}
        feats = {}
        with ps("discriminator/patch_gan"):
            # Create all scale inputs first.
            inputs = [x]
            for i in range(self.n_scales - 1):
                inputs.append(F.average_pooling(
                    inputs[-1], (3, 3), (2, 2), pad=(1, 1), including_pad=False))

            for i in range(self.n_scales):
                # Get input in reverse order of its scale (from coarse to fine) to preserve discriminator indexes.
                h = inputs.pop()
                d_name = "d_{}".format(i)

                feats[d_name] = {}
                with ps(d_name):
                    # first layer
                    fdim = self.base_ndf
                    with ps("layer_0"):
                        h = self.pad_conv(h, fdim, stride=(2, 2))
                        h = F.leaky_relu(h, alpha=0.2, inplace=True)

                    feats[d_name]["layer_0"] = h

                    # middle
                    for j in range(1, self.n_layers - 1):
                        fdim = min(fdim * 2, 512)
                        layer_name = "layer_{}".format(j)
                        with ps(layer_name):
                            h = self.pad_conv(h, fdim, stride=(2, 2))
                            h = self.instance_norm_lrelu(h)

                        feats[d_name][layer_name] = h

                    # last 2 layers
                    fdim = min(fdim * 2, 512)
                    layer_name = "layer_{}".format(self.n_layers - 1)
                    with ps(layer_name):
                        h = self.pad_conv(h, fdim, stride=(1, 1))
                        h = self.instance_norm_lrelu(h)

                    feats[d_name][layer_name] = h

                    with nn.parameter_scope("last_layer"):
                        h = self.pad_conv(h, 1, stride=(1, 1))
                        if self.use_sigmoid:
                            h = F.sigmoid(h)

                    outs[d_name] = h

        return outs, feats
