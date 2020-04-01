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

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF

from utils import spade, w_init, spectral_norm_callback

ps = nn.parameter_scope


def _check_intput(x):
    assert isinstance(x, nn.Variable)
    assert len(x.shape) == 3 or (len(x.shape) == 4 and x.shape[-1] == 1)


def inst_to_boundary(inst_label):

    pad = F.pad(inst_label, (1, 1, 1, 1))
    bm = F.constant(val=0, shape=pad.shape)
    bm = F.logical_or(bm, F.not_equal(pad, F.pad(inst_label, (1, 1, 0, 2))))
    bm = F.logical_or(bm, F.not_equal(pad, F.pad(inst_label, (1, 1, 2, 0))))
    bm = F.logical_or(bm, F.not_equal(pad, F.pad(inst_label, (0, 2, 1, 1))))
    bm = F.logical_or(bm, F.not_equal(pad, F.pad(inst_label, (2, 0, 1, 1))))

    return bm[:, 1:-1, 1:-1]  # (N, H, W)


def encode_inputs(inst_label, id_label, n_ids, use_encoder=False, channel_last=False):
    """
    :param inst_label: (N, H, W) or (N, H, W, 1)
    :param id_label: (N, H, W) or (N, H, W, 1)
    :param use_encoder: boolean
    :return:
    """
    # id (index) -> onehot
    _check_intput(id_label)
    if len(id_label.shape) == 3:
        id_label = id_label.reshape(id_label.shape + (1,))
    id_onehot = F.one_hot(id_label, shape=(n_ids,))

    # inst -> boundary map
    _check_intput(inst_label)
    bm = inst_to_boundary(inst_label)
    if len(bm.shape) == 3:
        bm = bm.reshape(bm.shape + (1,))

    if use_encoder:
        # todo: implement encoder network
        pass

    if channel_last:
        return id_onehot, bm

    return F.transpose(id_onehot, (0, 3, 1, 2)), F.transpose(bm, (0, 3, 1, 2))


class SpadeResidualBlock(object):
    def __init__(self, out_dim, use_sn=True):
        self.out_dim = out_dim

        self.conv_opts = dict(with_bias=False)

        if use_sn:
            self.conv_opts["apply_w"] = spectral_norm_callback(dim=0)

    def act(self, x):
        return F.leaky_relu(x, 2e-1)

    def shortcut(self, x, m):
        s = x
        if x.shape[1] != self.out_dim:
            with ps("shortcut"):
                s = spade(s, m)
                s = PF.convolution(s, self.out_dim, kernel=(1, 1),
                                   w_init=w_init(s, self.out_dim), **self.conv_opts)

        return s

    def __call__(self, x, m):
        # x: (N, C, H, W)
        s = self.shortcut(x, m)

        hidden_dim = min(x.shape[1], self.out_dim)
        with ps("res_layer1"):
            h = spade(x, m)
            h = self.act(h)
            h = PF.convolution(h, hidden_dim, kernel=(3, 3), pad=(1, 1),
                               w_init=w_init(h, hidden_dim), **self.conv_opts)

        with ps("res_layer2"):
            h = spade(h, m)
            h = self.act(h)
            h = PF.convolution(h, self.out_dim, kernel=(3, 3), pad=(1, 1),
                               w_init=w_init(h, self.out_dim), **self.conv_opts)

        return s + h


class SpadeGenerator(object):
    def __init__(self, nf, image_shape, ext_upsamples=0):
        ext_upsamples = int(ext_upsamples)
        assert isinstance(ext_upsamples, int) and 0 <= ext_upsamples <= 2,\
            "ext_upsamples must be in the range of [0, 2]."

        self.nf = nf
        self.image_shape = image_shape
        self.num_upsample = 5 + ext_upsamples

        self.head_0 = SpadeResidualBlock(16 * nf)

        self.G_middle_0 = SpadeResidualBlock(16 * nf)
        self.G_middle_1 = SpadeResidualBlock(16 * nf)

        self.up_0 = SpadeResidualBlock(8 * nf)
        self.up_1 = SpadeResidualBlock(4 * nf)
        self.up_2 = SpadeResidualBlock(2 * nf)
        self.up_3 = SpadeResidualBlock(nf)

        if self.num_upsample > 6:
            self.up_4 = SpadeResidualBlock(nf // 2)

        self.up = lambda x: F.interpolate(x, scale=(2, 2), mode="nearest")

    def __call__(self, z, m):
        # m has target image shape: (N, emb, H, W)
        # z: (N, z_dim)

        N = m.shape[0]
        H, W = self.image_shape
        sh = H // (2 ** self.num_upsample)
        sw = W // (2 ** self.num_upsample)

        with ps("spade_generator"):
            with ps("z_embedding"):
                x = PF.affine(z, 16 * self.nf * sh * sw,
                              w_init=w_init(z, 16 * self.nf * sh * sw))
                x = F.reshape(x, (N, 16*self.nf, sh, sw))

            with ps("head"):
                x = self.head_0(x, m)

            with ps("middle0"):
                x = self.up(x)
                x = self.G_middle_0(x, m)

            with ps("middel1"):
                if self.num_upsample > 5:
                    x = self.up(x)
                x = self.G_middle_1(x, m)

            with ps("up0"):
                x = self.up(x)
                x = self.up_0(x, m)

            with ps("up1"):
                x = self.up(x)
                x = self.up_1(x, m)

            with ps("up2"):
                x = self.up(x)
                x = self.up_2(x, m)

            with ps("up3"):
                x = self.up(x)
                x = self.up_3(x, m)

            if self.num_upsample > 6:
                with ps("up4"):
                    x = self.up(x)
                    x = self.up_4(x, m)

            with ps("last_conv"):
                x = PF.convolution(F.leaky_relu(x, 2e-1), 3,
                                   kernel=(3, 3), pad=(1, 1), w_init=w_init(x, 3))
                x = F.tanh(x)

        return x
