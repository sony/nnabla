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

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.initializer as I


def namescope_decorator(scope):
    def wrapper1(builder_function):

        def wrapper2(*args, **kwargs):
            with nn.parameter_scope(scope):
                return builder_function(*args, **kwargs)

        return wrapper2

    return wrapper1


def get_symmetric_padwidth(pad, dims=2, channel_last=False):
    pad = (pad, ) * dims * 2

    if channel_last:
        pad += (0, 0)

    return pad


class BaseGenerator(object):
    def __init__(self, padding_type="reflect", channel_last=False):
        self.initializer = I.NormalInitializer(0.02)
        self.padding_type = padding_type
        self.channel_last = channel_last
        # currently deconv dose not support channel last.
        self.conv_opts = dict(w_init=self.initializer)
        # don't use adaptive parameter
        self.norm_opts = dict(fix_parameters=True)

    def instance_norm_relu(self, x):
        # return F.relu(PF.layer_normalization(x, **self.norm_opts), inplace=True)
        return F.relu(PF.instance_normalization(x, **self.norm_opts), inplace=True)

    def residual_block(self, x, o_channels):
        pad_width = get_symmetric_padwidth(1, channel_last=self.channel_last)

        with nn.parameter_scope("residual_1"):
            h = F.pad(x, pad_width=pad_width, mode=self.padding_type)
            h = PF.convolution(h, o_channels, (3, 3), **self.conv_opts)
            h = self.instance_norm_relu(h)

        with nn.parameter_scope("residual_2"):
            h = F.pad(h, pad_width=pad_width, mode=self.padding_type)
            h = PF.convolution(h, o_channels, (3, 3), **self.conv_opts)
            h = PF.instance_normalization(h, **self.norm_opts)

        return x + h

    def residual_loop(self, x, o_channels, num_layers):
        h = x
        for i in range(num_layers):
            with nn.parameter_scope("layer_{}".format(i)):
                h = self.residual_block(h, o_channels)

        return h


class Encoder(BaseGenerator):
    def __init__(self, padding_type="reflect"):
        super(Encoder, self).__init__(padding_type=padding_type)

    @namescope_decorator("encdec")
    def encdec(self, x, n_downsamples):

        with nn.parameter_scope("first layer"):
            pad_width = get_symmetric_padwidth(
                3, channel_last=self.channel_last)
            h = F.pad(x, pad_width=pad_width, mode=self.padding_type)
            h = PF.convolution(h, 32, (7, 7), **self.conv_opts)
            h = self.instance_norm_relu(h)

        # down sample layers
        for i in range(n_downsamples):
            with nn.parameter_scope("down_{}".format(i)):
                c = 32 * 2 ** (i + 1)
                h = PF.convolution(h, c, (3, 3), strides=(
                    2, 2), pad=(1, 1), **self.conv_opts)
                h = self.instance_norm_relu(h)

        # up sample layers
        for i in range(n_downsamples):
            with nn.parameter_scope("up_{}".format(i)):
                c = 32 * 2 ** (n_downsamples - i - 1)
                h = PF.deconvolution(h, c, (3, 3), stride=(
                    2, 2), pad=(1, 1), **self.conv_opts)
                h = F.pad(h, pad_width=(0, 1, 0, 1))  # output padding
                h = self.instance_norm_relu(h)

        with nn.parameter_scope("last layer"):
            pad_width = get_symmetric_padwidth(
                3, channel_last=self.channel_last)
            h = F.pad(h, pad_width=pad_width, mode=self.padding_type)
            h = PF.convolution(h, 3, (7, 7), **self.conv_opts)
            h = F.tanh(h)

        return h

    @namescope_decorator("instance_wise_avg_pooling")
    def instance_wise_avg_pooling(self, x, inst):
        return x

    def __call__(self, x, inst, n_downsamples=4):

        with nn.parameter_scope("encoder"):
            h = self.encdec(x, n_downsamples)
            feat = self.instance_wise_avg_pooling(h, inst)

        return feat


class GlobalGenerator(BaseGenerator):
    def __init__(self, padding_type="reflect"):
        super(GlobalGenerator, self).__init__(padding_type=padding_type)

    @namescope_decorator("frontend")
    def front_end(self, x):
        with nn.parameter_scope("first_layer"):
            pad_width = get_symmetric_padwidth(
                3, channel_last=self.channel_last)
            h = F.pad(x, pad_width=pad_width, mode=self.padding_type)
            h = PF.convolution(h, 64, (7, 7), **self.conv_opts)
            h = self.instance_norm_relu(h)

        channels = [128, 256, 512, 1024]
        for i, channel in enumerate(channels):
            with nn.parameter_scope("down_sample_layer_{}".format(i)):
                h = PF.convolution(h, channel, (3, 3), stride=(
                    2, 2), pad=(1, 1), **self.conv_opts)
                h = self.instance_norm_relu(h)

        return h

    @namescope_decorator("residual")
    def residual(self, x):
        return self.residual_loop(x, 1024, 9)

    @namescope_decorator("backend")
    def back_end(self, x):
        h = x
        channels = [512, 256, 128, 64]
        for i, channel in enumerate(channels):
            with nn.parameter_scope("up_sample_layer_{}".format(i)):
                h = PF.deconvolution(h, channel, (4, 4), stride=(
                    2, 2), pad=(1, 1), **self.conv_opts)
                h = self.instance_norm_relu(h)

        last_feat = h

        with nn.parameter_scope("last_layer"):
            pad_width = get_symmetric_padwidth(
                3, channel_last=self.channel_last)
            h = F.pad(h, pad_width=pad_width, mode=self.padding_type)
            h = PF.convolution(h, 3, (7, 7), **self.conv_opts)
            h = F.tanh(h)

        return h, last_feat

    def __call__(self, x, downsample_input=True):
        if downsample_input:
            x = F.average_pooling(
                x, (3, 3), (2, 2), pad=(1, 1), including_pad=False)

        with nn.parameter_scope("generator/global"):
            h = self.front_end(x)
            h = self.residual(h)
            out, feat = self.back_end(h)

        return out, feat


class LocalGenerator(BaseGenerator):
    def __init__(self, padding_type="reflect"):
        super(LocalGenerator, self).__init__(padding_type=padding_type)

    @namescope_decorator("frontend")
    def front_end(self, x):
        with nn.parameter_scope("first_layer"):
            pad_width = get_symmetric_padwidth(
                3, channel_last=self.channel_last)
            h = F.pad(x, pad_width=pad_width, mode=self.padding_type)
            h = PF.convolution(h, 32, (7, 7), **self.conv_opts)
            h = self.instance_norm_relu(h)

        with nn.parameter_scope("down_sample_layer"):
            h = PF.convolution(h, 64, (3, 3), stride=(2, 2),
                               pad=(1, 1), **self.conv_opts)
            h = self.instance_norm_relu(h)

        return h

    @namescope_decorator("residual")
    def residual(self, x):
        return self.residual_loop(x, 64, 3)

    @namescope_decorator("backend")
    def back_end(self, x):
        with nn.parameter_scope("up_sample_layer"):
            h = PF.deconvolution(x, 32, (4, 4), stride=(
                2, 2), pad=(1, 1), **self.conv_opts)
            h = self.instance_norm_relu(h)

        last_feat = h

        with nn.parameter_scope("last_layer"):
            pad_width = get_symmetric_padwidth(
                3, channel_last=self.channel_last)
            h = F.pad(h, pad_width=pad_width, mode=self.padding_type)
            h = PF.convolution(h, 3, (7, 7), **self.conv_opts)
            h = F.tanh(h)

        return h, last_feat

    def __call__(self, x, n_scales=1):
        if n_scales < 1:
            raise ValueError("n_scale must be equal or greater than 1.")

        # create all scale inputs
        inputs = [x]
        for _ in range(n_scales - 1):
            inputs.append(F.average_pooling(
                inputs[-1], (3, 3), (2, 2), pad=(1, 1), including_pad=False))

        # global generator (coarsest scale generator)
        gg = GlobalGenerator(self.padding_type)

        _input = inputs.pop()  # get the coarsest scale input
        last_scale_out, last_scale_feat = gg(_input, downsample_input=False)

        # if n_scales == 1, below loop is skipped and local generator is identical to global generator.
        for scale_id in range(n_scales - 1):
            _input = inputs.pop()  # get input from coarser scale.
            with nn.parameter_scope("generator/local_{}".format(scale_id)):
                h = self.front_end(_input)
                h = self.residual(h + last_scale_feat)
                last_scale_out, last_scale_feat = self.back_end(h)

        return last_scale_out, last_scale_feat


class Discriminator(object):
    def __init__(self):
        self.n_layers = 4
        self.base_fdim = 64

        self.initializer = I.NormalInitializer(0.02)
        self.conv_opts = dict(w_init=self.initializer)
        self.norm_opts = dict(fix_parameters=True)

    def instance_norm_lrelu(self, x, alpha=0.2):
        # return F.leaky_relu(PF.layer_normalization(x, **self.norm_opts), alpha=alpha, inplace=True)
        return F.leaky_relu(PF.instance_normalization(x, **self.norm_opts), alpha=alpha, inplace=True)

    def pad_conv(self, x, fdim, stride):
        h = PF.convolution(x, fdim, (4, 4), stride=stride,
                           pad=(2, 2), **self.conv_opts)

        return h

    @namescope_decorator("discriminator")
    def __call__(self, x, n_scales, use_sigmoid=False):
        outs = {}
        feats = {}

        # Create all scale inputs first.
        inputs = [x]
        for i in range(n_scales - 1):
            inputs.append(F.average_pooling(
                inputs[-1], (3, 3), (2, 2), pad=(1, 1), including_pad=False))

        for i in range(n_scales):
            # Get input in reverse order of its scale (from coarse to fine) to preserve discriminator indexes.
            h = inputs.pop()
            d_name = "discriminator_{}".format(i)

            feats[d_name] = {}
            with nn.parameter_scope(d_name):
                # first layer
                fdim = self.base_fdim
                with nn.parameter_scope("layer_0"):
                    h = self.pad_conv(h, fdim, stride=(2, 2))
                    h = F.leaky_relu(h, alpha=0.2, inplace=True)

                feats[d_name]["layer_0"] = h

                # middle
                for j in range(1, self.n_layers - 1):
                    fdim = min(fdim * 2, 512)
                    layer_name = "layer_{}".format(j)
                    with nn.parameter_scope(layer_name):
                        h = self.pad_conv(h, fdim, stride=(2, 2))
                        h = self.instance_norm_lrelu(h)

                    feats[d_name][layer_name] = h

                # last 2 layers
                fdim = min(fdim * 2, 512)
                layer_name = "layer_{}".format(self.n_layers - 1)
                with nn.parameter_scope(layer_name):
                    h = self.pad_conv(h, fdim, stride=(1, 1))
                    h = self.instance_norm_lrelu(h)

                feats[d_name][layer_name] = h

                with nn.parameter_scope("last_layer"):
                    h = self.pad_conv(h, 1, stride=(1, 1))
                    if use_sigmoid:
                        h = F.sigmoid(h)

                outs[d_name] = h

        return outs, feats


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


def define_loss(real_out, real_feats, fake_out, fake_feats, use_fm=True, fm_lambda=10.):
    g_gan = 0
    g_feat = 0 if use_fm else F.constant(0)
    g_vgg = F.constant(0)  # todo
    d_real = 0
    d_fake = 0

    n_disc = len(real_out)

    for disc_id in real_out.keys():
        r_out = real_out[disc_id]
        r_feats = real_feats[disc_id]
        f_out = fake_out[disc_id]
        f_feats = fake_feats[disc_id]

        # define GAN loss
        # # for D
        d_real += F.mean(F.squared_error(r_out,
                                         F.constant(1., shape=r_out.shape)))
        d_fake += F.mean(F.squared_error(f_out,
                                         F.constant(0., shape=f_out.shape)))

        # # for G
        g_gan += F.mean(F.squared_error(f_out,
                                        F.constant(1., shape=f_out.shape)))

        # feature matching
        if use_fm:
            assert r_out.shape == f_out.shape

            for layer_id, r_feat in r_feats.items():
                g_feat += F.mean(F.absolute_error(r_feat,
                                                  f_feats[layer_id])) * fm_lambda / n_disc

    return g_gan, g_feat, g_vgg, d_real, d_fake
