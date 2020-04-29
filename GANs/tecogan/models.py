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
import numpy as np
import nnabla.functions as F
import nnabla.parametric_functions as PF
import cv2
import numpy
from utils import *
from utils.utils import *
from vgg19 import Load_vgg19

# Definition of the flow_estimation : Flow-Estimation network


def flow_estimator(fnet_input):
    def down_block(inputs, output_channel=64, stride=1, scope='down_block'):
        with nn.parameter_scope(scope):
            net = PF.convolution(inputs, output_channel, (3, 3), stride=(
                stride, stride), pad=(1, 1), with_bias=True, name='conv_1', channel_last=True)
            net = F.leaky_relu(net, 0.2)
            net = PF.convolution(net, output_channel, (3, 3), stride=(
                stride, stride), pad=(1, 1), with_bias=True, name='conv_2', channel_last=True)
            net = F.leaky_relu(net, 0.2)
            net = F.max_pooling(net, (2, 2), channel_last=True)
        return net

    def up_block(inputs, output_channel=64, stride=1, scope='up_block'):
        with nn.parameter_scope(scope):
            net = PF.convolution(inputs, output_channel, (3, 3), stride=(
                stride, stride), pad=(1, 1), with_bias=True, name='conv_1', channel_last=True)
            net = F.leaky_relu(net, 0.2)
            net = PF.convolution(net, output_channel, (3, 3), stride=(
                stride, stride), pad=(1, 1), with_bias=True, name='conv_2', channel_last=True)
            net = F.leaky_relu(net, 0.2)
            net = F.interpolate(net, scale=(2, 2), channel_last=True)
        return net

    with nn.parameter_scope('autoencode_unit'):
        net = down_block(fnet_input, 32, scope='encoder_1')
        net = down_block(net, 64, scope='encoder_2')
        net = down_block(net, 128, scope='encoder_3')

        net = up_block(net, 256, scope='decoder_1')
        net = up_block(net, 128, scope='decoder_2')
        net = up_block(net, 64, scope='decoder_3')

        with nn.parameter_scope('output_stage'):
            net = PF.convolution(net, 32, (3, 3), stride=(1, 1), pad=(
                1, 1), with_bias=True, name='conv1', channel_last=True)
            net = F.leaky_relu(net, 0.2)
            net = PF.convolution(net, 2, (3, 3), stride=(1, 1), pad=(
                1, 1), with_bias=True, name='conv2', channel_last=True)
            # the 24.0 is the max Velocity, details can be found in TecoGAN paper
            net = F.tanh(net) * 24.0
    return net

# Definition of the generator network


def generator(gen_inputs, gen_output_channels, num_resblock=16):
    # The residual blocks
    def residual_block(inputs, output_channel=64, stride=1, scope='res_block'):
        with nn.parameter_scope(scope):
            net = PF.convolution(inputs, output_channel, kernel=(3, 3), stride=(
                stride, stride), pad=(1, 1), with_bias=True, name='conv_1', channel_last=True)
            net = F.relu(net)
            net = PF.convolution(net, output_channel, (3, 3), stride=(
                stride, stride), pad=(1, 1), with_bias=True, name='conv_2', channel_last=True)
            net = net + inputs
        return net

    with nn.parameter_scope('generator_unit'):
        # The input layer
        with nn.parameter_scope('input_stage'):
            net = PF.convolution(gen_inputs, 64, (3, 3), stride=(1, 1), pad=(
                1, 1), with_bias=True, name='conv', channel_last=True)
            stage1_output = F.relu(net)
        net = stage1_output

        # The residual block parts
        for i in range(0, num_resblock, 1):  # should be 16 for TecoGAN, and 10 for TecoGANmini
            net = residual_block(net, 64, 1, scope='resblock_%d' % (i+1))

        with nn.parameter_scope('conv_tran2highres'):
            net = F.pad(net, (1, 0, 1, 0, 0, 0), mode='constant')
            net = PF.deconvolution(net, 64, (3, 3), stride=(2, 2), pad=(
                1, 1), with_bias=True, name='conv_tran1', channel_last=True)
            net = net[:, 1:, 1:, :]
            net = F.relu(net)
            net = F.pad(net, (1, 0, 1, 0, 0, 0), mode='constant')
            net = PF.deconvolution(net, 64, (3, 3), stride=(2, 2), pad=(
                1, 1), with_bias=True, name='conv_tran2', channel_last=True)
            net = net[:, 1:, 1:, :]
            net = F.relu(net)

        with nn.parameter_scope('output_stage'):
            net = PF.convolution(net, gen_output_channels, (3, 3), stride=(
                1, 1), pad=(1, 1), with_bias=True, name='conv', channel_last=True)
            low_res_in = gen_inputs[:, :, :, 0:3]  # ignore warped pre high res
            bicubic_hi = bicubic_four(low_res_in)  # can put on GPU
            net = net + bicubic_hi
            net = net * 2 - 1
    return net


def discriminator_block(inputs, out_channel, kernel_size, stride, scope):
    with nn.parameter_scope(scope):
        h = PF.convolution(inputs, out_channel, kernel=(kernel_size, kernel_size), stride=(
            stride, stride), pad=(1, 1), with_bias=False, channel_last=True)
        h = PF.batch_normalization(h)
        h = F.leaky_relu(h, alpha=0.2, inplace=True)
    return h


def discriminator(dis_inputs):
    layer_list = []
    h = PF.convolution(dis_inputs, 64, kernel=(3, 3), stride=(
        1, 1), pad=(1, 1), name='conv', channel_last=True)
    h = F.leaky_relu(h, alpha=0.2, inplace=True)
    # (b,h,w,64)
    h = discriminator_block(h, 64, 4, 2, 'disblock_1')
    layer_list += [h]
    # (b,h/2,w/2,64)
    h = discriminator_block(h, 64, 4, 2, 'disblock_3')
    layer_list += [h]
    # (b,h/4,w/4,64)
    h = discriminator_block(h, 128, 4, 2, 'disblock_5')
    layer_list += [h]
    # (b,h/8,w/8,128)
    h = discriminator_block(h, 256, 4, 2, 'disblock_7')
    layer_list += [h]
    # (b,h/16,w/16,256)
    h = PF.affine(h, 1, base_axis=3, name='affine0')  # channel wise affine
    h = F.sigmoid(h)
    # (b,h/16,w/16,1)
    return h, layer_list


def tecogan(conf, r_inputs, r_targets, scope_name, tecogan=True):
    # r_inputs, r_targets : shape (b,frame,h,w,c)
    inputimages = conf.train.rnn_n

    # list for all outputs
    gen_outputs = []
    if tecogan:
        r_inputs_rev_input = r_inputs[:, -2::-1, :, :, :]
        r_targets_rev_input = r_targets[:, -2::-1, :, :, :]
        r_inputs = F.concatenate(r_inputs, r_inputs_rev_input, axis=1)
        r_targets = F.concatenate(r_targets, r_targets_rev_input, axis=1)
        inputimages = conf.train.rnn_n * 2 - 1

    frame_t_pre = r_inputs[:, 0:-1, :, :, :]
    frame_t = r_inputs[:, 1:, :, :, :]
    fnet_input = F.concatenate(frame_t_pre, frame_t)
    fnet_input = F.reshape(fnet_input, (conf.train.batch_size*(inputimages-1),
                                        conf.train.crop_size, conf.train.crop_size, 2*3))
    # batch*(frame-1), args.crop_size, args.crop_size, 3
    with nn.parameter_scope(scope_name + "fnet"):
        gen_flow_lr = flow_estimator(fnet_input)
        # batch * (inputimages-1), args.crop_size, args.crop_size, 2
    gen_flow = upscale_four(gen_flow_lr*4.0)  # a linear up-sampling
    # batch * (inputimages-1), args.crop_size*4, args.crop_size*4, 2
    gen_flow = F.reshape(gen_flow, (conf.train.batch_size,
                                    (inputimages-1), conf.train.crop_size*4, conf.train.crop_size*4, 2), inplace=False)  # Reshape
    input_frames = F.reshape(frame_t, (conf.train.batch_size*(inputimages-1),
                                       conf.train.crop_size, conf.train.crop_size, 3))
    frame_t_pre = F.reshape(frame_t_pre, (conf.train.batch_size*(inputimages-1),
                                          conf.train.crop_size, conf.train.crop_size, 3))
    s_input_warp = warp_by_flow(frame_t_pre, gen_flow_lr)

    # for the first frame, concat with zeros
    input0 = F.concatenate(r_inputs[:, 0, :, :, :], F.constant(
        0, (conf.train.batch_size, conf.train.crop_size, conf.train.crop_size, 3*4*4)))
    with nn.parameter_scope(scope_name + "generator"):
        gen_pre_output = generator(input0, 3, conf.train.num_resblock)
    # batch, FLAGS.crop_size*4, FLAGS.crop_size*4, 3
    gen_outputs.append(gen_pre_output)  # frame 0, done
 
    for frame_i in range(inputimages - 1):
        # warp the previously generated frame
        cur_flow = gen_flow[:, frame_i, :, :, :]

        gen_pre_output_warp = warp_by_flow(gen_pre_output, cur_flow)
        gen_pre_output_warp = F.identity(deprocess(gen_pre_output))
        # warp frame [0,n-1] to frame [1,n]
        # apply space-to-depth transform
        gen_pre_output_warp = space_to_depth(gen_pre_output_warp)
        # pack it as the recurrent input
        inputs = F.concatenate(
            r_inputs[:, frame_i+1, :, :, :], gen_pre_output_warp)
        # super-resolution part
        with nn.parameter_scope(scope_name + "generator"):
            gen_output = generator(inputs, 3, conf.train.num_resblock)
        gen_outputs.append(gen_output)
        gen_pre_output = gen_output

    # gen_outputs, a list, len = frame, shape = (batch, FLAGS.crop_size*4, FLAGS.crop_size*4, 3)
    gen_outputs = gen_outputs
    gen_outputs = F.stack(*gen_outputs, axis=1)
    # batch, frame, FLAGS.crop_size*4, FLAGS.crop_size*4, 3

    s_gen_output = F.reshape(
        gen_outputs, (conf.train.batch_size*inputimages, conf.train.crop_size*4, conf.train.crop_size*4, 3), inplace=False)
    s_targets = F.reshape(r_targets, (conf.train.batch_size *
                                      inputimages, conf.train.crop_size*4, conf.train.crop_size*4, 3), inplace=False)

    # Content loss (l2 loss)
    content_loss = F.mean(
        F.sum(F.squared_error(s_gen_output, s_targets), axis=[3]))
    gen_loss = content_loss

    # Warp loss (l2 loss)
    warp_loss = F.mean(F.sum(F.squared_error(
        input_frames, s_input_warp), axis=[3]))
		

    if tecogan:
        # 3 frames are used as one entry, the last inputimages%3 frames are abandoned
        t_size = int(3 * (inputimages // 3))
        t_gen_output = F.reshape(gen_outputs[:, :t_size, :, :, :], (conf.train.batch_size *
                                                                    t_size, conf.train.crop_size*4, conf.train.crop_size*4, 3), inplace=False)
        t_targets = F.reshape(r_targets[:, :t_size, :, :, :], (conf.train.batch_size *
                                                               t_size, conf.train.crop_size*4, conf.train.crop_size*4, 3), inplace=False)
        t_batch = conf.train.batch_size*t_size//3
        t_inputs_v_pre_batch = F.identity(
            gen_flow[:, 0:t_size:3, :, :, :])  # forward motion reused,
        t_inputs_v_batch = nn.Variable(t_inputs_v_pre_batch.shape)
        # no motion for middle frames, todo remove for better performance
        t_inputs_v_batch.data.zero()
        t_inputs_v_nxt_batch = F.identity(
            gen_flow[:, -2:-1-t_size:-3, :, :, :])  # backward motion

        t_vel = F.stack(
            *[t_inputs_v_pre_batch, t_inputs_v_batch, t_inputs_v_nxt_batch], axis=2)
        # batch, t_size/3, 3, FLAGS.crop_size*4, FLAGS.crop_size*4, 2
        t_vel = F.reshape(t_vel, (conf.train.batch_size*t_size,
                                  conf.train.crop_size*4, conf.train.crop_size*4, 2), inplace=False)
        t_vel.need_grad = False

        if conf.gan.crop_dt < 1.0:  # to crop out unstable part for temporal discriminator, details in TecoGAN supplemental paper
            crop_size_dt = int(conf.train.crop_size * 4 * conf.gan.crop_dt)
            offset_dt = (conf.train.crop_size * 4 - crop_size_dt) // 2
            crop_size_dt = conf.train.crop_size * 4 - offset_dt*2
            paddings = (0, 0, offset_dt, offset_dt, offset_dt, offset_dt, 0, 0)

        # Build the tempo discriminator for the real part
        with nn.parameter_scope("discriminator"):
            real_warp0 = warp_by_flow(t_targets, t_vel)
            real_warp = space_to_depth_disc(real_warp0, t_batch)
            if(conf.gan.crop_dt < 1.0):
                # equivalent to tf.image.crop_to_bounding_box
                real_warp = real_warp[:, offset_dt:offset_dt +
                                      crop_size_dt, offset_dt:offset_dt+crop_size_dt, :]
                real_warp = F.pad(real_warp, paddings)
                before_warp = space_to_depth_disc(t_targets, t_batch)
                t_input = space_to_depth_disc(
                    r_inputs[:, :t_size, :, :, :], t_batch)
                # resizing usin gbilinear interpolation
                input_hi = F.interpolate(t_input, scale=(
                    4, 4), mode='linear', channel_last=True)
                real_warp = F.concatenate(before_warp, real_warp, input_hi)

            tdiscrim_real_output, real_layers = discriminator(real_warp)

        # Build the tempo discriminator for the fake part
        with nn.parameter_scope("discriminator"):
            fake_warp0 = warp_by_flow(t_gen_output, t_vel)
            fake_warp = space_to_depth_disc(fake_warp0, t_batch)
            fake_warp = fake_warp[:, offset_dt:offset_dt +
                                  crop_size_dt, offset_dt:offset_dt+crop_size_dt, :]
            fake_warp = F.pad(fake_warp, paddings)
            before_warp = space_to_depth_disc(t_gen_output, t_batch, inplace=False)
            fake_warp = F.concatenate(before_warp, fake_warp, input_hi)
            tdiscrim_fake_output, fake_layers = discriminator(fake_warp)

        # prepare the layer between discriminators
        with nn.parameter_scope("layer_loss"):
            fix_Range = 0.02  # hard coded, all layers are roughly scaled to this value
            fix_margin = 0.0  # 0.0 will ignore losses on the Discriminator part, which is good,
            # because it is too strong usually. details in paper
            sum_layer_loss = 0  # adds-on for generator
            d_layer_loss = 0  # adds-on for discriminator, clipped with Fix_margin

            layer_loss_list = []
            layer_n = len(real_layers)

            # hard coded, an overall average of all layers
            layer_norm = [12.0, 14.0, 24.0, 100.0]

            for layer_i in range(layer_n):
                real_layer = real_layers[layer_i]
                false_layer = fake_layers[layer_i]

                layer_diff = real_layer - false_layer
                layer_loss = F.mean(
                    F.sum(F.abs(layer_diff), axis=[3]))  # an l1 loss
                layer_loss_list += [layer_loss]
                scaled_layer_loss = fix_Range * \
                    layer_loss / layer_norm[layer_i]
                sum_layer_loss += scaled_layer_loss

        # vgg loss (cosine similarity)
        vgg_gen_fea = Load_vgg19(s_gen_output)
        vgg_target_fea = Load_vgg19(s_targets)
        vgg_loss = None
        vgg_loss_list = []
        if conf.gan.vgg_scaling > 0.0:
            # we use 4 VGG layers
            vgg_wei_list = [1.0, 1.0, 1.0, 1.0]
            vgg_loss = 0
            vgg_layer_n = 4
		
            for layer_i in range(vgg_layer_n):
                curvgg_diff = F.sum(
                    vgg_gen_fea[layer_i]*vgg_target_fea[layer_i], axis=[3])
                # cosine similarity, -1~1, 1 best
                curvgg_diff = 1.0 - F.mean(curvgg_diff)  # 0 ~ 2, 0 best
                scaled_layer_loss = vgg_wei_list[layer_i] * curvgg_diff
                vgg_loss_list += [curvgg_diff]
                vgg_loss += scaled_layer_loss

            gen_loss += conf.gan.vgg_scaling * vgg_loss

        # ping pong loss (an l1 loss)
        gen_out_first = gen_outputs[:, 0:conf.train.rnn_n-1, :, :, :]
        gen_out_last_rev = gen_outputs[:, -1:-conf.train.rnn_n:-1, :, :, :]
        pp_loss = F.mean(F.abs(gen_out_first - gen_out_last_rev))
        if conf.gan.pp_scaling > 0:
            gen_loss += pp_loss * conf.gan.pp_scaling

        # adversarial loss
        t_adversarial_loss = F.mean(-F.log(tdiscrim_fake_output +
                                           conf.train.eps))
        # we can fade in of the discrim_loss,
        # but for TecoGAN paper, we always use FLAGS.Dt_ratio_0 and Dt_ratio_max as 1.0 (no fading in
        # iteration_global = 10  # for testing purpose
        # dt_ratio = F.minimum_scalar(conf.gan.dt_ratio_0 + conf.gan.dt_ratio_add *
                                    # F.constant(iteration_global, (1,)), val=conf.gan.dt_ratio_max)

        t_adversarial_fading = t_adversarial_loss * conf.gan.dt_ratio_0

        gen_loss += conf.gan.ratio * t_adversarial_fading
        if conf.gan.d_layerloss:
            gen_loss += sum_layer_loss * conf.gan.dt_ratio_0

        # Discriminator loss
        t_discrim_fake_loss = F.log(1 - tdiscrim_fake_output + conf.train.eps)
        t_discrim_real_loss = F.log(tdiscrim_real_output + conf.train.eps)

        t_discrim_loss = F.mean(-(t_discrim_fake_loss + t_discrim_real_loss))


    fnet_loss = gen_loss + warp_loss
    content_loss.persistent = True
    gen_flow_lr.persistent = True
    s_gen_output.persistent = True
    gen_flow.persistent = True
    warp_loss.persistent = True
    gen_loss.persistent = True
    fnet_loss.persistent = True
	
    if tecogan:
        r_targets.persistent = True
        r_inputs.persistent = True
        vgg_loss.persistent = True
        gen_out_first.persistent = True
        gen_out_last_rev.persistent = True
        pp_loss.persistent = True
        sum_layer_loss.persistent = True
        t_adversarial_loss.persistent = True
        t_discrim_loss.persistent = True        
        t_discrim_real_loss.persistent = True
        t_vel.persistent = True
        t_gen_output.persistent = True

        Network = collections.namedtuple(
            'Network', 'gen_flow, r_inputs, r_targets, gen_out_first, gen_out_last_rev, content_loss,  warp_loss, fnet_loss, vgg_loss, gen_loss, pp_loss,sum_layer_loss,'
		                         't_adversarial_loss,t_discrim_loss, t_discrim_real_loss,t_vel,t_gen_output, s_gen_output')
        return Network(
            gen_flow = gen_flow,  
            r_inputs = r_inputs,  
            r_targets = r_targets,  
            gen_out_first = gen_out_first,  
            gen_out_last_rev = gen_out_last_rev,  
            content_loss = content_loss,  
            warp_loss = warp_loss,
            fnet_loss = fnet_loss,
            vgg_loss = vgg_loss,
            gen_loss = gen_loss,
            pp_loss = pp_loss,
            sum_layer_loss = sum_layer_loss,
            t_adversarial_loss = t_adversarial_loss,
            t_discrim_loss = t_discrim_loss,
            t_discrim_real_loss = t_discrim_real_loss,
            t_vel = t_vel,
            t_gen_output = t_gen_output,
            s_gen_output = s_gen_output        
        )

    else:
        Network = collections.namedtuple(
            'Network', 'content_loss, warp_loss, fnet_loss, gen_loss')
        return Network(
            content_loss = content_loss,  
            warp_loss = warp_loss,
            fnet_loss = fnet_loss,
            gen_loss = gen_loss,
        )

def frvsr(conf, r_inputs, r_targets, scope_name):
    return tecogan(conf, r_inputs, r_targets, scope_name, False)
