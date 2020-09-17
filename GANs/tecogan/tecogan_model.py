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

import collections
import nnabla as nn
import nnabla.functions as F
from nnabla.monitor import Monitor, MonitorSeries, MonitorTimeElapsed
from models import flow_estimator, generator, discriminator
from utils import set_persistent_all
from utils.utils import deprocess, warp_by_flow, space_to_depth, space_to_depth_disc, upscale_four
from vgg19 import VGG19


def get_common_monitors(monitor):
    """
    Create monitors for displaying and storing losses.
    """
    monitor_content_loss = MonitorSeries(
        'content loss', monitor, interval=20)
    monitor_gen_loss = MonitorSeries(
        'generator loss', monitor, interval=20)
    monitor_warp_loss = MonitorSeries(
        'warp loss', monitor, interval=20)
    monitor_lr = MonitorSeries(
        'learning rate', monitor, interval=20)
    monitor_time = MonitorTimeElapsed(
        "training time per iteration", monitor, interval=20)
    Monitor_common = collections.namedtuple('Monitor_common',
                                            ['monitor_content_loss', 'monitor_gen_loss', 'monitor_warp_loss',
                                             'monitor_lr', 'monitor_time'])
    return Monitor_common(monitor_content_loss, monitor_gen_loss, monitor_warp_loss, monitor_lr, monitor_time)


def get_tecogan_monitors(monitor):
    """
    Create monitors for displaying and storing TECOGAN losses.
    """
    monitor_vgg_loss = MonitorSeries(
        'vgg loss', monitor, interval=20)
    monitor_pp_loss = MonitorSeries(
        'ping pong', monitor, interval=20)
    monitor_sum_layer_loss = MonitorSeries(
        'd layer loss', monitor, interval=20)
    monitor_adv_loss = MonitorSeries(
        'adversarial loss', monitor, interval=20)
    monitor_disc_loss = MonitorSeries(
        'discriminator loss', monitor, interval=20)
    monitor_tb = MonitorSeries(
        'tb', monitor, interval=20)
    Monitor_tecogan = collections.namedtuple('Monitor_tecogan',
                                             ['monitor_vgg_loss', 'monitor_pp_loss', 'monitor_sum_layer_loss',
                                              'monitor_adv_loss', 'monitor_disc_loss', 'monitor_tb'])
    return Monitor_tecogan(monitor_vgg_loss, monitor_pp_loss, monitor_sum_layer_loss, monitor_adv_loss, monitor_disc_loss, monitor_tb)


def get_tecogan_inputs(r_inputs, r_targets):
    """
    Generate and return the ping-pong sequence (forward and backward) from given inputs and targets
    """
    r_inputs = F.concatenate(r_inputs, r_inputs[:, -2::-1, :, :, :], axis=1)
    r_targets = F.concatenate(r_targets, r_targets[:, -2::-1, :, :, :], axis=1)
    return r_inputs, r_targets


def get_fnet_output(conf, rnn_length, frame_t_pre, frame_t, scope_name):
    """
    Return the flow estimations for LR and HR from flow-estimator network
    """
    fnet_input = F.concatenate(frame_t_pre, frame_t)
    fnet_input = F.reshape(fnet_input, (conf.train.batch_size*(rnn_length-1),
                                        conf.train.crop_size, conf.train.crop_size, 2*3))
    with nn.parameter_scope(scope_name + "fnet"):
        flow_lr = flow_estimator(fnet_input)
    flow_hr = upscale_four(flow_lr*4.0)  # a linear up-sampling
    flow_hr = F.reshape(flow_hr, (conf.train.batch_size, (rnn_length-1), conf.train.crop_size*4,
                                  conf.train.crop_size*4, 2), inplace=False)
    fnet_output = collections.namedtuple('fnet_output', 'flow_lr, flow_hr')
    return fnet_output(
        flow_lr=flow_lr,
        flow_hr=flow_hr,
                )


def get_generator_output(conf, rnn_length, r_inputs, flow_hr, scope_name):
    """
    Return the generated HR frames
    """
    # list for all outputs
    gen_outputs = []

    # for the first frame, concat with zeros
    input0 = F.concatenate(r_inputs[:, 0, :, :, :], F.constant(
        0, (conf.train.batch_size, conf.train.crop_size, conf.train.crop_size, 3*4*4)))
    with nn.parameter_scope(scope_name + "generator"):
        gen_pre_output = generator(input0, 3, conf.train.num_resblock)
    gen_outputs.append(gen_pre_output)  # append generated HR frame-0

    for frame_i in range(rnn_length - 1):
        cur_flow = flow_hr[:, frame_i, :, :, :]
        # warp the previously generated frame
        gen_pre_output_warp = warp_by_flow(gen_pre_output, cur_flow)
        gen_pre_output_warp = F.identity(deprocess(gen_pre_output_warp))
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
    gen_outputs = F.stack(*gen_outputs, axis=1)
    # gen_outputs, nn.Variable with shape = (batch, frame, FLAGS.crop_size*4, FLAGS.crop_size*4, 3)

    return gen_outputs


def get_warp_loss(conf, rnn_length, frame_t, frame_t_pre, flow_lr):
    """
    Warp loss
    """
    input_frames = F.reshape(frame_t, (conf.train.batch_size*(rnn_length-1),
                                       conf.train.crop_size, conf.train.crop_size, 3))
    frame_t_pre_reshaped = F.reshape(frame_t_pre, (conf.train.batch_size*(rnn_length-1),
                                                   conf.train.crop_size, conf.train.crop_size, 3))
    s_input_warp = warp_by_flow(frame_t_pre_reshaped, flow_lr)

    warp_loss = F.mean(F.sum(F.squared_error(
        input_frames, s_input_warp), axis=[3]))

    return warp_loss


def get_d_data(conf, flow_hr, gen_outputs, r_targets, rnn_length):
    """
    prepare data for temporal Discriminators
    """
    # 3 frames are used as one entry, the last input images%3 frames are abandoned
    t_size = int(3 * (rnn_length // 3))
    t_gen_output = F.reshape(gen_outputs[:, :t_size, :, :, :], (conf.train.batch_size *
                                                                t_size, conf.train.crop_size*4, conf.train.crop_size*4, 3),
                             inplace=False)
    t_targets = F.reshape(r_targets[:, :t_size, :, :, :], (conf.train.batch_size *
                                                           t_size, conf.train.crop_size*4, conf.train.crop_size*4, 3),
                          inplace=False)
    t_batch = conf.train.batch_size * t_size // 3
    t_inputs_v_pre_batch = F.identity(
                flow_hr[:, 0:t_size:3, :, :, :])  # forward motion reused,
    t_inputs_v_batch = nn.Variable(t_inputs_v_pre_batch.shape)
    # no motion for middle frames
    t_inputs_v_batch.data.zero()
    t_inputs_v_nxt_batch = F.identity(
                flow_hr[:, -2:-1-t_size:-3, :, :, :])  # backward motion

    t_vel = F.stack(
                *[t_inputs_v_pre_batch, t_inputs_v_batch, t_inputs_v_nxt_batch], axis=2)
    # batch, t_size/3, 3, FLAGS.crop_size*4, FLAGS.crop_size*4, 2
    t_vel = F.reshape(t_vel, (conf.train.batch_size*t_size,
                              conf.train.crop_size*4, conf.train.crop_size*4, 2), inplace=False)
    # Stop gradient to fnet from discriminator, details in TecoGAN supplemental paper
    t_vel.need_grad = False

    disc_data = collections.namedtuple('disc_data',
                                       't_vel, t_gen_output, t_batch, t_targets, t_size')
    return disc_data(
        t_vel=t_vel,
        t_gen_output=t_gen_output,
        t_batch=t_batch,
        t_targets=t_targets,
        t_size=t_size
                )


def get_t_d(conf, r_inputs, d_data):
    """
    Create Real and fake temoral discriminators
    """
    # to crop out unstable part for temporal discriminator, details in TecoGAN supplemental paper
    crop_size_dt = int(conf.train.crop_size * 4 * conf.gan.crop_dt)
    offset_dt = (conf.train.crop_size * 4 - crop_size_dt) // 2
    crop_size_dt = conf.train.crop_size * 4 - offset_dt*2
    paddings = (0, 0, offset_dt, offset_dt, offset_dt, offset_dt, 0, 0)

    with nn.parameter_scope("discriminator"):
        real_warp = warp_by_flow(d_data.t_targets, d_data.t_vel)
        real_warp = space_to_depth_disc(real_warp, d_data.t_batch)

        # equivalent to tf.image.crop_to_bounding_box
        real_warp = real_warp[:, offset_dt:offset_dt +
                              crop_size_dt, offset_dt:offset_dt+crop_size_dt, :]
        real_warp = F.pad(real_warp, paddings)
        before_warp = space_to_depth_disc(d_data.t_targets, d_data.t_batch)
        t_input = space_to_depth_disc(
            r_inputs[:, :d_data.t_size, :, :, :], d_data.t_batch)
        # resizing using bilinear interpolation
        input_hi = F.interpolate(t_input, scale=(4, 4), mode='linear',
                                 channel_last=True)
        real_warp = F.concatenate(before_warp, real_warp, input_hi)

        tdiscrim_real_output, real_layers = discriminator(real_warp)

        fake_warp = warp_by_flow(d_data.t_gen_output, d_data.t_vel)
        fake_warp = space_to_depth_disc(fake_warp, d_data.t_batch)
        fake_warp = fake_warp[:, offset_dt:offset_dt +
                              crop_size_dt, offset_dt:offset_dt+crop_size_dt, :]
        fake_warp = F.pad(fake_warp, paddings)
        before_warp = space_to_depth_disc(
            d_data.t_gen_output, d_data.t_batch, inplace=False)
        fake_warp = F.concatenate(before_warp, fake_warp, input_hi)
        tdiscrim_fake_output, fake_layers = discriminator(fake_warp)

    temporal_disc = collections.namedtuple('temporal_disc', 'tdiscrim_real_output,'
                                           'real_layers, tdiscrim_fake_output, fake_layers')
    return temporal_disc(
        tdiscrim_real_output=tdiscrim_real_output,
        real_layers=real_layers,
        tdiscrim_fake_output=tdiscrim_fake_output,
        fake_layers=fake_layers
                )


def get_d_layer(real_layers, fake_layers):
    """
    discriminator layer loss
    """
    fix_range = 0.02  # hard coded, all layers are roughly scaled to this value
    sum_layer_loss = 0  # adds-on for generator
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
        scaled_layer_loss = fix_range * \
            layer_loss / layer_norm[layer_i]
        sum_layer_loss += scaled_layer_loss

    return sum_layer_loss


def get_vgg_loss(s_gen_output, s_targets):
    """
    VGG loss
    """
    vgg19 = VGG19()
    vgg_gen_fea = vgg19(s_gen_output)
    vgg_target_fea = vgg19(s_targets)
    vgg_loss_list = []
    # we use 4 VGG layers
    vgg_wei_list = [1.0, 1.0, 1.0, 1.0]
    vgg_loss = 0
    vgg_layer_n = 4

    for layer_i in range(vgg_layer_n):
        # cosine similarity, -1~1, 1 best
        curvgg_diff = F.sum(
            vgg_gen_fea[layer_i]*vgg_target_fea[layer_i], axis=[3])
        curvgg_diff = 1.0 - F.mean(curvgg_diff)  # 0 ~ 2, 0 best
        scaled_layer_loss = vgg_wei_list[layer_i] * curvgg_diff
        vgg_loss_list += [curvgg_diff]
        vgg_loss += scaled_layer_loss

    return vgg_loss


def get_tecogan_model(conf, r_inputs, r_targets, scope_name, tecogan=True):
    """
    Create computation graph and variables for TecoGAN.
    """
    # r_inputs, r_targets : shape (batch, conf.train.rnn_n, h, w, c)
    rnn_length = conf.train.rnn_n
    if tecogan:
        r_inputs, r_targets = get_tecogan_inputs(r_inputs, r_targets)
        rnn_length = rnn_length * 2 - 1

    # get the consecutive frame sequences from the input sequence
    frame_t_pre, frame_t = r_inputs[:, 0:-1, :, :, :], r_inputs[:, 1:, :, :, :]

    # Get flow estimations
    fnet_output = get_fnet_output(
        conf, rnn_length, frame_t_pre, frame_t, scope_name)

    # Get the generated HR output frames
    gen_outputs = get_generator_output(
        conf, rnn_length, r_inputs, fnet_output.flow_hr, scope_name)

    s_gen_output = F.reshape(gen_outputs, (conf.train.batch_size * rnn_length,
                                           conf.train.crop_size * 4,
                                           conf.train.crop_size*4, 3), inplace=False)
    s_targets = F.reshape(r_targets, (conf.train.batch_size * rnn_length, conf.train.crop_size * 4,
                                      conf.train.crop_size * 4, 3), inplace=False)

    # Content loss (l2 loss)
    content_loss = F.mean(
        F.sum(F.squared_error(s_gen_output, s_targets), axis=[3]))
    # Warp loss (l2 loss)
    warp_loss = get_warp_loss(
        conf, rnn_length, frame_t, frame_t_pre, fnet_output.flow_lr)

    if tecogan:
        d_data = get_d_data(conf, fnet_output.flow_hr,
                            gen_outputs, r_targets, rnn_length)
        # Build the tempo discriminator for the real part and fake part
        t_d = get_t_d(conf, r_inputs, d_data)

        # Discriminator layer loss:
        d_layer_loss = get_d_layer(t_d.real_layers, t_d.fake_layers)
        # vgg loss (cosine similarity)
        loss_vgg = get_vgg_loss(s_gen_output, s_targets)
        # ping pong loss (an l1 loss)
        gen_out_first = gen_outputs[:, 0:conf.train.rnn_n-1, :, :, :]
        gen_out_last_rev = gen_outputs[:, -1:-conf.train.rnn_n:-1, :, :, :]
        pp_loss = F.mean(F.abs(gen_out_first - gen_out_last_rev))
        # adversarial loss
        t_adversarial_loss = F.mean(
            -F.log(t_d.tdiscrim_fake_output + conf.train.eps))

        # Overall generator loss
        gen_loss = content_loss + pp_loss * conf.gan.pp_scaling + conf.gan.ratio * \
            t_adversarial_loss + conf.gan.vgg_scaling * loss_vgg + \
            conf.gan.dt_ratio_0 * d_layer_loss

        # Discriminator loss
        t_discrim_fake_loss = F.log(
            1 - t_d.tdiscrim_fake_output + conf.train.eps)
        t_discrim_real_loss = F.log(t_d.tdiscrim_real_output + conf.train.eps)
        t_discrim_loss = F.mean(-(t_discrim_fake_loss + t_discrim_real_loss))

        fnet_loss = gen_loss + warp_loss

        set_persistent_all(r_targets, r_inputs, loss_vgg, gen_out_first, gen_out_last_rev, pp_loss,
                           d_layer_loss, content_loss, warp_loss, gen_loss, t_adversarial_loss,
                           t_discrim_loss, t_discrim_real_loss, d_data.t_vel, d_data.t_gen_output,
                           s_gen_output, s_targets)

        Network = collections.namedtuple('Network', 'content_loss,  warp_loss, fnet_loss, vgg_loss,'
                                         'gen_loss, pp_loss, sum_layer_loss,t_adversarial_loss,'
                                         't_discrim_loss,t_gen_output,t_discrim_real_loss')
        return Network(
            content_loss=content_loss,
            warp_loss=warp_loss,
            fnet_loss=fnet_loss,
            vgg_loss=loss_vgg,
            gen_loss=gen_loss,
            pp_loss=pp_loss,
            sum_layer_loss=d_layer_loss,
            t_adversarial_loss=t_adversarial_loss,
            t_discrim_loss=t_discrim_loss,
            t_gen_output=d_data.t_gen_output,
            t_discrim_real_loss=t_discrim_real_loss
        )

    gen_loss = content_loss
    fnet_loss = gen_loss + warp_loss
    set_persistent_all(content_loss, s_gen_output,
                       warp_loss, gen_loss, fnet_loss)

    Network = collections.namedtuple(
        'Network', 'content_loss, warp_loss, fnet_loss, gen_loss')
    return Network(
        content_loss=content_loss,
        warp_loss=warp_loss,
        fnet_loss=fnet_loss,
        gen_loss=gen_loss,
    )


def get_frvsr_model(conf, r_inputs, r_targets, scope_name):
    """
    Create computation graph and variables for FRVSR.
    """
    return get_tecogan_model(conf, r_inputs, r_targets, scope_name, False)
