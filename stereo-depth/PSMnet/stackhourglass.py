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
import nnabla.functions as F
import nnabla.parametric_functions as PF
import numpy as np
import math
import nnabla.initializer as I


def convbn(x, planes, kernel, pad, stride, dilation, batch_stat):
    h = conv(x, planes, kernel=kernel, pad=pad, stride=stride,
             dilation=dilation, with_bias=False)
    h = PF.batch_normalization(h, batch_stat=batch_stat)
    return h


def conv(x, planes, kernel, pad, stride, dilation, with_bias):
    inchannels = x.shape[1]
    outchannels = planes
    s = I.calc_normal_std_he_backward(inchannels, outchannels, kernel)
    w_init = I.NormalInitializer(s)
    if dilation[0] > 1:
        pad2 = dilation
    else:
        pad2 = pad
    h = PF.convolution(x, planes, kernel=kernel, pad=pad2, stride=stride,
                       dilation=dilation, with_bias=with_bias, w_init=w_init)

    return h


def deconvbn(x, planes, kernel, pad, stride, batch_stat):
    h = deconv(x, planes, kernel=kernel, pad=pad,
               stride=stride, with_bias=False)
    h = PF.batch_normalization(h, batch_stat=batch_stat)
    return h


def deconv(x, planes, kernel, pad, stride, with_bias):
    h = PF.deconvolution(x, planes, kernel=kernel, pad=pad,
                         stride=stride, with_bias=with_bias)
    return h


def upsample(x, factor, training, left_shape=None):
    if len(x.shape) == 4:
        if training:
            h = F.interpolate(x, scale=(factor, factor),
                              mode='linear', align_corners=True)
        else:
            h = F.interpolate(x, output_size=(
                left_shape[2]//4, left_shape[3]//4), mode='linear', align_corners=True)
    elif len(x.shape) == 5:
        planes = x.shape[1]
        kernel_size = 2*factor - factor % 2
        stride = int(factor)
        pad = int(math.ceil((factor-1) / 2.))
        scale_factor = (kernel_size + 1) // 2
        if kernel_size % 2 == 1:
            center = scale_factor - 1
        else:
            center = scale_factor - 0.5
        bilinear_kernel = np.zeros(
            [kernel_size, kernel_size, kernel_size], dtype=np.float32)
        for i in range(kernel_size):
            for j in range(kernel_size):
                for d in range(kernel_size):
                    bilinear_kernel[i, j, d] = (1 - abs(i - center) / scale_factor) * (
                        1 - abs(j - center) / scale_factor) * (1 - abs(d - center) / scale_factor)
        w_filter = np.zeros([1, planes, kernel_size, kernel_size, kernel_size])
        for i in range(planes):
            w_filter[:, i, :, :, :] = bilinear_kernel
        h = PF.deconvolution(x, planes, kernel=(kernel_size, kernel_size, kernel_size), pad=(pad, pad, pad), stride=(
            stride, stride, stride), w_init=w_filter, fix_parameters=True, group=planes)
    return h


def conv1(x, batch_stat):
    with nn.parameter_scope('conv1'):
        cv1 = convbn(x, 64, kernel=(3, 3, 3), pad=(1, 1, 1), stride=(
            2, 2, 2), dilation=(1, 1, 1), batch_stat=batch_stat)
        cv1 = F.relu(cv1, True)
    return cv1


def conv2(x, batch_stat):
    with nn.parameter_scope('conv2'):
        cv2 = convbn(x, 64, kernel=(3, 3, 3), pad=(1, 1, 1), stride=(
            1, 1, 1), dilation=(1, 1, 1), batch_stat=batch_stat)
    return cv2


def conv3(x, batch_stat):
    with nn.parameter_scope('conv3'):
        cv3 = convbn(x, 64, kernel=(3, 3, 3), pad=(1, 1, 1), stride=(
            2, 2, 2), dilation=(1, 1, 1), batch_stat=batch_stat)
        cv3 = F.relu(cv3, True)
    return cv3


def conv4(x, batch_stat):
    with nn.parameter_scope('conv4'):
        cv4 = convbn(x, 64, kernel=(3, 3, 3), pad=(1, 1, 1), stride=(
            1, 1, 1), dilation=(1, 1, 1), batch_stat=batch_stat)
        cv4 = F.relu(cv4, True)
    return cv4


def conv5(x, batch_stat):
    with nn.parameter_scope('conv5'):
        cv5 = deconvbn(x, 64, kernel=(2, 2, 2), pad=(0, 0, 0), stride=(
            2, 2, 2), batch_stat=batch_stat)
    return cv5


def conv6(x, batch_stat):
    with nn.parameter_scope('conv6'):
        cv6 = deconvbn(x, 32, kernel=(2, 2, 2), pad=(0, 0, 0), stride=(
            2, 2, 2), batch_stat=batch_stat)
    return cv6


def hourglass(x, presqu, postsqu, batch_stat):
    out = conv1(x, batch_stat)  # in:1/4 out:1/8
    pre = conv2(out, batch_stat)  # in:1/8 out:1/8
    if postsqu is not None:
        pre = F.relu(pre + postsqu, True)
    else:
        pre = F.relu(pre, True)
    out = conv3(pre, batch_stat)  # in:1/8 out:1/16
    out = conv4(out, batch_stat)  # in:1/16 out:1/16
    if presqu is not None:
        post = F.relu(conv5(out, batch_stat) + presqu, True)  # in:1/16 out:1/8
    else:
        post = F.relu(conv5(out, batch_stat) + pre, True)
    out = conv6(post, batch_stat)  # in:1/8 out:1/4
    return out, pre, post


def dres0(x, batch_stat):
    with nn.parameter_scope('dres0_conv1'):
        dr0_conv = convbn(x, 32, kernel=(3, 3, 3), pad=(1, 1, 1), stride=(
            1, 1, 1), dilation=(1, 1, 1), batch_stat=batch_stat)
        dr0_conv = F.relu(dr0_conv, True)
    with nn.parameter_scope('dres0_conv2'):
        dr0 = convbn(dr0_conv, 32, kernel=(3, 3, 3), pad=(1, 1, 1), stride=(
            1, 1, 1), dilation=(1, 1, 1), batch_stat=batch_stat)
        dr0 = F.relu(dr0, True)
    return dr0


def dres1(x, batch_stat):
    with nn.parameter_scope('dres1_conv1'):
        dr1_conv1 = convbn(x, 32, kernel=(3, 3, 3), pad=(1, 1, 1), stride=(
            1, 1, 1), dilation=(1, 1, 1), batch_stat=batch_stat)
        dr1_conv1 = F.relu(dr1_conv1, True)
    with nn.parameter_scope('dres1_conv2'):
        dr1 = convbn(dr1_conv1, 32, kernel=(3, 3, 3), pad=(1, 1, 1), stride=(
            1, 1, 1), dilation=(1, 1, 1), batch_stat=batch_stat)
    return dr1


def classif1(x, batch_stat):
    with nn.parameter_scope('classif1_conv1'):
        cl1_conv1 = convbn(x, 32, kernel=(3, 3, 3), pad=(1, 1, 1), stride=(
            1, 1, 1), dilation=(1, 1, 1), batch_stat=batch_stat)
        cl1_conv1 = F.relu(cl1_conv1, True)
    with nn.parameter_scope('classif1_conv2'):
        cl1 = conv(cl1_conv1, 1, kernel=(3, 3, 3), pad=(1, 1, 1),
                   stride=(1, 1, 1), dilation=(1, 1, 1), with_bias=True)
    return cl1


def classif2(x, batch_stat):
    with nn.parameter_scope('classif2_conv1'):
        cl2_conv1 = convbn(x, 32, kernel=(3, 3, 3), pad=(1, 1, 1), stride=(
            1, 1, 1), dilation=(1, 1, 1), batch_stat=batch_stat)
        cl2_conv1 = F.relu(cl2_conv1, True)
    with nn.parameter_scope('classif2_conv2'):
        cl2 = conv(cl2_conv1, 1, kernel=(3, 3, 3), pad=(1, 1, 1),
                   stride=(1, 1, 1), dilation=(1, 1, 1), with_bias=True)
    return cl2


def classif3(x, batch_stat):
    with nn.parameter_scope('classif3_conv1'):
        cl3_conv1 = convbn(x, 32, kernel=(3, 3, 3), pad=(1, 1, 1), stride=(
            1, 1, 1), dilation=(1, 1, 1), batch_stat=batch_stat)
        cl3_conv1 = F.relu(cl3_conv1, True)
    with nn.parameter_scope('classif3_conv2'):
        cl3 = conv(cl3_conv1, 1, kernel=(3, 3, 3), pad=(1, 1, 1),
                   stride=(1, 1, 1), dilation=(1, 1, 1), with_bias=True)
    return cl3


def build_cost_volume(limg, rimg, maxdisp):
    left_stack = []
    right_stack = []
    for i in range(int(maxdisp/4)):
        sliced_limg = limg[:, :, :, i:]
        sliced_rimg = rimg[:, :, :, :limg.shape[3] - i]
        if i == 0:
            padded_limg = sliced_limg
            padded_rimg = sliced_rimg
        else:
            # Padd i pixels on the left edge
            # The shape of padded_* becomes [B, C, H, W]
            padded_limg = F.pad(sliced_limg, (i, 0))
            padded_rimg = F.pad(sliced_rimg, (i, 0))

        left_stack.append(padded_limg)
        right_stack.append(padded_rimg)

    left_stacked = F.stack(*left_stack, axis=2)  # [B, C, D, H, W]
    right_stacked = F.stack(*right_stack, axis=2)  # [B, C, D, H, W]

    cost_volume = F.concatenate(
        left_stacked, right_stacked, axis=1)  # [B, 2C, D, H, W]
    return cost_volume


def psm_net(left, right, maxdisp, training):
    print(training)
    if training:
        batch_stat = True
    else:
        batch_stat = False

    # feature extraction
    refimg_fea = feature_extraction(left, batch_stat, training)
    targetimg_fea = feature_extraction(right, batch_stat, training)

    # matching
    cost = build_cost_volume(refimg_fea, targetimg_fea, maxdisp)

    cost0 = dres0(cost, batch_stat)
    cost0 = dres1(cost0, batch_stat) + cost0

    out1, pre1, post1 = hourglass(cost0, None, None, batch_stat)
    out1 = out1 + cost0

    out2, pre2, post2 = hourglass(out1, pre1, post1, batch_stat)
    out2 = out2 + cost0

    out3, pre3, post3 = hourglass(out2, pre1, post2, batch_stat)
    out3 = out3 + cost0

    cost1 = classif1(out1, batch_stat)
    cost2 = classif2(out2, batch_stat) + cost1
    cost3 = classif3(out3, batch_stat) + cost2

    if training:
        with nn.parameter_scope('cost1_upsample'):
            cost1_upsample = upsample(cost1, 4, True)
            cost1_upsample = F.softmax(cost1_upsample, axis=2)
            pred1 = disparityregression(cost1_upsample, maxdisp)

        with nn.parameter_scope('cost2_upsample'):
            cost2_upsample = upsample(cost2, 4, True)
            cost2_upsample = F.softmax(cost2_upsample, axis=2)
            pred2 = disparityregression(cost2_upsample, maxdisp)

    with nn.parameter_scope('cost3_upsample'):
        cost3_upsample = upsample(cost3, 4, True)
        cost3_upsample = F.softmax(cost3_upsample, axis=2)
        pred3 = disparityregression(cost3_upsample, maxdisp)

    if training:
        return pred1, pred2, pred3
    else:
        return pred3


def make_layer(x, planes, blocks, pad, stride, dilation, batch_stat):
    if stride != (1, 1) or x.shape[1] != planes:
        ml = BasicBlock(x, planes, pad, stride, dilation, True, batch_stat)
    else:
        ml = BasicBlock(x, planes, pad, stride, dilation, False, batch_stat)
    for i in range(1, blocks):
        with nn.parameter_scope('bb_loop_' + str(i)):
            ml = BasicBlock(ml, planes, pad, (1, 1),
                            dilation, False, batch_stat)
    return ml


def downsample(x, planes, stride, batch_stat):
    # convbn
    ds_conv = convbn(x, planes=planes, kernel=(1, 1), pad=(
        0, 0), stride=stride, dilation=(1, 1), batch_stat=batch_stat)
    return ds_conv


def BasicBlock(x, planes, pad, stride, dilation, isDownsample, batch_stat):
    # conv1
    with nn.parameter_scope('bb_conv1'):
        bb_conv1 = convbn(x, planes, kernel=(3, 3), pad=pad, stride=stride,
                          dilation=dilation, batch_stat=batch_stat)
        bb_conv1 = F.relu(bb_conv1, True)
    # conv2
    with nn.parameter_scope('bb_conv2'):
        bb_conv2 = convbn(bb_conv1, planes, kernel=(3, 3), pad=pad, stride=(
            1, 1), dilation=dilation, batch_stat=batch_stat)
    # downsample
    if isDownsample is True:
        with nn.parameter_scope('bb_downsample'):
            x = downsample(x, planes, stride, batch_stat)
    bb_conv2 += x
    return bb_conv2


def feature_extraction(x, batch_stat, training):
    # conv0_1
    with nn.parameter_scope('conv0_1'):
        fe_conv01 = convbn(x, 32, kernel=(3, 3), pad=(1, 1), stride=(
            2, 2), dilation=(1, 1), batch_stat=batch_stat)
        fe_conv01 = F.relu(fe_conv01, True)
    # conv0_2
    with nn.parameter_scope('conv0_2'):
        fe_conv02 = convbn(fe_conv01, 32, kernel=(3, 3), pad=(1, 1), stride=(
            1, 1), dilation=(1, 1), batch_stat=batch_stat)
        fe_conv02 = F.relu(fe_conv02, True)
    # conv0_3
    with nn.parameter_scope('conv0_3'):
        fe_conv03 = convbn(fe_conv02, 32, kernel=(3, 3), pad=(1, 1), stride=(
            1, 1), dilation=(1, 1), batch_stat=batch_stat)
        fe_conv03 = F.relu(fe_conv03, True)
    # conv1_x
    with nn.parameter_scope('conv1_x'):
        fe_conv1x = make_layer(fe_conv03, 32, 3, pad=(1, 1), stride=(
            1, 1), dilation=(1, 1), batch_stat=batch_stat)
    # conv2_x
    with nn.parameter_scope('conv2_x'):
        fe_conv2x = make_layer(fe_conv1x, 64, 16, pad=(1, 1), stride=(
            2, 2), dilation=(1, 1), batch_stat=batch_stat)
    # conv3_x
    with nn.parameter_scope('conv3_x'):
        fe_conv3x = make_layer(fe_conv2x, 128, 3, pad=(1, 1), stride=(
            1, 1), dilation=(1, 1), batch_stat=batch_stat)
    # conv4_x
    with nn.parameter_scope('conv4_x'):
        fe_conv4x = make_layer(fe_conv3x, 128, 3, pad=(1, 1), stride=(
            1, 1), dilation=(2, 2), batch_stat=batch_stat)
    # SPP
    with nn.parameter_scope('SPP'):
        fe = SPP(fe_conv4x, fe_conv2x, batch_stat, training, x.shape)

    return fe


def branch(x, kernel, batch_stat):
    # AveragePooling
    br_avg = F.average_pooling(x, kernel, kernel)
    # convbn
    br_conv = convbn(br_avg, 32, kernel=(1, 1), pad=(0, 0), stride=(
        1, 1), dilation=(1, 1), batch_stat=batch_stat)
    # ReLU
    br_conv = F.relu(br_conv, True)
    return br_conv


def lastconv(x, batch_stat):
    # convbn
    with nn.parameter_scope('lastconv_conv1'):
        lc_conv1 = convbn(x, 128, kernel=(3, 3), pad=(1, 1), stride=(
            1, 1), dilation=(1, 1), batch_stat=batch_stat)
        # ReLU
        lc_conv1 = F.relu(lc_conv1, True)
    # conv
    with nn.parameter_scope('lastconv_conv2'):
        lc = conv(lc_conv1, 32, kernel=(1, 1), pad=(0, 0),
                  stride=(1, 1), dilation=(1, 1), with_bias=True)
    return lc


def SPP(x0, x1, batch_stat, training, left_shape):

    # branch_1
    with nn.parameter_scope('branch_1_conv'):
        branch_1_avg = branch(x0, (64, 64), batch_stat)

    with nn.parameter_scope('branch_1_upsample'):
        branch_1 = upsample(branch_1_avg, 64, training, left_shape)

    # branch_2
    with nn.parameter_scope('branch_2_conv'):
        branch_2_avg = branch(x0, (32, 32), batch_stat)
    with nn.parameter_scope('branch_2_upsample'):
        branch_2 = upsample(branch_2_avg, 32, training, left_shape)

    # branch_3
    with nn.parameter_scope('branch_3_conv'):
        branch_3_avg = branch(x0, (16, 16), batch_stat)
    with nn.parameter_scope('branch_3_upsample'):
        branch_3 = upsample(branch_3_avg, 16, training, left_shape)

    # branch
    with nn.parameter_scope('branch_4_conv'):
        branch_4_avg = branch(x0, (8, 8), batch_stat)
    with nn.parameter_scope('branch_4_upsample'):
        branch_4 = upsample(branch_4_avg, 8, training, left_shape)

    # Concatenate
    spp_concat = F.concatenate(
        x1, x0, branch_4, branch_3, branch_2, branch_1, axis=1)

    # Lastconv
    with nn.parameter_scope('spp_lastconv'):
        spp = lastconv(spp_concat, batch_stat)
    return spp


def disparityregression(x, maxdisp):
    disp = nn.Variable((x.shape), need_grad=False)
    for i in range(0, maxdisp):
        disp.d[:, :, i, :, :] = i
    dispx = F.mul2(disp, x)
    out = F.sum(dispx, axis=2)
    return out
