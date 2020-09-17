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
import cv2 as cv
import numpy as np


class ExponentialMovingAverage():
    def __init__(self, decay):
        self.decay = decay
        self.shadow_variable = nn.NdArray()

    def register(self, val):
        self.shadow_variable.copy_from(val)

    def __call__(self, variable):
        self.shadow_variable.copy_from(
            self.decay * self.shadow_variable + (1.0 - self.decay) * variable)
        return self.shadow_variable.data


def save_img(out_path, img):
    img = np.clip(img*255.0, 0, 255).astype(np.uint8)
    cv.imwrite(out_path, img[:, :, ::-1])


def deprocess(image):
    # [-1, 1] => [0, 1]
    return (image + 1) / 2


def warp_by_flow(data, flow):
    """
    Warp by flow implementation equivalent to TF's tfa.image.dense_image_warp()
    """
    flow = -flow[:, :, :, ::-
                 1]  # Flow direction changed and it's channel's order is reversed
    warp_data = F.warp_by_flow(F.transpose(data, (0, 3, 1, 2)), F.transpose(
        flow, (0, 3, 1, 2)))  # Apply NNabla's Warping after converting NHWC ==> NCHW format
    warp_data = F.transpose(warp_data, (0, 2, 3, 1))  # NCHW ==> NHWC format

    return warp_data


def space_to_depth(input):
    """
    Apply space-to-depth transform
    input  : nn.Variable of shape (B, H*4, W*4, 3)
    output : nn.Variable of shape (B, H, W, 3*4*4)
    """
    batch, height, width, depth = input.shape
    output = F.reshape(
        input, (batch, height/4, 4, width/4, 4, depth), inplace=False)
    output = F.reshape(F.transpose(output, (0, 1, 3, 2, 4, 5)),
                       (batch, height/4, width/4, -1), inplace=False)
    return output


def space_to_depth_disc(input, t_batch, inplace=False):
    if input.ndim == 4:
        batch, height, width, depth = input.shape
    if input.ndim == 5:
        batch, f, height, width, depth = input.shape

    output = F.reshape(
        input, (t_batch, 3, height, width, depth), inplace=inplace)
    output = F.reshape(F.transpose(output, (0, 2, 3, 4, 1)),
                       (t_batch, height, width, 3*3), inplace=False)
    return output


def upscale_four(inputs, scope='upscale_four'):
    """
    Mimic the tensorflow bilinear-upscaling for a fix ratio of 4.
    """
    with nn.parameter_scope(scope):
        b, h, w, c = inputs.shape

        p_inputs = F.concatenate(
            inputs, inputs[:, -1:, :, :], axis=1)  # pad bottom
        p_inputs = F.concatenate(
            p_inputs, p_inputs[:, :, -1:, :], axis=2)  # pad right

        hi_res_bin = [
            [
                    inputs,  # top-left
                    p_inputs[:, :-1, 1:, :]  # top-right
            ],
            [
                    p_inputs[:, 1:, :-1, :],  # bottom-left
                    p_inputs[:, 1:, 1:, :]  # bottom-right
            ]
            ]

        hi_res_array = []
        for hi in range(4):
            for wj in range(4):
                hi_res_array.append(
                        hi_res_bin[0][0] *
                            (1.0 - 0.25 * hi) * (1.0 - 0.25 * wj)
                        + hi_res_bin[0][1] * (1.0 - 0.25 * hi) * (0.25 * wj)
                        + hi_res_bin[1][0] * (0.25 * hi) * (1.0 - 0.25 * wj)
                        + hi_res_bin[1][1] * (0.25 * hi) * (0.25 * wj)
                        )

        hi_res = F.stack(*hi_res_array, axis=3)  # shape (b,h,w,16,c)
        hi_res_reshape = F.reshape(hi_res, (b, h, w, 4, 4, c))
        hi_res_reshape = F.transpose(hi_res_reshape, (0, 1, 3, 2, 4, 5))
        hi_res_reshape = F.reshape(hi_res_reshape, (b, h*4, w*4, c))

    return hi_res_reshape


def bicubic_four(inputs, scope='bicubic_four'):
    """
    Equivalent to tf.image.resize_bicubic( inputs, (h*4, w*4) ) for a fix ratio of 4 FOR API <=1.13
    For API 2.0, tf.image.resize_bicubic will be different, old version is tf.compat.v1.image.resize_bicubic
    **Parallel Catmull-Rom Spline Interpolation Algorithm for Image Zooming Based on CUDA*[Wu et. al.]**
    """
    with nn.parameter_scope(scope):
        b, h, w, c = inputs.shape

        p_inputs = F.concatenate(
            inputs[:, :1, :, :], inputs, axis=1)  # pad top
        p_inputs = F.concatenate(
            p_inputs[:, :, :1, :], p_inputs, axis=2)  # pad left
        p_inputs = F.concatenate(
            p_inputs, p_inputs[:, -1:, :, :], p_inputs[:, -1:, :, :], axis=1)  # pad bottom
        p_inputs = F.concatenate(
            p_inputs, p_inputs[:, :, -1:, :], p_inputs[:, :, -1:, :], axis=2)  # pad right

        hi_res_bin = [p_inputs[:, bi:bi+h, :, :] for bi in range(4)]
        r = 0.75
        mat = np.float32([[0, 1, 0, 0], [-r, 0, r, 0],
                          [2*r, r-3, 3-2*r, -r], [-r, 2-r, r-2, r]])
        weights = [np.float32([1.0, t, t*t, t*t*t]).dot(mat)
                   for t in [0.0, 0.25, 0.5, 0.75]]

        hi_res_array = []  # [hi_res_bin[1]]
        for hi in range(4):
            cur_wei = weights[hi]
            cur_data = cur_wei[0] * hi_res_bin[0] + cur_wei[1] * hi_res_bin[1] + \
                cur_wei[2] * hi_res_bin[2] + cur_wei[3] * hi_res_bin[3]
            hi_res_array.append(cur_data)
        hi_res_y = F.stack(*hi_res_array, axis=2)  # shape (b,h,4,w,c)
        hi_res_y = F.reshape(hi_res_y, (b, h*4, w+3, c))
        hi_res_bin = [hi_res_y[:, :, bj:bj+w, :] for bj in range(4)]

        hi_res_array = []  # [hi_res_bin[1]]
        for hj in range(4):
            cur_wei = weights[hj]
            cur_data = cur_wei[0] * hi_res_bin[0] + cur_wei[1] * hi_res_bin[1] + \
                cur_wei[2] * hi_res_bin[2] + cur_wei[3] * hi_res_bin[3]
            hi_res_array.append(cur_data)
        hi_res = F.stack(*hi_res_array, axis=3)  # shape (b,h*4,w,4,c)
        hi_res = F.reshape(hi_res, (b, h*4, w*4, c))

    return hi_res
