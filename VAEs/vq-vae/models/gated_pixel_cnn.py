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

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF

import numpy as np


class GatedPixelCNN(object):

    def __init__(self, config):
        self.out_channels = config['out_channels']
        self.num_layers = config['num_layers']
        self.num_features = config['num_features']
        self.num_classes = config['num_classes']
        self.conditional = config['conditional']

        self.input_shape = config['latent_shape']

    def mask_type_A(self, W):
        c_x, c_y = W.shape[2]//2, W.shape[3]//2

        mask = np.ones(W.shape)
        mask[:, :, c_x, c_y+1:] = 0
        mask[:, :, c_x+1:, :] = 0
        mask[:, :, c_x, c_y] = 0
        mask = nn.Variable.from_numpy_array(mask)
        W = mask*W
        return W

    def mask_type_B(self, W):
        c_x, c_y = W.shape[2]//2, W.shape[3]//2

        mask = np.ones(W.shape)
        mask[:, :, c_x, c_y+1:] = 0
        mask[:, :, c_x+1:, :] = 0
        mask = nn.Variable.from_numpy_array(mask)
        W = mask*W
        return W

    def gated_conv(self, x, kernel_shape, h=None, mask_type='', gated=True, payload=None, return_payload=False, scope_name='gated_conv'):
        pad_dim_0 = (kernel_shape[0]-1)/2
        pad_dim_1 = (kernel_shape[1]-1)/2
        if mask_type == '':
            mask_type = self.mask_type_B
        with nn.parameter_scope(scope_name):
            if gated:
                out_f = PF.convolution(x, self.num_features, kernel_shape,
                                       pad=(pad_dim_0, pad_dim_1), apply_w=mask_type, name='conv_f')
                out_g = PF.convolution(x, self.num_features, kernel_shape,
                                       pad=(pad_dim_0, pad_dim_1), apply_w=mask_type, name='conv_g')
                if isinstance(payload, nn.Variable):
                    out_f += payload[:, :self.num_features, :, :]
                    out_g += payload[:, self.num_features:, :, :]
                if self.conditional:
                    h_out_f = PF.affine(
                            h, self.num_features, name='h_out_f')
                    h_out_f = h_out_f.reshape(
                            (h_out_f.shape[0], h_out_f.shape[1], 1, 1))
                    h_out_g = PF.affine(
                            h, self.num_features, name='h_out_g')
                    h_out_g = h_out_g.reshape(
                            (h_out_g.shape[0], h_out_g.shape[1], 1, 1))
                    out = F.tanh(out_f+h_out_f) * F.sigmoid(out_g+h_out_g)
                else:
                    out = F.tanh(out_f) * F.sigmoid(out_g)
                if return_payload:
                    payload = PF.convolution(F.concatenate(
                        out_f, out_g, axis=1), 2*self.num_features, (1, 1), name='conv_1x1')
                    payload = F.relu(payload)
                    return out, payload

            else:
                out = PF.convolution(x, self.num_features, kernel_shape, stride=(1, 1),
                                     pad=(pad_dim_0, pad_dim_1), apply_w=mask_type)
                out = F.relu(out)

        return out

    def __call__(self, conv_in, h=None):

        v_stack_in = conv_in
        h_stack_in = conv_in

        features = []
        with nn.parameter_scope('ConditionalPixelCNN'):
            for i in range(self.num_layers):
                if i == 0:
                    kernel_shape = (7, 7)
                    mask_type = self.mask_type_A
                    residual = False
                else:
                    kernel_shape = (3, 3)
                    mask_type = self.mask_type_B
                    residual = True

                v_stack_gated, v_stack_conv = self.gated_conv(v_stack_in, kernel_shape, h, mask_type=mask_type, return_payload=True,
                                                              scope_name='vertical_stack_gated_'+str(i))
                h_stack_gated = self.gated_conv(h_stack_in, (1, kernel_shape[0]), h, mask_type=mask_type,
                                                payload=v_stack_conv, scope_name='horizontal_stack_gated_'+str(i))
                h_stack_conv = self.gated_conv(h_stack_gated, (1, 1), h, mask_type=mask_type, gated=False,
                                               scope_name='horizontal_stack_conv_'+str(i))
                if residual:
                    h_stack_conv += h_stack_in

                v_stack_in = v_stack_gated
                h_stack_in = h_stack_conv

            fc_1 = self.gated_conv(
                    h_stack_in, (1, 1), gated=False, scope_name='fc_1')
            fc_2 = PF.convolution(fc_1, self.out_channels,
                                  (1, 1), apply_w=self.mask_type_B, name='fc_2')

        fc_2 = F.transpose(fc_2, (0, 2, 3, 1))
        fc_2 = F.reshape(fc_2, (-1, fc_2.shape[-1]), inplace=True)

        return fc_2
