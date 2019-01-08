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

from config import wavenet_config, data_config


def causal_padding(x, kernel_size, dilation):
    # x.shape = (..., channels, time)
    padding_size = (kernel_size - 1) * dilation

    return F.pad(x, (padding_size, 0), mode="constant", constant_value=0)


class waveNet(object):

    def __init__(self):
        self.hidden_dims = wavenet_config.hidden_dims
        self.skip_dims = wavenet_config.skip_dims
        self.kernel_size = wavenet_config.kernel_size
        self.dilations = wavenet_config.dilations

        self.output_channels = data_config.q_bit_len

    def residual_block(self, x, dilation):
        # x.shape = (Batch, channels, Time)

        # padding
        pad = causal_padding(x, self.kernel_size, dilation)

        # gated convolution
        with nn.parameter_scope("filter"):
            f = F.tanh(PF.convolution(pad, self.hidden_dims, kernel=(self.kernel_size, ), dilation=(dilation,)))

        with nn.parameter_scope("gate"):
            g = F.sigmoid(PF.convolution(pad, self.hidden_dims, kernel=(self.kernel_size, ), dilation=(dilation,)))

        h = f * g

        # 1 x 1 convolution
        with nn.parameter_scope("skip"):
            skip = PF.convolution(h, self.skip_dims, kernel=(1, ))

        out = x + skip

        return out, skip

    def call(self, x):
        # causal convolution
        with nn.parameter_scope("causal"):
            pad = causal_padding(x, kernel_size=2, dilation=1)
            current = PF.convolution(pad, self.skip_dims, kernel=(2, ), dilation=(1,))

        # residual
        skips = []
        for index, dilation in enumerate(self.dilations):
            with nn.parameter_scope("residual_{}".format(index)):
                current, skip = self.residual_block(current, dilation)
            skips.append(skip)

        # output
        out = F.relu(sum(skips))
        with nn.parameter_scope("out1"):
            y1 = F.relu(PF.convolution(out, self.skip_dims, kernel=(1,)))
        with nn.parameter_scope("out2"):
            y2 = F.softmax(PF.convolution(y1, self.output_channels, kernel=(1, )))

        return y2

    def __call__(self, x):
        with nn.parameter_scope("WaveNet"):
            y = self.call(x)

        return y

