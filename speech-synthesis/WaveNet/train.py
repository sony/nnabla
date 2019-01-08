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

from __future__ import absolute_import

import nnabla as nn
import nnabla.functions as F
import nnabla.solvers as S
from nnabla.logger import logger

from model import waveNet
from args import get_args
from config import data_config
from dataset import data_iterator_librispeech


def train():
    args = get_args()

    # Set context.
    from nnabla.ext_utils import get_extension_context
    logger.info("Running in {}:{}".format(args.context, args.type_config))
    ctx = get_extension_context(args.context,
                                device_id=args.device_id,
                                type_config=args.type_config)
    nn.set_default_context(ctx)

    data_iterator = data_iterator_librispeech(args.batch_size, args.data_dir)

    x = nn.Variable(shape=(args.batch_size, data_config.duration, 1))  # (B, T, 1)
    onehot = F.one_hot(x, shape=(data_config.q_bit_len, ))  # (B, T, C)
    wavenet_input = F.transpose(onehot, (0, 2, 1))  # (B, C, T)

    net = waveNet()
    wavenet_output = net(wavenet_input)

    pred = F.transpose(wavenet_output, (0, 2, 1))

    # (B, T, 1)
    t = nn.Variable(shape=(args.batch_size, data_config.duration, 1))

    loss = F.mean(F.softmax_cross_entropy(pred, t))

    # Create Solver.
    solver = S.Adam(args.learning_rate)
    solver.set_parameters(nn.get_parameters())

    # Training loop.
    for i in range(args.max_iter):
        # todo: validation

        x.d, t.d = data_iterator.next()

        solver.zero_grad()
        loss.forward(clear_no_need_grad=True)
        loss.backward(clear_buffer=True)
        solver.update()

        if i % 100 == 0:
            logger.info("iter: {}, loss: {}".format(i, loss.d))


if __name__ == '__main__':
    train()
