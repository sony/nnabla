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

import os
import numpy as np
import librosa

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import nnabla.utils.save as save

from nnabla.logger import logger
from nnabla.monitor import Monitor, MonitorSeries
from _checkpoint_nnp_util import save_checkpoint, load_checkpoint, save_nnp

from model import WaveNet
from args import get_args
from config import data_config, WavenetConfig
from dataset import data_iterator_librispeech, mu_law_decode


def save_audio(xs, iter, save_dir):
    for index, audio in enumerate(xs):
        librosa.output.write_wav(os.path.join(save_dir, "result_{}_{}.wav".format(iter, index)), audio,
                                 sr=data_config.sample_rate)


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
    _data_source = data_iterator._data_source  # dirty hack...

    # model
    x = nn.Variable(
        shape=(args.batch_size, data_config.duration, 1))  # (B, T, 1)
    onehot = F.one_hot(x, shape=(data_config.q_bit_len, ))  # (B, T, C)
    wavenet_input = F.transpose(onehot, (0, 2, 1))  # (B, C, T)

    # speaker embedding
    if args.use_speaker_id:
        s_id = nn.Variable(shape=(args.batch_size, 1))
        with nn.parameter_scope("speaker_embedding"):
            s_emb = PF.embed(s_id, n_inputs=_data_source.n_speaker,
                             n_features=WavenetConfig.speaker_dims)
            s_emb = F.transpose(s_emb, (0, 2, 1))
    else:
        s_emb = None

    net = WaveNet()
    wavenet_output = net(wavenet_input, s_emb)

    pred = F.transpose(wavenet_output, (0, 2, 1))

    # (B, T, 1)
    t = nn.Variable(shape=(args.batch_size, data_config.duration, 1))

    loss = F.mean(F.softmax_cross_entropy(pred, t))

    # for generation
    prob = F.softmax(pred)

    # Create Solver.
    solver = S.Adam(args.learning_rate)
    solver.set_parameters(nn.get_parameters())

    # load checkpoint
    start_point = 0
    if args.checkpoint is not None:
        # load weights and solver state info from specified checkpoint file.
        start_point = load_checkpoint(args.checkpoint, solver)

    # Create monitor.
    monitor = Monitor(args.monitor_path)
    monitor_loss = MonitorSeries("Training loss", monitor, interval=10)

    # setup save env.
    audio_save_path = os.path.join(os.path.abspath(
        args.model_save_path), "audio_results")
    if audio_save_path and not os.path.exists(audio_save_path):
        os.makedirs(audio_save_path)

    # save_nnp
    contents = save_nnp({'x': x}, {'y': wavenet_output}, args.batch_size)
    save.save(os.path.join(args.model_save_path,
                           'Speechsynthesis_result_epoch0.nnp'), contents)
    # Training loop.
    for i in range(start_point, args.max_iter):
        # todo: validation

        x.d, _speaker, t.d = data_iterator.next()
        if args.use_speaker_id:
            s_id.d = _speaker.reshape(-1, 1)

        solver.zero_grad()
        loss.forward(clear_no_need_grad=True)
        loss.backward(clear_buffer=True)
        solver.update()

        loss.data.cast(np.float32, ctx)
        monitor_loss.add(i, loss.d.copy())

        if i % args.model_save_interval == 0:
            prob.forward()
            audios = mu_law_decode(
                np.argmax(prob.d, axis=-1), quantize=data_config.q_bit_len)  # (B, T)
            save_audio(audios, i, audio_save_path)
            # save checkpoint file
            save_checkpoint(audio_save_path, i, solver)

    # save_nnp
    contents = save_nnp({'x': x}, {'y': wavenet_output}, args.batch_size)
    save.save(os.path.join(args.model_save_path,
                           'Speechsynthesis_result.nnp'), contents)


if __name__ == '__main__':
    train()
