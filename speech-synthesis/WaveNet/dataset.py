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

from __future__ import division

import os
import fnmatch
import numpy as np
import soundfile as sf
import librosa

from nnabla.logger import logger
from nnabla.utils.data_iterator import data_iterator
from nnabla.utils.data_source import DataSource

from config import data_config


def mu_law_encode(x, quantize=data_config.q_bit_len):
    """
    Applies mu-law encoding algorithm and quantization for an input signal.

    Args:
        x (numpy.ndarray): An array of a signal.
        quantize (int): A bit length of quantization.

    Returns:
         numpy.ndarray: An array of mu-law encoded signal.
    """
    if not isinstance(x, np.ndarray):
        raise ValueError("Input 'x' must be numpy.ndarray")

    if x.max() > 1. or x.min() < -1:
        raise ValueError("All values of 'x' must be between [-1, 1]")

    # The range of return values is between [0, mu] of int32

    mu = quantize - 1

    compressed = np.sign(x) * np.log(1 + mu * np.abs(x)) / \
        np.log(1 + mu)  # [-1, 1]

    # [0, mu]
    return np.asarray((compressed + 1) / 2 * mu + 0.5, dtype=np.int32)


def mu_law_decode(x, quantize=data_config.q_bit_len):
    """
    Applies dequantization and mu-law decoding algorithm for an input signal.

    Args:
        x (numpy.ndarray): A data array of mu-law encoded signal.
        quantize (int): A bit length of quantization.

    Returns:
         numpy.ndarray: An array of a signal (which is mu-law decoded).
    """
    mu = quantize - 1

    if not isinstance(x, np.ndarray):
        raise ValueError("Input 'x' must be numpy.ndarray")

    if x.max() > mu or x.min() < 0:
        raise ValueError(
            "All values of 'x' must be between [0, (quantization bit length) - 1]")

    # The range of return values is between [-1, 1] of float64

    normalized = (x / mu) * 2 - 1

    return np.asarray(np.sign(normalized) * ((1 + mu) ** np.abs(normalized) - 1) / mu, dtype=np.float64)


class LibriSpeechDataSource(DataSource):
    def __init__(self, data_dir, shuffle=True, rng=None):
        super(LibriSpeechDataSource, self).__init__(shuffle=shuffle, rng=rng)

        self._data_list = self.get_all_path_and_label(data_dir)

        speaker_list = sorted(set([x[1] for x in self._data_list]))

        self.n_speaker = len(speaker_list)
        self.speaker2id = dict(zip(speaker_list, np.arange(self.n_speaker)))

        self._size = len(self._data_list)
        self._variables = ('audio_input', 'speaker_id', 'y')

        self.duration = data_config.duration + 1

        self.reset()

    @staticmethod
    def get_all_path_and_label(data_dir):
        ret = []

        for dirpath, dirnames, filenames in os.walk(data_dir):
            flac_files = fnmatch.filter(filenames, "*.flac")
            if len(flac_files) == 0:
                continue

            # each element of ret is (path, speaker_id, script_id)
            for filename in flac_files:
                speaker_id, script_id = filename.split("-")[:2]
                ret.append(
                    (os.path.join(dirpath, filename), speaker_id, script_id))

        return ret

    def _get_data(self, position):
        index = self._indexes[position]

        flac_path, speaker, _ = self._data_list[index]

        # all values of data are between [-1, 1]
        data, sr = sf.read(flac_path)

        data, _ = librosa.effects.trim(data, top_db=20)

        # clip
        if len(data) < self.duration:
            lack = self.duration - len(data)
            before = lack // 2
            after = lack // 2 + lack % 2
            clipped = np.pad(data, pad_width=(before, after), mode="constant")

        else:
            start = np.random.randint(0, len(data) - self.duration)
            clipped = data[start:start + self.duration]

        # shape of clipped == (T,)

        quantized = mu_law_encode(clipped)

        _input = quantized[:-1].reshape(-1, 1)
        _speaker_id = self.speaker2id.get(speaker)

        _output = quantized[1:].reshape(-1, 1)

        return _input, _speaker_id, _output

    def reset(self):
        self._indexes = self._rng.permutation(
            self._size) if self._shuffle else np.arange(self._size)

        super(LibriSpeechDataSource, self).reset()


def data_iterator_librispeech(batch_size, data_dir, shuffle=True, rng=None,
                              with_memory_cache=False, with_file_cache=False):
    return data_iterator(LibriSpeechDataSource(data_dir, shuffle=shuffle, rng=rng),
                         batch_size,
                         rng,
                         with_memory_cache,
                         with_file_cache)
