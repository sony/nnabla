# Copyright 2020,2021 Sony Corporation.
# Copyright 2021 Sony Group Corporation.
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

from __future__ import absolute_import, division

import math
import os
import warnings

import numpy as np

warnings.simplefilter("ignore", RuntimeWarning)
from pydub import AudioSegment
warnings.simplefilter("default", RuntimeWarning)


from .common import _auread_before, _auread_after, _ausave_before, _auresize_before


def get_nparray_from_pydub(audio_segment):
    # shape is (samples, channels) liked
    audio_arr = audio_segment.get_array_of_samples()
    return np.array(audio_arr).reshape(-1, audio_segment.channels)


def get_audiosegment_from_nparray(nparr, frame_rate=48000):
    # nparr is (samples, channels) liked
    audio_segment = AudioSegment(nparr.tobytes(),
                                 frame_rate=frame_rate,
                                 sample_width=nparr.dtype.itemsize,
                                 channels=nparr.shape[1])
    return audio_segment


def auread(source, channel_first=False, raw_format_param=None):
    """
    Read audio with pydub module.
    Currently only support .wav format audio, .raw format audio could be read only when
    additional params are provided.

    Args:
        source (class ResourceFileReader): source handler.
        channel_first (bool):
            This argument specifies the shape of audio is whether (samples, channels) or (channels, samples).
            Default value is False, which means the audio shape shall be (samples, channels).
        raw_format_param(object):
            If audio is raw format, user should provide this object, example:
            { 'sample_width': 2, 'channels': 2, 'frame_rate': 44100 }

    Returns:
         numpy.ndarray
    """

    _auread_before(source, raw_format_param)

    audio_format = source.ext[1:]
    with source.open() as f:
        if audio_format == 'raw':
            audio = AudioSegment.from_file(
                f, format=audio_format, **raw_format_param)
        else:
            audio = AudioSegment.from_file(f, format=audio_format)

    audio_arr = get_nparray_from_pydub(audio)
    if audio_arr.dtype.itemsize == 1 and audio_format == 'wav':
        # 8-bit wav file value should in uint8 range, but pydub read it as int8
        audio_arr = audio_arr.astype(np.uint8)

    return _auread_after(audio_arr, channel_first)


def ausave(path, audio_arr, channel_first=False, frame_rate=48000):
    """
    Save audio with pydub module.

    Args:
        path (str): output path with filename.
                    Currently only support .wav format.
        audio_arr (numpy.ndarray): Audio array to save. Audio shape is considered as (samples, channels) by default.
                    Pydub does not support save .wav file with float value.
        channel_first (bool):
            This argument specifies the shape of audio is whether (samples, channels) or (channels, samples).
            Default value is False, which means the audio_arr shape is (samples, channels)
    """
    audio_arr = _ausave_before(path, audio_arr, channel_first)

    if str(audio_arr.dtype).find('float') > -1:
        raise ValueError(
            "pydub backend does not support save .wav file with float value.")
    filepath = path if isinstance(path, str) else path.name
    audio_format = os.path.splitext(filepath)[-1][1:]
    audio_segment = get_audiosegment_from_nparray(
        audio_arr, frame_rate=frame_rate)

    audio_segment.export(path, format=audio_format)


def auresize(audio_arr, size, channel_first=False):
    """
    Resize audio_arr by pydub module.
    Args:
        audio_arr (numpy.ndarray): Audio array to save.
            Audio shape is considered as (samples, channels) by default.
        size (tupple of int): Output shape. The order is (samples, channels).
        channel_first (bool):
            This argument specifies the shape of audio is whether (samples, channels) or (channels,
            samples).
            Default value is False, which means the audio_arr shape is (samples, channels)
    Returns:
         numpy.ndarray whose shape is (samples, channels) or (channels, samples)
    """

    audio = _auresize_before(audio_arr, size, channel_first)
    n_channel_num = size[1]
    n_sample_num = size[0]
    o_channel_num = audio.shape[1]
    o_sample_num = audio.shape[0]

    if o_channel_num != 1 and n_channel_num != 1:
        if o_channel_num != n_channel_num:
            raise ValueError("pydub set_channels only supports mono-to-multi channel"
                             " and multi-to-mono channel conversion")

    audio_segment = get_audiosegment_from_nparray(audio)
    new_rate = math.floor(48000 * n_sample_num / o_sample_num + 1)
    audio_segment = audio_segment.set_frame_rate(new_rate)
    audio_segment = audio_segment.set_channels(n_channel_num)
    audio_segment = audio_segment.get_sample_slice(0, n_sample_num)

    resized = get_nparray_from_pydub(audio_segment)

    return resized
