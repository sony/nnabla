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

from __future__ import division, absolute_import

import os

import numpy as np


def normalize_intensity(audio_arr, input_low, input_high):
    denominator = input_high - input_low

    if denominator == 0:
        # all values are same
        if input_low == 0:
            return audio_arr  # output all values are 0
        else:
            return audio_arr / input_low  # output all values are 1

    return (audio_arr - input_low) / denominator


def rescale_intensity(audio_arr, input_low=0, input_high=255, output_low=0, output_high=255, output_type=None):
    if not isinstance(audio_arr, np.ndarray):
        raise ValueError(
            "rescale_intensity() supports only numpy.ndarray as input")

    input_low = np.float64(input_low)
    input_high = np.float64(input_high)
    output_low = np.float64(output_low)
    output_high = np.float64(output_high)
    if output_type is None:
        output_type = audio_arr.dtype

    if input_low == output_low and output_low == output_high:
        return np.asarray(audio_arr, output_type)

    # [input_low, input_high] -> [0, input_high - input_low] -> [0, 1]
    normalized = normalize_intensity(audio_arr, input_low, input_high)

    # [0, 1] -> [0, output_high - output_low] -> [output_low, output_high]
    scaled = normalized * (output_high - output_low) + output_low

    return scaled.astype(output_type)


def _auto_scale_before(audio_arr, datatype):
    if not isinstance(audio_arr, np.ndarray):
        raise ValueError(
            "the input audio_arr for rescaling must be numpy.ndarray.")

    if not isinstance(datatype, (np.dtype, type)):
        raise ValueError(
            "the out_datatype for rescaling must be numpy.dtype or type.")

    if np.dtype(datatype).itemsize > 4:
        raise ValueError(
            "the itemsize of out_datatype for rescaling should not greater than 4.")


def _auread_before(source, raw_format_param):
    if source.ext not in ['.wav', '.raw']:
        raise ValueError("path should contains filename extension and"
                         " only .raw .wav formats are now supported.")

    if source.ext == '.raw' and raw_format_param is None:
        raise ValueError("raw_format_param which contains sample_width, frame_rate, and channels"
                         " need to be provided when read raw format audio file.")


def _auread_after(audio, channel_first):
    # after-process for auread. resize and transpose audio.
    if channel_first and len(audio.shape) == 2:
        audio = audio.transpose((1, 0))

    return audio


def _ausave_before(path, audio_arr, channel_first):
    if not isinstance(audio_arr, np.ndarray):
        raise ValueError(
            "the input audio_arr for ausave must be numpy.ndarray.")

    if len(audio_arr.shape) != 2:
        raise ValueError("size must be (samples, channels) liked")

    filepath = path if isinstance(path, str) else path.name

    npath = os.path.abspath(filepath)
    dir_name = os.path.dirname(npath)
    os.makedirs(dir_name, exist_ok=True)

    ext = os.path.splitext(filepath)[-1]
    if ext not in ['.wav']:
        raise ValueError("only support save as .wav format for now.")

    if audio_arr.dtype.itemsize > 4:
        raise ValueError(
            "audio_arr's dtype.itemsize should not greater than 4")

    if channel_first and len(audio_arr.shape) == 2:
        audio_arr = audio_arr.transpose((1, 0))

    return audio_arr


def _auresize_before(audio_arr, size, channel_first):
    if not isinstance(audio_arr, np.ndarray):
        raise ValueError(
            "the input audio_arr for auresize must be numpy.ndarray.")

    if not isinstance(size, (list, tuple)):
        raise ValueError("size must be list or tuple")

    if len(size) != 2:
        raise ValueError("size must be (samples, channels) liked")

    if audio_arr.dtype not in [np.uint8, np.int16]:
        audio_arr = np.asarray(audio_arr, np.float32)

    if channel_first and len(audio_arr.shape) == 2:
        audio_arr = audio_arr.transpose((1, 0))

    return audio_arr
