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

from __future__ import absolute_import

import numpy as np

from .backend_manager import backend_manager
from .common import rescale_intensity, _auto_scale_before


def set_backend(backend):
    """
    Set utils.audio_utils' backend module.
    If the backend which you specify is not installed in your environment, Exception is raised.

    Args:
        backend (str): the name of audio_utils' backend
    """

    backend_manager.backend = backend


def get_backend():
    """
    Get audio_utils' backend module used now.

    Returns:
        str
    """

    return backend_manager.backend


def get_available_backends():
    """
    Get all available audio_utils' backend modules.

    Returns:
        list of str
    """

    return backend_manager.get_available_backends()


def minmax_auto_scale(audio_arr, out_datatype):
    """
    Utility function for rescaling all audio sample values of input audio to fit the range specified
    by out_datatype.

    Rescaling method is min-max, which is all audio sample values are normalized to [0, 1] by using 
    audio_arr.min() and audio_arr.max() and then are scaled up by certain times (eg. 255 if out_datatype is uint8).

    Args:
        audio_arr (numpy.ndarray): input audio ndarray.
        out_datatype (numpy.dtype): define data type as well as range after scaling. For example, np.uint8 or
                                    np.dtype('uint8'). Available options' dtype.itemsize should not greater than 4,
                                    such as uint8, int8, int16, int32, float32.

    Return:
        numpy.ndarray
    """

    _auto_scale_before(audio_arr, out_datatype)

    if str(out_datatype).find('int') > -1:
        output_high = np.iinfo(out_datatype).max
        output_low = np.iinfo(out_datatype).min
    else:
        output_high = np.finfo(out_datatype).max
        output_low = np.finfo(out_datatype).min
    output_type = out_datatype

    return rescale_intensity(audio_arr, input_low=audio_arr.min(),
                             input_high=audio_arr.max(), output_low=output_low,
                             output_high=output_high, output_type=output_type)


def auread(path, channel_first=False, raw_format_param=None,
           **kwargs):
    """
    Read audio from ''path''.
    Currently only support .wav format audio, .raw format audio could be read only when additional params
    are provided.  
    Default output shape is (samples, channels).

    Args:
        path (String, File Object, BytesIO): Input audio source.
        channel_first (bool): If True, the shape of the output array is (channels, samples)
                              for audio. Default is False.
        raw_format_param(object): "raw" files require 3 additional keyword arguments ->
                                  sample_width, frame_rate, and channels

                                  sample_width | example: 2 - Use 1 for 8-bit audio 2 for 16-bit and
                                        4 for 32-bit. It's the number of bytes per sample.
                                  channels | example: 1 - 1 for mono, 2 for stereo.
                                  frame_rate | example: 44100 - Also known as sample rate, common values
                                        are 44100 and 48000

                                  If audio is raw format, user should provide this object, example:
                                  { 'sample_width': 2, 'channels': 2, 'frame_rate': 44100 }


    Returns:
         numpy.ndarray
    """

    from nnabla.utils.data_source_loader import ResourceFileReader
    source = ResourceFileReader(path)
    return backend_manager.module.auread(source, channel_first=channel_first, raw_format_param=None,
                                         **kwargs)


def ausave(path, audio_arr, channel_first=False, frame_rate=48000, **kwargs):
    """
    Save ''audio_arr'' to the file specified by ''path''.
    As default, the shape of ''audio_arr'' has to be (samples, channels).

    Args:
        path (str): Output path which contains file name and format info.
            Currently only support save as .wav format.
        audio_arr (numpy.ndarray):
            Input audio ndarray whose dtype.itemsize should not greater than 4,
        channel_first (bool):
            If True, you can input the audio whose shape is (samples, channels). Default is False.
        frame_rate (int):
            Also known as sample rate. Default is 48000
    """

    backend_manager.module.ausave(
        path, audio_arr, channel_first=channel_first, frame_rate=frame_rate, **kwargs)


def auresize(audio_arr, size, channel_first=False, **kwargs):
    """
    Resize ''audio_arr'' to ''size''.
    As default, the shape of input audio has to be (samples, channels).

    Args:
        audio_arr (numpy.ndarray): Input audio.
        size (tuple of int): Output shape. The order is (samples, channels).
        channel_first (bool):
            This argument specifies the shape of audio is whether (samples, channels) or (channels, samples).
            Default value is False, which means the audio shape is (samples, channels)
    Returns:
         numpy.ndarray (same sample-channel order as audio_arr)
    """

    return backend_manager.module.auresize(audio_arr, size, channel_first=channel_first, **kwargs)


# alias
auwrite = ausave
auload = auread
