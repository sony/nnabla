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

import pytest
import numpy as np
from six import BytesIO, StringIO

try:
    from nnabla.utils import audio_utils
except ImportError:
    pytest.skip('no available backend for audio_utils',
                allow_module_level=True)
from nnabla.logger import logger
from nnabla.testing import assert_allclose

# (samples, channels)
SIZE = (100000, 2)

audios = [
    np.random.randint(low=0, high=255, size=SIZE).astype(np.uint8),
    np.random.randint(low=np.iinfo(np.int16).min,
                      high=np.iinfo(np.int16).max, size=SIZE).astype(np.int16),
    np.random.randint(low=np.iinfo(np.int32).min,
                      high=np.iinfo(np.int32).max, size=SIZE).astype(np.int32),
    np.random.uniform(-1, 1, size=SIZE).astype(np.float32)
]


def _change_backend(backend):
    if backend not in audio_utils.get_available_backends():
        pytest.skip("{} is not installed".format(backend))

    # change backend arbitrary
    audio_utils.set_backend(backend)
    assert audio_utils.get_backend() == backend


def test_audio_utils_backend_manager():
    with pytest.raises(ValueError):
        audio_utils.set_backend("not_installed_module")


def check_save_condition(backend, audio_arr):
    if audio_arr.dtype.itemsize > 4:
        return False
    if backend == 'pydub' and str(audio_arr.dtype).find('float') > -1:
        return False

    return True


@pytest.mark.parametrize("audio", [
    np.zeros(shape=SIZE),
    np.ones(shape=SIZE) * 10,
    np.ones(shape=SIZE) * -10,
    np.random.random(size=SIZE) * 10,
    np.random.random(size=SIZE) * -10,
])
@pytest.mark.parametrize("out_datatype", [
    np.uint8,
    np.int16,
    np.int32,
    np.float32,
])
def test_minmax_auto_scale(audio, out_datatype):
    rescaled = audio_utils.minmax_auto_scale(audio, out_datatype)

    # check dtype
    if str(out_datatype).find('int') > -1:
        scale_max = np.iinfo(out_datatype).max
        scale_min = np.iinfo(out_datatype).min
    else:
        scale_max = np.finfo(out_datatype).max
        scale_min = np.finfo(out_datatype).min

    assert rescaled.dtype == out_datatype

    # check the range of values
    assert rescaled.max() <= scale_max
    assert rescaled.min() >= scale_min

    # check if correctly scaled
    reverted = audio_utils.common.rescale_intensity(rescaled, rescaled.min(), rescaled.max(),
                                                    audio.min(), audio.max(), audio.dtype)
    assert_allclose(audio, reverted, atol=1 / 2 ** 4)


@pytest.mark.parametrize("backend", ["pydub"])
@pytest.mark.parametrize("channel_first", [False, True])
@pytest.mark.parametrize("audio", audios)
@pytest.mark.parametrize("source_type", ['string', 'binaryFileHandler', 'BytesIO', 'StringIO', 'strFileHandler'])
def test_ausave_and_auread(tmpdir, backend, channel_first, audio, source_type):

    _change_backend(backend)

    tmpdir.ensure(dir=True)
    tmppath = tmpdir.join("tmp.wav")
    audio_file_path = tmppath.strpath

    if channel_first:
        audio = audio.transpose((1, 0))

    # do ausave
    def save_audio_function(audio_file_path):
        audio_utils.ausave(audio_file_path, audio, channel_first=channel_first)

    if check_save_condition(backend, audio):
        save_audio_function(audio_file_path)
    else:
        with pytest.raises(ValueError):
            save_audio_function(audio_file_path)

        return True

    # do auread
    def read_audio_function(source):
        return audio_utils.auread(source, channel_first=channel_first)

    # 'string', 'binaryFileHandler', 'BytesIO', 'StringIO', 'strFileHandler'
    if source_type == 'string':
        read_audio = read_audio_function(audio_file_path)
    elif source_type == 'binaryFileHandler':
        with open(audio_file_path, 'rb') as f:
            read_audio = read_audio_function(f)
    elif source_type == 'BytesIO':
        with open(audio_file_path, 'rb') as f:
            read_audio = read_audio_function(BytesIO(f.read()))
    elif source_type == 'StringIO':
        with pytest.raises(ValueError):
            read_audio = read_audio_function(StringIO(audio_file_path))
        return True
    elif source_type == 'strFileHandler':
        with pytest.raises(ValueError):
            with open(audio_file_path, 'r') as f:
                read_audio = read_audio_function(f)
        return True

    logger.info(read_audio.shape)
    # ---check size and channels---
    assert read_audio.shape == audio.shape

    # ---check dtype---
    assert read_audio.dtype == audio.dtype

    # ---check close between before ausave and after auread---
    assert_allclose(audio, read_audio)


@pytest.mark.parametrize("backend", ["pydub"])
@pytest.mark.parametrize("size", [(50000, 2), (150000, 2), (200000, 1)])
@pytest.mark.parametrize("channel_first", [False, True])
@pytest.mark.parametrize("audio", audios)
def test_auresize(backend, size, channel_first, audio):
    _change_backend(backend)

    if channel_first:
        audio = audio.transpose((1, 0))

    resized_audio = audio_utils.auresize(
        audio, size, channel_first=channel_first)

    assert resized_audio.shape == size
