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

from nnabla.utils import image_utils
from nnabla.logger import logger
from nnabla.testing import assert_allclose

SIZE = (8, 8, 3)

imgs = [
    np.random.randint(low=0, high=255, size=SIZE).astype(np.uint8),
    np.random.randint(low=0, high=65535, size=SIZE).astype(np.uint16),
    np.random.random(size=SIZE).astype(np.float),
    np.random.random(size=SIZE).astype(np.float) - 10,
    np.random.randint(low=0, high=255, size=SIZE[:-1]).astype(np.uint8),
]


def _change_backend(backend):
    if backend not in image_utils.get_available_backends():
        pytest.skip("{} is not installed".format(backend))

    # change backend arbitrary
    image_utils.set_backend(backend)
    assert image_utils.get_backend() == backend


def check_imsave_condition(backend, img, as_uint16, auto_scale):
    if img.dtype not in [np.uint8, np.uint16]:
        if not auto_scale:
            return False
        if np.any(img < 0) or np.any(img > 1):
            return False

    if backend == "PilBackend":
        if img.dtype == np.uint16 or as_uint16:
            return False
    else:
        if img.dtype == np.uint16 and not as_uint16:
            return False

    return True


def get_scale_factor(img, auto_scale, as_uint16):
    if auto_scale:
        if img.dtype == np.uint8:
            if as_uint16:
                return 256
        elif img.dtype != np.uint16:
            return 65535 if as_uint16 else 255

    return 1


@pytest.mark.parametrize("img", [
    np.zeros(shape=SIZE),
    np.ones(shape=SIZE) * 10,
    np.ones(shape=SIZE) * -10,
    np.random.random(size=SIZE) * 10,
    np.random.random(size=SIZE) * -10,
])
@pytest.mark.parametrize("as_uint16", [True, False])
def test_minmax_auto_scale(img, as_uint16):
    rescaled = image_utils.minmax_auto_scale(img, as_uint16)

    # check dtype
    if as_uint16:
        assert rescaled.dtype == np.uint16
        scale_max = 65535
    else:
        assert rescaled.dtype == np.uint8
        scale_max = 255

    # check the range of values
    assert rescaled.max() <= scale_max
    assert rescaled.min() >= 0

    # check if correctly scaled up
    reverted = image_utils.backend_events.common.rescale_pixel_intensity(rescaled, rescaled.min(), rescaled.max(),
                                                                         img.min(), img.max(), img.dtype)
    assert_allclose(img, reverted, atol=1 / 2 ** 4)


@pytest.mark.parametrize("backend", ["PilBackend", "PngBackend", "Cv2Backend"])
@pytest.mark.parametrize("grayscale", [False, True])
@pytest.mark.parametrize("size", [None, (16, 16)])
@pytest.mark.parametrize("channel_first", [False, True])
@pytest.mark.parametrize("as_uint16", [False, True])
@pytest.mark.parametrize("num_channels", [-1, 0, 1, 3, 4])
@pytest.mark.parametrize("auto_scale", [False, True])
@pytest.mark.parametrize("img", imgs)
def test_imsave_and_imread(tmpdir, backend, grayscale, size, channel_first, as_uint16, num_channels, auto_scale, img):
    # import pdb
    # pdb.set_trace()
    # preprocess
    _change_backend(backend)

    tmpdir.ensure(dir=True)
    tmppath = tmpdir.join("tmp.png")
    img_file = tmppath.strpath

    ref_size_axis = 0
    if channel_first and len(img.shape) == 3:
        img = img.transpose((2, 0, 1))
        ref_size_axis = 1

    # do imsave
    def save_image_function():
        image_utils.imsave(img_file, img, channel_first=channel_first,
                           as_uint16=as_uint16, auto_scale=auto_scale)

    if check_imsave_condition(backend, img, as_uint16, auto_scale):
        save_image_function()
    else:
        with pytest.raises(ValueError):
            save_image_function()

        return True

    # do imread
    def read_image_function():
        return image_utils.imread(img_file, grayscale=grayscale, size=size, channel_first=channel_first,
                                  as_uint16=as_uint16, num_channels=num_channels)

    if not grayscale and num_channels in [0, 1]:
        with pytest.raises(ValueError):
            _ = read_image_function()

        return True
    else:
        read_image = read_image_function()

    logger.info(read_image.shape)
    # ---check size---
    ref_size = img.shape[ref_size_axis:ref_size_axis +
                         2] if size is None else size
    size_axis = 1 if len(read_image.shape) == 3 and channel_first else 0
    assert read_image.shape[size_axis:size_axis + 2] == ref_size

    # ---check channels---
    if num_channels == 0 or (num_channels == -1 and (len(img.shape) == 2 or grayscale)):
        assert len(read_image.shape) == 2
    else:
        channel_axis = 0 if channel_first else -1
        ref_channels = num_channels if num_channels > 0 else img.shape[channel_axis]
        assert read_image.shape[channel_axis] == ref_channels

    # ---check dtype---
    if as_uint16 or img.dtype == np.uint16:
        assert read_image.dtype == np.uint16
    else:
        assert read_image.dtype == np.uint8

    # ---check close between before imsave and after imread---
    if size is None and not grayscale and img.shape == read_image.shape:
        scaler = get_scale_factor(img, auto_scale, as_uint16)
        dtype = img.dtype if img.dtype in [np.uint8, np.uint16] else np.float32

        assert_allclose(
            (img.astype(dtype) * scaler).astype(read_image.dtype), read_image)


@pytest.mark.parametrize("backend", ["PilBackend", "PngBackend", "Cv2Backend"])
@pytest.mark.parametrize("size", [(16, 16), (64, 64)])
@pytest.mark.parametrize("channel_first", [False, True])
@pytest.mark.parametrize("img", imgs)
def test_imresize(backend, size, channel_first, img):
    _change_backend(backend)

    channel_axis = 0
    if channel_first and len(img.shape) == 3:
        img = img.transpose((2, 0, 1))
        channel_axis = 1

    resized_img = image_utils.imresize(img, size, channel_first=channel_first)

    assert resized_img.shape[channel_axis:channel_axis + 2] == size
