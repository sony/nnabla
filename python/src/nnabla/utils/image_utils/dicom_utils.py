# Copyright (c) 2019 Sony Corporation. All Rights Reserved.
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
import pydicom
from pydicom.dataset import Dataset, FileDataset
import numpy as np
import datetime
import random
import cv2

from nnabla.logger import logger
from .common import upscale_pixel_intensity, check_type_and_cast_if_necessary,\
    _imread_before, _imread_after, _imsave_before


DEFAULT_LITTLE_ENDIAN = True

# Using cv2.resize, since only cv2.resize support int16
from .cv2_utils import imresize


def _convert_channel_from_gray(img, num_channels):
    if num_channels in [-1, 0]:
        return img

    elif num_channels == 1:
        return img[..., np.newaxis]

    elif num_channels == 3:  # GRAY => RGB (just expand)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    elif num_channels == 4:  # GRAY => RGBA (just expand)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGBA)

    raise ValueError("num_channels must be [-1, 0, 1, 3, 4]")


def _convert_channel_from_rgb(img, num_channels):
    if num_channels in [0, 1]:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if num_channels == 1:
            img = img[..., np.newaxis]

        return img

    elif num_channels in [-1, 3]:  # RGB => RGB
        return img

    elif num_channels == 4:  # RGB => RGBA
        return cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)

    raise ValueError("num_channels must be [-1, 0, 1, 3, 4]")


def _convert_channel_from_rgba(img, num_channels):
    if num_channels in [0, 1]:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        if num_channels == 1:
            img = img[..., np.newaxis]

        return img

    elif num_channels == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    elif num_channels in [-1, 4]:  # RGBA => RGBA
        return img

    raise ValueError("num_channels must be [-1, 0, 1, 3, 4]")


def _cvtColor_helper(img, num_channels):
    # convert num_channels which means (h, w, c) -> (h, w, num_channels)
    if len(img.shape) == 2:
        return _convert_channel_from_gray(img, num_channels)

    elif img.shape[-1] == 3:
        return _convert_channel_from_rgb(img, num_channels)

    elif img.shape[-1] == 4:
        return _convert_channel_from_rgba(img, num_channels)

    raise ValueError("Bad image shape. ({})".format(img.shape))


def imread(path, grayscale=False, size=None, interpolate="bilinear",
           channel_first=False, as_uint16=False, num_channels=-1):
    """
    Read image by pydicom module.

    Before using pydicom module, opencv needs to be installed.
    Args:
        path (str or 'file object'): File path or object to read.
        grayscale (bool):
        size (tupple of int):
            (width, height).
            If None, output img shape depends on the files to read.
        channel_first (bool):
            This argument specifies the shape of img is whether (height, width, channel) or (channel, height, width).
            Default value is False, which means the img shape is (height, width, channel).
        interpolate (str):
            must be one of ["nearest", "bilinear", "bicubic", "lanczos"].
        as_uint16 (bool):
            If you specify this argument, you can use only False for pil backend.
        num_channels (int):
            channel size of output array.
            Default is -1 which preserves raw image shape.

    Returns:
         numpy.ndarray
    """
    _imread_before(grayscale, num_channels)

    dicom_dataset = pydicom.dcmread(path)

    img = dicom_dataset.pixel_array
    if as_uint16 and img.dtype != np.uint16:
        if img.dtype == np.uint8:
            logger.warning("You want to read image as uint16, but the original bit-depth is 8 bit."
                           "All pixel values are simply increased by 256 times.")
            img = img.astype(np.uint16) * 256
        else:
            raise ValueError(
                "casting {} to uint16 is not safe.".format(img.dtype))

    if num_channels == -1 and grayscale:
        num_channels = 0
    img = _cvtColor_helper(img, num_channels)

    return _imread_after(img, size, interpolate, channel_first, imresize)


def imsave(path, img, channel_first=False, as_uint16=False, auto_scale=True):
    """
    Save image by dicom module.

    Args:
        path (str): output filename
        img (numpy.ndarray): Image array to save. Image shape is considered as (height, width, channel) by default.
        channel_first (bool):
            This argument specifies the shape of img is whether (height, width, channel) or (channel, height, width).
            Default value is False, which means the img shape is considered as (height, width, channel)
        as_uint16 (bool):
            In this backend, this argument is always False because pillow dose not support uint16.
            If True, exception will be raised.
        auto_scale (bool) :
            Whether upscale pixel values or not.
            If you want to save float image, this argument must be True.
    """
    img = _imsave_before(img, channel_first, auto_scale)
    if auto_scale:
        img = upscale_pixel_intensity(img, as_uint16)
    img = check_type_and_cast_if_necessary(img, as_uint16)

    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
    file_meta.MediaStorageSOPInstanceUID = "1.2.3"
    file_meta.ImplementationClassUID = "1.2.3.4"
    file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian

    ds = FileDataset(path, {},
                     file_meta=file_meta, preamble=b"\0" * 128)

    ds.PatientName = "NNablaImageUtils^Autogen"
    ds.PatientID = ''.join(random.choice('01234567890') for i in range(10))

    ds.is_little_endian = True
    ds.is_implicit_VR = True

    dt = datetime.datetime.now()
    ds.ContentDate = dt.strftime('%Y%m%d')
    timeStr = dt.strftime('%H%M%S.%f')  # long format with micro seconds
    ds.ContentTime = timeStr
    if len(img.shape) == 2:
        ds.Rows, ds.Columns = img.shape
        ds.SamplesPerPixel = 1
    else:
        ds.Rows, ds.Columns, ds.SamplesPerPixel = img.shape
    ds.PixelData = img.tobytes()
    ds.BitsAllocated = 16 if as_uint16 else 8  # 8 or 16 bits
    ds.PixelRepresentation = 0  # unsigned
    ds.PlanarConfiguration = 0   # W, H, C style

    ds.save_as(path)
