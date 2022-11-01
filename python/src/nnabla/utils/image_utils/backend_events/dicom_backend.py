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

import numpy as np
import pydicom

from .common import _imread_before, _imread_after
from .image_utils_backend import ImageUtilsBackend


def _apply_gamma_correction(dicom_dataset):
    d_min = 0
    d_max = 255
    bpp = dicom_dataset.BitsAllocated
    spp = dicom_dataset.SamplesPerPixel
    if 'WindowCenter' in dicom_dataset:
        win_center = float(dicom_dataset.WindowCenter)
    else:
        win_center = (1 << bpp) / 2
    if 'WindowWidth' in dicom_dataset:
        win_width = float(dicom_dataset.WindowWidth)
    else:
        win_width = (1 << bpp)
    ################ NCTB #######
    if 'PhotometricInterpretation' in dicom_dataset:
        photo_interpretation = dicom_dataset.PhotometricInterpretation
    else:
        photo_interpretation = 'MONOCHROME2'

    if 'RescaleSlope' in dicom_dataset:
        rescale_slope = float(dicom_dataset.RescaleSlope)
    else:
        rescale_slope = 1.0

    if 'RescaleIntercept' in dicom_dataset:
        rescale_intercept = float(dicom_dataset.RescaleIntercept)
    else:
        rescale_intercept = 0
    ############################
    win_max = win_center + 0.5 * win_width - 0.5
    win_min = win_max - win_width - 0.5
    range = max(win_max - win_min, 1)
    factor = (d_max - d_min) / range
    img = np.array(dicom_dataset.pixel_array)
    dtype = img.dtype
    if photo_interpretation == 'MONOCHROME1':
        img = (1 << bpp) - (img * rescale_slope + rescale_intercept)
        img = img.astype(dtype)
    else:
        img = (img * rescale_slope + rescale_intercept).astype(dtype)
    dest = np.zeros_like(img).astype(np.uint8)
    dest[img <= win_min] = d_min
    dest[img > win_max] = d_max
    dest[(win_min < img) & (img <= win_max)] = (
        img[(win_min < img) & (img <= win_max)] - win_min) * factor + d_min
    if spp == 1:
        rgb_img = np.stack([dest, dest, dest], axis=2)
    else:
        rgb_img = dest
    return rgb_img


class DicomBackend(ImageUtilsBackend):
    def __init__(self):
        ImageUtilsBackend.__init__(self)

    def accept(self, path, ext, operator):
        if operator in ['resize', 'save']:
            return "NG"
        else:
            if ext in ['.dcm', '.dicom']:
                return "OK"
            else:
                return "NG"

    def imread(self, path, grayscale=False, size=None, interpolate="bilinear",
               channel_first=False, as_uint16=False, num_channels=-1, return_palette_indices=False):
        """
        Read image by DICOM module.
        Notice that PIL only supports uint8 for RGB (not uint16).
        So this imread function returns only uint8 array for both RGB and gray-scale.
        (Currently ignore "I" mode for gray-scale (32bit integer).)

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
                must be one of ["nearest", "box", "bilinear", "hamming", "bicubic", "lanczos"].
            as_uint16 (bool):
                If you specify this argument, you can use only False for pil backend.
            num_channels (int):
                channel size of output array.
                Default is -1 which preserves raw image shape.
            return_palette_indices (bool):
                Whether to return a raw palette indices without any conversion or not.
                If this flag is True and read Image has the mode "P",
                then this function returns 2-D array containing the indices into palette.
                We recommend that this flag should be False unless you intend to use the raw palette indices.

        Returns:
            numpy.ndarray
        """
        _imread_before(grayscale, num_channels)
        dicom_dataset = pydicom.dcmread(path)
        img = _apply_gamma_correction(dicom_dataset)
        return _imread_after(img, size, interpolate, channel_first, self.imresize)
