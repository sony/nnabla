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

import numpy as np


def rescale_pixel_intensity(img, input_low=0, input_high=255, output_low=0, output_high=255, output_type=None):
    if not isinstance(img, np.ndarray):
        raise ValueError(
            "rescale_pixel_intensity() supports only numpy.ndarray as input")

    if output_type is None:
        output_type = img.dtype

    if input_low == output_low and output_low == output_high:
        return img.copy().astype(output_type)

    else:
        # [input_low, input_high] -> [0, input_high - input_low] -> [0, 1]
        normalized = (img - input_low) / (input_high - input_low)

        # [0, 1] -> [0, output_high - output_low] -> [output_low, output_high]
        scaled = normalized * (output_high - output_low) + output_low

        return scaled.astype(output_type)
