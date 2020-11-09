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

import nnabla.functions as F
import nnabla.parametric_functions as PF


############################################################
# nnabla callback helpers. (used in apply_w, apply_b and so on)
############################################################


def spectral_norm_callback(dim=0):
    """
    :args
        dim: axis along which normalization is taken. default is 0, for convolution.
    :returns
        callback function.
    """

    def callback(x):
        return PF.spectral_norm(x, dim=dim)

    return callback


def weitgh_standardization_callback(dim=0):
    """
    :args
        dim: axis along which normalization is taken. default is 0, for convolution.
    :returns
        callback function.
    """

    def callback(x):
        return F.weight_standardization(x, channel_axis=dim)

    return callback
