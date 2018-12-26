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

from __future__ import absolute_import

import os
import pytest

import numpy as np
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF

from nnabla.parameter import get_parameter_or_create


def create_scale_bias(idx, maps, ndim=4):
    shape = [1] * ndim
    shape[1] = maps
    a = get_parameter_or_create("a{}".format(idx), list(shape),
                                None, True, True)
    b = get_parameter_or_create("b{}".format(idx), list(shape),
                                None, True, True)
    return a, b
