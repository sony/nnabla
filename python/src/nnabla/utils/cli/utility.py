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

import os
import numpy as np

from nnabla.logger import logger


def is_float(x):
    # x is string
    try:
        float(x)
        return True
    except ValueError:
        return False


def compute_full_path(root_path, file_path):
    full_path = os.path.join(root_path, file_path)
    full_path = full_path.replace('\\', os.path.sep)
    full_path = full_path.replace('/', os.path.sep)
    full_path = full_path.replace(os.path.sep + '.' + os.path.sep, os.path.sep)
    return full_path


def let_data_to_variable(variable, data, ctx=None, data_name=None, variable_name=None):
    try:
        if data.dtype <= np.float64:
            variable.data.cast(data.dtype)[...] = data
        else:
            variable.d = data
    except:
        if variable.shape != data.shape:
            logger.critical('Shape does not match between data{} and variable{} ({} != {}).'.format(
                ' "' + data_name + '"' if data_name else '',
                ' "' + variable_name + '"' if variable_name else '',
                data.shape, variable.shape))
        raise
    variable.need_grad = False

    # Copy to device
    if ctx:
        try:
            variable.data.cast(variable.data.dtype, ctx)
        except:
            if ctx.array_class != 'CpuArray':
                # Fallback to cpu
                ctx.array_class = 'CpuArray'
                variable.data.cast(variable.data.dtype, ctx)
            else:
                raise
