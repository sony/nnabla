# Copyright 2020,2021 Sony Corporation.
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

import numpy as np


np_dtpye_to_int = dict()
np_dtpye_to_int[bool] = 0
np_dtpye_to_int[np.byte] = 1
np_dtpye_to_int[np.ubyte] = 2
np_dtpye_to_int[np.short] = 3
np_dtpye_to_int[np.ushort] = 4
np_dtpye_to_int[np.int32] = 5
np_dtpye_to_int[np.uint32] = 6
# np_dtpye_to_int[np.long] = 7
# np_dtpye_to_int[np.ulong] = 8
np_dtpye_to_int[np.longlong] = 9
np_dtpye_to_int[np.int64] = 9
np_dtpye_to_int[np.ulonglong] = 10
np_dtpye_to_int[np.uint64] = 10
np_dtpye_to_int[np.float32] = 11
np_dtpye_to_int[np.double] = 12
np_dtpye_to_int[np.longdouble] = 13


int_to_np_dtype = {v: k for k, v in np_dtpye_to_int.items()}

# TODO:
# Currently np.dtype is only supported, but we should support nn.dtype somehow,
# since np.dtype does not support lower bit types like 1, 2, 4 bits.
