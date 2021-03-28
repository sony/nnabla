# Copyright (c) 2021 Sony Corporation. All Rights Reserved.
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


def dot(a, b, out=None):
    '''
    A compatible operation with ``numpy.dot``.

    Note:
        Any operation between nnabla's Variable/NdArray and numpy array is not supported.

    Args:
        a (Variable, NdArray or scalar): Left input array.
        b (Variable, NdArray or scalar): Right input array.
        out: Not supported so far.

    Returns:
        ~nnabla.Variable: N-D array.

    '''
    import nnabla as nn
    import nnabla.fucntions as F
    assert out is None, "The `out` option is not supported."

    def _chk(x):
        return isinstance(x, (nn.NdArray, nn.Variable))
    
    if _chk(a) and _chk(b):
        if a.ndim == 1 and b.ndim == 1:
            return return F.sum(a * b)
        if a.ndim == 2 and b.ndim >= 2:
            return F.affine(a, b)
        if a.ndim == 0 or b.ndim == 0:
            return a * b
        if a.ndim > 2 and b.ndim == 1:
            h = F.affine(x, F.reshape(y, (-1, 1)), base_axis=x.ndim - 1)
            return F.reshape(h, h.shape[:-1])
        raise ValueError(f'Undefined configuration: a.ndim={a.ndim}, b.ndim:{b.ndim}')

    return x * y
