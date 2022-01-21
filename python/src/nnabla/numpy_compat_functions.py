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

import numpy as np


def dot(a, b, out=None):
    '''
    A compatible operation with ``numpy.dot``.

    Note:
        Any operation between nnabla's Variable/NdArray and numpy array is not supported.

        If both arguments are 1-D, it is inner product of vectors.
        If both arguments are 2-D, it is matrix multiplication.
        If either a or b is 0-D(scalar), it is equivalent to multiply.
        If b is a 1-D array, it is a sum product over the last axis of a and b.
        If b is an M-D array (M>=2), it is a sum product over the last axis of a and the second-to-last axis of b.

    Args:
        a (Variable, NdArray or scalar): Left input array.
        b (Variable, NdArray or scalar): Right input array.
        out: Output argument. This must have the same shape, dtype, and type as the result that would be returned for F.dot(a,b).

    Returns:
        ~nnabla.Variable or ~nnabla.NdArray

    Examples:

    .. code-block:: python

        import numpy as np
        import nnabla as nn
        import nnabla.functions as F

        # 2-D matrix * 2-D matrix
        arr1 = np.arange(5*6).reshape(5, 6)
        arr2 = np.arange(6*8).reshape(6, 8)
        nd1 = nn.NdArray.from_numpy_array(arr1)
        nd2 = nn.NdArray.from_numpy_array(arr2)
        ans1 = F.dot(nd1, nd2)
        print(ans1.shape)
        #(5, 8)

        var1 = nn.Variable.from_numpy_array(arr1)
        var2 = nn.Variable.from_numpy_array(arr2)
        ans2 = F.dot(var1, var2)
        ans2.forward()
        print(ans2.shape)
        #(5, 8)

        out1 = nn.NdArray((5, 8))
        out1.cast(np.float32)
        F.dot(nd1, nd2, out1)
        print(out1.shape)
        #(5, 8)

        out2 = nn.Variable((5, 8))
        out2.data.cast(np.float32)
        F.dot(var1, var2, out2)
        out2.forward()
        print(out2.shape)
        #(5, 8)

        # N-D matrix * M-D matrix (M>=2)
        arr1 = np.arange(5*6*7*8).reshape(5, 6, 7, 8)
        arr2 = np.arange(2*3*8*6).reshape(2, 3, 8, 6)
        nd1 = nn.NdArray.from_numpy_array(arr1)
        nd2 = nn.NdArray.from_numpy_array(arr2)
        ans1 = F.dot(nd1, nd2)
        print(ans1.shape)
        #(5, 6, 7, 2, 3, 6)

        var1 = nn.Variable.from_numpy_array(arr1)
        var2 = nn.Variable.from_numpy_array(arr2)
        ans2 = F.dot(var1, var2)
        ans2.forward()
        print(ans2.shape)
        #(5, 6, 7, 2, 3, 6)

        out1 = nn.NdArray((5, 6, 7, 2, 3, 6))
        out1.cast(np.float32)
        F.dot(nd1, nd2, out1)
        print(out1.shape)
        #(5, 6, 7, 2, 3, 6)

        out2 = nn.Variable((5, 6, 7, 2, 3, 6))
        out2.data.cast(np.float32)
        F.dot(var1, var2, out2)
        out2.forward()
        print(out2.shape)
        #(5, 6, 7, 2, 3, 6)
    '''
    import nnabla as nn
    import nnabla.functions as F

    def _chk(x, mark=0):
        if isinstance(x, nn.NdArray):
            return x.data, 1
        elif isinstance(x, nn.Variable):
            return x.d, 1
        else:
            return x, mark

    m, mark1 = _chk(a)
    n, mark2 = _chk(b)

    if mark1 and mark2:
        if a.ndim == 1 and b.ndim == 1:
            result = F.sum(a * b)
        elif a.ndim == 2 and b.ndim == 2:
            result = F.affine(a, b)
        elif a.ndim == 0 or b.ndim == 0:
            if a.ndim == 0:
                result = F.mul_scalar(b, m)
                if isinstance(a, nn.NdArray) and isinstance(b, nn.Variable):
                    result.forward()
                    result = result.data
            else:
                result = F.mul_scalar(a, n)
                if isinstance(a, nn.Variable) and isinstance(b, nn.NdArray):
                    result.forward()
                    result = result.data
        elif b.ndim == 1:
            h = F.affine(a, F.reshape(b, (-1, 1)), base_axis=a.ndim - 1)
            result = F.reshape(h, h.shape[:-1])
        elif b.ndim >= 2:
            index = [*range(0, b.ndim)]
            index.insert(0, index.pop(b.ndim - 2))
            b = F.transpose(b, index)
            h = F.affine(a, b, base_axis=a.ndim - 1)
            result = h
    else:
        result = np.dot(a, b)

    if out is not None:
        out_, _ = _chk(out)
        result_, _ = _chk(result)
        if type(out) == type(result) and out_.shape == result_.shape and out_.dtype == result_.dtype:
            if isinstance(out, nn.NdArray):
                out.cast(result.data.dtype)[...] = result.data
            elif isinstance(out, nn.Variable):
                out.rewire_on(result)
            else:
                out = result
        else:
            raise ValueError(f"Output argument must have the same shape, type and dtype as the result that would be "
                             f"returned for F.dot(a,b).")
    else:
        return result
