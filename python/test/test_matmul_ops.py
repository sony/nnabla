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
import pytest
import numpy as np

import nnabla as nn
from nnabla.testing import assert_allclose


@pytest.mark.parametrize("seed", [313, 314])
@pytest.mark.parametrize("shape", [
                                   ((100,), (100,)),  # vector @ vector
                                   ((2, 1), (1, 2)),  # 2-D matrix @ 2-D matrix
                                   ((100,), (100, 50)),  # vector @ 2-D matrix
                                   ((50, 100), (100,)),  # 2-D matrix @ vector
                                   # M-D matrix @ N-D matrix (M,N>2, M=N)
                                   ((50, 50, 100), (1, 100, 4)),
                                   # M-D matrix @ N-D matrix (M>2 and N=1)
                                   ((50, 50, 100), (100,)),
                                   # M-D matrix @ N-D matrix (M,N>2 and M>N)
                                   ((5, 50, 50, 100), (100, 5)),
                                   # M-D matrix @ N-D matrix (M=1 and N>2)
                                   ((100,), (5, 100, 5)),
                                   # M-D matrix @ N-D matrix (M,N>2 and M<N)
                                   ((4, 1, 50, 100), (3, 4, 5, 100, 5))
                                    ])
@pytest.mark.parametrize("is_dynamic", [True, False])
def test_ndarray_arithmetic_matmul_ops(seed, shape, is_dynamic):
    rng = np.random.RandomState(seed)
    a1 = rng.randn(*shape[0]).astype(np.float32)
    a2 = rng.randn(*shape[1]).astype(np.float32)
    n1 = nn.NdArray.from_numpy_array(a1)
    n2 = nn.NdArray.from_numpy_array(a2)
    v1 = nn.Variable.from_numpy_array(a1)
    v2 = nn.Variable.from_numpy_array(a2)

    ref = a1 @ a2

    # NdArray @ NdArray
    ans1 = n1 @ n2
    assert_allclose(ref, ans1.data, atol=1e-5)

    # NdArray @ Variable
    ans2 = n1 @ v2
    assert_allclose(ref, ans2.data, atol=1e-5)

    # Variable @ NdArray
    ans3 = v1 @ n2
    assert_allclose(ref, ans3.data, atol=1e-5)

    # Variable @ Variable
    with nn.auto_forward(is_dynamic):
        ans4 = v1 @ v2
        if not is_dynamic:
            ans4.forward()
        assert_allclose(ref, ans4.d, atol=1e-5)


@pytest.mark.parametrize("seed", [313, 314])
@pytest.mark.parametrize("shape", [
                                   ((), ()),  # scalar @ scalar
                                   ((100,), ()),  # vector @ scalar
                                   ((), (100,))  # scalar @ vector
                                   ])
def test_wrong_case_ndarray_arithmetic_matmul_ops(seed, shape):
    rng = np.random.RandomState(seed)

    if not shape[0]:
        a1 = rng.randn()
        n1 = nn.NdArray()
        n1.cast(np.float32)[...] = a1
        v1 = nn.Variable()
        v1.data.cast(np.float32)[...] = a1
    else:
        a1 = rng.randn(*shape[0]).astype(np.float32)
        n1 = nn.NdArray.from_numpy_array(a1)
        v1 = nn.Variable.from_numpy_array(a1)

    if not shape[1]:
        a2 = rng.randn()
        n2 = nn.NdArray()
        n2.cast(np.float32)[...] = a2
        v2 = nn.Variable()
        v2.data.cast(np.float32)[...] = a2
    else:
        a2 = rng.randn(*shape[1]).astype(np.float32)
        n2 = nn.NdArray.from_numpy_array(a2)
        v2 = nn.Variable.from_numpy_array(a2)

    with pytest.raises(AssertionError) as excinfo:
        # NdArray @ NdArray
        ans1 = n1 @ n2

        # NdArray @ Variable
        ans2 = n1 @ v2

        # Variable @ NdArray
        ans3 = v1 @ n2

        # Variable @ Variable
        ans4 = v1 @ v2

        # numpy.ndarray or float @ NdArray
        ans5 = a1 @ n1

        # NdArray @ numpy.ndarray or float
        ans6 = n1 @ a2

        # numpy.ndarray or float @ Variable
        ans7 = a1 @ v2

        # Variable @ numpy.ndarray or float
        ans8 = v1 @ a2
