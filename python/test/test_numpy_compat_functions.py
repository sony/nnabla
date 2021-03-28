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

import pytest
import numpy as np
import nnabla as nn
import nnabla.functions as F
from nnabla.testing import assert_allclose
from nbla_test_function_network import function_network_tester


@pytest.mark.parametrize("seed", [313, 314])
@pytest.mark.parametrize("scalar", [3.444, 5])
@pytest.mark.parametrize("is_dynamic", [True, False])
def test_scalar_dot(seed, scalar, is_dynamic):
    rng = np.random.RandomState(seed)
    a1 = scalar
    a2 = rng.randn(3, 4, 5, 6).astype(np.float32)
    n = nn.NdArray.from_numpy_array(a2)
    v = nn.Variable.from_numpy_array(a2)

    ref = F.dot(a1, a2)

    ans1 = F.dot(a1, n)
    assert_allclose(ans1.data, ref)

    out1 = nn.NdArray((3, 4, 5, 6))
    F.dot(a1, n, out1)
    assert_allclose(out1.data, ref)

    with nn.auto_forward(is_dynamic):
        ans2 = F.dot(a1, v)
        if not is_dynamic:
            ans2.forward()
        assert_allclose(ans2.d, ref)

        out2 = nn.Variable((3, 4, 5, 6))
        F.dot(a1, v, out2)
        if not is_dynamic:
            out2.forward()
        assert_allclose(out2.d, ref)


@pytest.mark.parametrize("seed", [313, 314])
@pytest.mark.parametrize("shape", [
                                   ((1,), (1,)), ((100,), (100,)),  # both vector
                                   ((1, 1), (1, 1)), ((50, 100),
                                                      (100, 1)),  # both 2-D array
                                   ((), ()), ((), (3, 4, 5)
                                              ), ((3, 4, 5), ()),  # 0-D
                                   ((2, 3, 1), (1,)), ((2, 3, 100), (100,)), ((
                                       100, 100), (100,)),  # b is 1-D_array
                                   # M-D array (where M>=2)
                                   # ((5, 3, 4, 5, 5), (100, 5, 3)) skipped
                                   ])
@pytest.mark.parametrize("is_dynamic", [True, False])
def test_ndarray_dot(seed, shape, is_dynamic):
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

    ref = F.dot(a1, a2)

    ans1_1 = F.dot(n1, n2)
    ans1_2 = F.dot(n1, v2)
    ans1_3 = F.dot(v1, n2)
    assert_allclose(ans1_1.data, ref, atol=1e-3)
    assert_allclose(ans1_2.data, ref, atol=1e-3)
    assert_allclose(ans1_3.data, ref, atol=1e-3)
    with nn.auto_forward(is_dynamic):
        ans1_4 = F.dot(v1, v2)
        if is_dynamic:
            ans1_4.forward()
            assert_allclose(ans1_4.d, ref, atol=1e-3)

    out = ref.copy()
    F.dot(a1, a2, out)
    assert_allclose(out, ref, atol=1e-3)

    out1_1 = nn.NdArray(ans1_1.shape)
    out1_1.cast(np.float32)
    F.dot(n1, n2, out1_1)
    assert_allclose(out1_1.data, ref, atol=1e-3)

    out1_2 = nn.NdArray(ans1_2.shape)
    out1_2.cast(np.float32)
    F.dot(n1, v2, out1_2)
    assert_allclose(out1_2.data, ref, atol=1e-3)

    out1_3 = nn.NdArray(ans1_3.shape)
    out1_3.cast(np.float32)
    F.dot(v1, n2, out1_3)
    assert_allclose(out1_3.data, ref, atol=1e-3)

    out1_4 = nn.Variable(ans1_4.shape)
    out1_4.data.cast(np.float32)
    with nn.auto_forward(is_dynamic):
        F.dot(v1, v2, out1_4)
        if not is_dynamic:
            out1_4.forward()
        assert_allclose(out1_4.d, ref, atol=1e-3)

    # Ndarray with a wrong dtype
    out2_1 = nn.NdArray(ref.shape)
    out2_1.cast(int)
    out2_2 = nn.Variable(ref.shape)
    out2_2.data.cast(int)
    # should not exec
    with pytest.raises(ValueError) as excinfo:
        F.dot(n1, n2, out2_1)
        F.dot(n1, v2, out2_1)
        F.dot(v1, n2, out2_1)
        F.dot(v1, v2, out2_2)


@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("shape_a, shape_b", [
                                    ((1,), (1,)), ((100,), (100,)),  # both vector
                                    # both 2-D array
                                    ((1, 1), (1, 1)), ((3, 2), (2, 3)),
                                    # b is 1-D_array
                                    ((2, 3, 1), (1,)), ((1,), (2, 1, 2)),
                                    # M-D array (where M>=2)
                                    ((2, 3, 2), (2, 2, 2)),
                                   ])
def test_backward_dot_muti_array(seed, shape_a, shape_b):
    rng = np.random.RandomState(seed)

    a = rng.randn(*shape_a).astype(np.float32)
    b = rng.randn(*shape_b).astype(np.float32)

    inputs = [
        a,
        b,
    ]

    function_network_tester(rng, F.dot, inputs)


@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("shape_a, shape_b", [
                                    ((1,), (1,)), ((100,), (100,)),  # both vector
                                    # both 2-D array
                                    ((1, 1), (1, 1)), ((3, 2), (2, 3)),
                                    # b is 1-D_array
                                    ((2, 3, 1), (1,)), ((1,), (2, 1, 2)),
                                    # M-D array (where M>=2)
                                    ((2, 3, 2), (2, 2, 2)),
                                   ])
def test_backward_dot_muti_array_out(seed, shape_a, shape_b):
    rng = np.random.RandomState(seed)

    a = rng.randn(*shape_a).astype(np.float32)
    b = rng.randn(*shape_b).astype(np.float32)

    inputs = [
        a,
        b,
    ]

    v_a = nn.Variable.from_numpy_array(a)
    v_b = nn.Variable.from_numpy_array(b)

    ans = F.dot(v_a, v_b)
    out = nn.Variable(ans.shape)
    out.data.cast(np.float32)

    function_network_tester(rng, F.dot, inputs, func_args=[out], args_out=True)


@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("shape_a, shape_b", [
    ((), ()), ((), (2, 2, 2)), ((2, 2, 2), ())  # 0-D
])
def test_backward_dot_have_scalar(seed, shape_a, shape_b):
    rng = np.random.RandomState(seed)

    inputs = []
    func_args = []

    if not shape_a:
        a = rng.randn()
        func_args += [a]
    else:
        a = rng.randn(*shape_a).astype(np.float32)
        inputs += [a]

    if not shape_b:
        b = rng.randn()
        func_args += [b]
    else:
        b = rng.randn(*shape_b).astype(np.float32)
        inputs += [b]
    function_network_tester(
        rng, F.dot, inputs, func_args=func_args, have_scalar=True)


@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("shape_a, shape_b", [
    ((), ()), ((), (2, 2, 2)), ((2, 2, 2), ())  # 0-D
])
def test_backward_dot_have_scalar_out(seed, shape_a, shape_b):
    rng = np.random.RandomState(seed)

    inputs = []
    func_args = []

    if not shape_a:
        a = rng.randn()
        func_args += [a]
        v_a = nn.Variable()
        v_a.data.cast(np.float32)[...] = a
    else:
        a = rng.randn(*shape_a).astype(np.float32)
        inputs += [a]
        v_a = nn.Variable.from_numpy_array(a)

    if not shape_b:
        b = rng.randn()
        func_args += [b]
        v_b = nn.Variable()
        v_b.data.cast(np.float32)[...] = b
    else:
        b = rng.randn(*shape_b).astype(np.float32)
        inputs += [b]
        v_b = nn.Variable.from_numpy_array(b)

    ans = F.dot(v_a, v_b)
    out = nn.Variable(ans.shape)
    out.data.cast(np.float32)
    func_args += [out]

    function_network_tester(
        rng, F.dot, inputs, func_args=func_args, have_scalar=True, args_out=True)
