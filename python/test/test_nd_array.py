# Copyright 2017,2018,2019,2020,2021 Sony Corporation.
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

import nnabla as nn

import pytest
from nnabla.ext_utils import get_extension_context
import nnabla.functions as F


def test_nd_array():
    shape = [2, 3, 4]
    a = nn.NdArray(shape)
    npa = np.arange(a.size).reshape(a.shape).astype(np.int32)
    a.data = npa
    b = nn.NdArray.from_numpy_array(npa)
    b.dtype == np.int32
    assert np.all(a.data == npa)
    assert np.all(a.data == b.data)
    assert a.shape == npa.shape
    assert b.size == np.prod(shape)
    a.cast(np.int32)
    assert a.data.dtype == np.int32
    b.zero()
    assert np.all(b.data == 0)
    a.fill(3)
    assert np.all(a.data == 3)
    b.copy_from(a)
    assert np.all(a.data == b.data)


def test_copy_from():
    shape = [2, 3, 4]
    src = nn.NdArray(shape)
    dst = nn.NdArray(shape)
    src.data = 0
    src.cast(dtype=np.uint8)
    dst.copy_from(src, use_current_context=False)
    assert dst.dtype == np.uint8

    from nnabla.ext_utils import get_extension_context
    with nn.context_scope(get_extension_context('cpu', dtype='float')):
        dst.copy_from(src, use_current_context=True)
    assert dst.dtype == np.float32


@pytest.mark.parametrize("value", [
    1,
    1.3,
    np.array(np.zeros((2, 3))),
    np.arange(6).reshape(2, 3)])
def test_nd_array_data(value):
    shape = (2, 3)

    # Use default dtype (float32) in getter
    a = nn.NdArray(shape)
    with pytest.raises(Exception):
        _ = a.dtype
    _ = a.data
    assert a.dtype == np.float32

    # Use value dtype in setter
    a = nn.NdArray(shape)
    a.data = value
    if not np.isscalar(value) or \
       (np.dtype(type(value)).kind != 'f' and value > (1 << 53)):
        assert a.dtype == np.asarray(value).dtype
        assert a.data.dtype == np.asarray(value).dtype
    else:
        assert a.data.dtype == np.float32


def test_clear_called():
    a = nn.NdArray(1)
    assert a.clear_called == False
    a.fill(3)
    assert a.clear_called == False
    a.clear()
    assert a.clear_called == True

    a.fill(3)
    assert a.clear_called == False
    a.clear()
    assert a.clear_called == True
    a.zero()
    assert a.clear_called == False
    a.clear()
    assert a.clear_called == True

    a.data[0] = -1
    assert a.clear_called == False


def test_clear_called_narrow():
    a = nn.NdArray(2, 2)
    assert a.clear_called == False
    a.fill(3)
    assert a.clear_called == False

    b = a.narrow(0, 0, 1)
    assert b.clear_called == False

    with pytest.raises(RuntimeError):
        b.clear()


def test_nd_array_initialize():
    shape = [2, 3, 4]
    a = nn.NdArray(shape)  # uninitialized, but is filled by zeros

    ref_a = np.zeros(shape, dtype=np.float32)
    assert (a.data == ref_a).all()


class TestNdArrayNarrow():

    def setup_method(self):
        self._ctx = nn.get_current_context()

    def teardown_method(self):
        nn.set_default_context(self._ctx)

    def try_to_set_ctx(self, ctx_name):
        try:
            ctx = get_extension_context(ctx_name)
        except Exception as e:
            pytest.skip("Import of nnabla-ext-cuda failed, so skip this test.")

        nn.set_default_context(ctx)

    @pytest.mark.parametrize("shape", [
        [10, 5], [100, 100], [1000, 1000], [20], [10, 10, 10]])
    @pytest.mark.parametrize("ctx_name", [
        "cpu", "cuda", "cudnn"])
    def test_nd_array_narrow(self, shape, ctx_name):
        self.try_to_set_ctx(ctx_name)

        x = shape[0]
        s = x // 10
        a = nn.NdArray(shape)
        a.fill(3.0)
        b = a.narrow(0, 0, 4 * s)
        b.fill(5.0)
        c = a.narrow(0, 6 * s, 4 * s)
        c.fill(7.0)
        d = c.narrow(0, 2 * s, 2 * s)
        d.fill(9.0)
        b += c

        ref_a = np.zeros(shape, dtype=np.float32)
        ref_a.fill(3.0)
        ref_b = ref_a[0:4*s]
        ref_b.fill(5.0)
        ref_c = ref_a[6*s:10*s]
        ref_c.fill(7.0)
        ref_d = ref_c[2*s:4*s]
        ref_d.fill(9.0)
        ref_b += ref_c

        assert a.shape == tuple(shape)

        assert (a.data == ref_a).all()
        assert (b.data == ref_b).all()
        assert (c.data == ref_c).all()
        assert (d.data == ref_d).all()

        assert a.data.dtype == ref_a.dtype
        assert b.data.dtype == ref_b.dtype
        assert c.data.dtype == ref_c.dtype
        assert d.data.dtype == ref_d.dtype

    @pytest.mark.parametrize("shape", [
        [10, 5], [100, 100], [1000, 1000], [20], [10, 10, 10]])
    @pytest.mark.parametrize("ctx_name", [
        "cpu", "cuda", "cudnn"])
    def test_nd_array_narrow_2(self, shape, ctx_name):
        self.try_to_set_ctx(ctx_name)

        x = shape[0]
        s = x // 10
        a = nn.NdArray(shape)  # uninitialized
        b = a.narrow(0, 0, 5 * s)
        b.fill(5.0)
        c = a.narrow(0, 5 * s, 5 * s)
        c.fill(7.0)
        d = c.narrow(0, 2 * s, 2 * s)
        d.fill(9.0)
        b += c

        ref_a = np.zeros(shape, dtype=np.float32)
        ref_b = ref_a[0:5*s]
        ref_b.fill(5.0)
        ref_c = ref_a[5*s:10*s]
        ref_c.fill(7.0)
        ref_d = ref_c[2*s:4*s]
        ref_d.fill(9.0)
        ref_b += ref_c

        assert a.shape == ref_a.shape
        assert b.shape == ref_b.shape
        assert c.shape == ref_c.shape
        assert d.shape == ref_d.shape

        assert a.size == ref_a.size
        assert b.size == ref_b.size
        assert c.size == ref_c.size
        assert d.size == ref_d.size

        assert (a.data == ref_a).all()
        assert (b.data == ref_b).all()
        assert (c.data == ref_c).all()
        assert (d.data == ref_d).all()

        assert a.data.dtype == ref_a.dtype
        assert b.data.dtype == ref_b.dtype
        assert c.data.dtype == ref_c.dtype
        assert d.data.dtype == ref_d.dtype

        a = nn.NdArray.from_numpy_array(np.zeros(shape, dtype=np.float32))

        # If created with from_numpy_array, head array's context of NdArray becomes cpu.
        # Therefore, use identity to change context into current_context(cpu or cuda or cudnn).
        a = F.identity(a)
        b = a.narrow(0, 0, 5 * s)
        b.fill(5.0)
        c = a.narrow(0, 5 * s, 5 * s)
        c.fill(7.0)
        d = c.narrow(0, 2 * s, 2 * s)
        d.fill(9.0)

        b += c
        assert (a.data == ref_a).all()
        assert (b.data == ref_b).all()
        assert (c.data == ref_c).all()
        assert (d.data == ref_d).all()

        a = nn.NdArray.from_numpy_array(np.zeros(shape, dtype=np.float64))
        b = a.narrow(0, 0, 5 * s)
        b.fill(5.0)
        c = a.narrow(0, 5 * s, 5 * s)
        b.fill(7.0)
        with pytest.raises(RuntimeError):
            # Failed `root_key == created_key || is_root()`: cast of child-arrays is not permitted
            # (0:cpu:DOUBLE cannot convert to 0:cpu:FLOAT)
            b += c

    @pytest.mark.parametrize("ctx_name", [
        "cpu", "cuda", "cudnn"])
    def test_nd_array_narrow_fill(self, ctx_name):
        self.try_to_set_ctx(ctx_name)

        # child fill first
        ref_0 = np.array([0, 0, 5, 5, 3, 3, 5, 5, 7, 7])
        # grandchild fill first
        ref_1 = np.array([0, 0, 5, 5, 5, 5, 5, 5, 7, 7])

        x = nn.NdArray(10)
        x.fill(0)
        x = F.identity(x)

        # parent-child relation
        # x -- a -- b
        #  \-- c
        a = x.narrow(0, 2, 6)
        b = a.narrow(0, 2, 2)
        c = x.narrow(0, 8, 2)

        a.fill(5)
        b.fill(3)
        c.fill(7)
        assert (x.get_data(mode="r") == ref_0).all()

        x.fill(0)
        b.fill(3)
        a.fill(5)
        c.fill(7)
        assert (x.get_data(mode="r") == ref_1).all()

    @pytest.mark.parametrize("ctx_name", [
        "cpu", "cuda", "cudnn"])
    def test_nd_array_narrow_get_cast(self, ctx_name):
        self.try_to_set_ctx(ctx_name)

        a = nn.NdArray(2, 2)
        a.fill(3)
        a = F.identity(a)

        b = a.narrow(0, 0, 1)
        c = b.narrow(0, 0, 1)

        assert (a.get_data(mode="r") == np.full([2, 2], 3)).all()
        assert (b.get_data(mode="r") == np.full([1, 2], 3)).all()
        assert (c.get_data(mode="r") == np.full([1, 2], 3)).all()

        # get_data with type conversion is allowed only when read-only-mode.
        b.get_data(mode='r', dtype=np.float16)
        c.get_data(mode='r', dtype=np.float16)

        with pytest.raises(RuntimeError):
            b.get_data(mode='rw', dtype=np.float16)

        with pytest.raises(RuntimeError):
            c.get_data(mode='rw', dtype=np.float16)

        # Check the change is reflected in parent array after cast.
        # Only the same dtype of cast for narrow array is allowed.
        b.cast(dtype=np.float32, ctx=nn.get_current_context())
        b += 1
        assert (a.get_data(mode="r") == np.array([[4, 4], [3, 3]])).all()

        c.cast(dtype=np.float32, ctx=nn.get_current_context())
        c += 1
        assert (a.get_data(mode="r") == np.array([[5, 5], [3, 3]])).all()

        with pytest.raises(RuntimeError):
            b.cast(dtype=np.float16)

        with pytest.raises(RuntimeError):
            c.cast(dtype=np.float16)

        assert a.dtype == np.float32
        assert b.dtype == np.float32
        assert c.dtype == np.float32

        # cast for parent array is allowed.
        a.cast(dtype=np.float16)
        assert a.dtype == np.float16

        # If parent array is cast, dtype in narrow array also changes.
        assert b.dtype == np.float16
        assert c.dtype == np.float16

        # Value of `a` remains the same and the parent-child relationship is maintained.
        assert (a.data == np.array([[5, 5], [3, 3]])).all()
        b.fill(0)
        assert (a.data == np.array([[0, 0], [3, 3]])).all()
        assert (c.data == np.array([[0, 0]])).all()

    @pytest.mark.parametrize("ctx_name", [
        "cpu", "cuda", "cudnn"])
    def test_nd_array_narrow_get_cache(self, ctx_name):
        self.try_to_set_ctx(ctx_name)

        a = nn.NdArray(3)
        a = F.identity(a)
        a.fill(1)

        b = a.narrow(0, 0, 2)
        c = b.narrow(0, 0, 1)

        b.fill(2)
        c.fill(3)

        # Make head array's context in `a` become current context
        a.cast(dtype=np.float32, ctx=nn.get_current_context())

        # Create a cache of different type data
        a.get_data(mode="r", dtype=np.float16)
        b.get_data(mode="r", dtype=np.float16)
        c.get_data(mode="r", dtype=np.float16)

        assert (a.get_data(mode="r", dtype=np.float16)
                == np.array([3, 2, 1])).all()

        c += 1

        # If the cache remains, the following test will fail.
        assert (a.get_data(mode="r", dtype=np.float16)
                == np.array([4, 2, 1])).all()
        assert (b.get_data(mode="r", dtype=np.float16)
                == np.array([4, 2])).all()
        assert (c.get_data(mode="r", dtype=np.float16) == np.array([4])).all()

        c.fill(5)

        assert (a.get_data(mode="r", dtype=np.float16)
                == np.array([5, 2, 1])).all()
        assert (b.get_data(mode="r", dtype=np.float16)
                == np.array([5, 2])).all()
        assert (c.get_data(mode="r", dtype=np.float16) == np.array([5])).all()

        c.zero()

        assert (a.get_data(mode="r", dtype=np.float16)
                == np.array([0, 2, 1])).all()
        assert (b.get_data(mode="r", dtype=np.float16)
                == np.array([0, 2])).all()
        assert (c.get_data(mode="r", dtype=np.float16) == np.array([0])).all()

    @pytest.mark.parametrize("ctx_name", [
        "cpu", "cuda", "cudnn"])
    def test_nd_array_narrow_copy(self, ctx_name):
        self.try_to_set_ctx(ctx_name)

        a = nn.NdArray(2, 2)
        a.fill(3)
        a = F.identity(a)
        assert (a.get_data(mode="r") == np.full([2, 2], 3)).all()

        b = a.narrow(0, 0, 1)

        c = nn.NdArray(b.shape)
        c.fill(5)

        # Check copy_from array to narrow array
        b.copy_from(c)
        assert (b.get_data(mode="r") == c.data).all()
        assert (b.get_data(mode="r") == np.full([1, 2], 5)).all()
        assert (a.data[0:1] == b.data).all()

        # Check copy_from narrow array to array
        a = nn.NdArray(2, 2)
        a.fill(8)
        a = F.identity(a)
        b = a.narrow(0, 0, 1)
        c.copy_from(b)
        assert (b.get_data(mode="r") == c.data).all()
        assert (b.get_data(mode="r") == np.full([1, 2], 8)).all()

        a = nn.NdArray(2, 2)
        a.fill(3)
        a = F.identity(a)
        assert (a.get_data(mode="r") == np.full([2, 2], 3)).all()

        b = a.narrow(0, 0, 1)

        c = nn.NdArray(b.shape)
        c.fill(5)

        # Check substitution with NdArray.data
        # Tested only when ctx_name is "cpu", since NdArray.data is implicitly use cpu context.
        if ctx_name != "cpu":
            with pytest.raises(RuntimeError):
                b.data = c.data
        else:
            b.data = c.data
            assert (b.get_data(mode="r") == c.data).all()
            assert (b.get_data(mode="r") == np.full([1, 2], 5)).all()
            assert (a.data[0:1] == b.data).all()

    @pytest.mark.parametrize("ctx_name", [
        "cpu", "cuda", "cudnn"])
    def test_nd_array_narrow_write_only_cast(self, ctx_name):
        self.try_to_set_ctx(ctx_name)

        # Using F.identity() case
        # F.identity() uses `write_only==true` cast for the destination array
        a = nn.NdArray((2,))
        a.fill(3)
        b = nn.NdArray((4,))
        b.fill(5)

        c = b.narrow(0, 0, 2)
        F.identity(a, outputs=[c])

        np.testing.assert_equal(c.get_data(mode="r"), np.array([3, 3]))
        np.testing.assert_equal(b.get_data(), np.array([3, 3, 5, 5]))

        # Using F.identity() case with default zeroing of parent array (`b`)
        a = nn.NdArray((2,))
        a.fill(3)
        b = nn.NdArray((4,))

        c = b.narrow(0, 0, 2)
        F.identity(a, outputs=[c])

        np.testing.assert_equal(c.get_data(mode="r"), np.array([3, 3]))
        np.testing.assert_equal(b.get_data(), np.array([3, 3, 0, 0]))

        # Using copy_from() and partial narrowing case
        # NdArray::copy_from() uses F.identity() internally
        # (Buf reported case)
        a = nn.NdArray((2,))
        a.fill(3)
        b = nn.NdArray((4,))
        b.fill(5)

        c = b.narrow(0, 0, 2)
        c.copy_from(a)

        np.testing.assert_equal(c.get_data(mode="r"), np.array([3, 3]))
        np.testing.assert_equal(b.get_data(), np.array([3, 3, 5, 5]))

    @pytest.mark.parametrize("ctx_name", [
        "cpu", "cuda", "cudnn"])
    def test_nd_array_narrow_errors(self, ctx_name):
        self.try_to_set_ctx(ctx_name)

        with pytest.raises(RuntimeError):
            # Failed `length >= 0`: negative number for `length` (-5) is not permitted
            a = nn.NdArray((10, 5))
            _ = a.narrow(0, 2, -5)
        with pytest.raises(RuntimeError):
            # Failed `length >= 0 && start + length <= size`: start (2) + length (50) exceeds dimension size (10)
            a = nn.NdArray((10, 5))
            _ = a.narrow(0, 2, 50)
        with pytest.raises(RuntimeError):
            # Failed `dim == 0`: `dim` is out of range (expected to be 0, but got 1)
            a = nn.NdArray((10, 5))
            _ = a.narrow(1, 2, 5)
        with pytest.raises(RuntimeError):
            # Failed `start0 >= -size && start0 < size`: `start0` is out of range (expected to be [-10, 9], but got -11)
            a = nn.NdArray((10, 5))
            _ = a.narrow(0, -11, 5)

        # Failed if overlapped area.
        a = nn.NdArray(10)
        b = a.narrow(0, 2, 2)
        with pytest.raises(RuntimeError):
            d = a.narrow(0, 0, 3)
        with pytest.raises(RuntimeError):
            d = a.narrow(0, 2, 2)
        with pytest.raises(RuntimeError):
            d = a.narrow(0, 3, 2)

    @pytest.mark.parametrize("ctx_name", [
        "cpu", "cuda", "cudnn"])
    def test_nd_array_narrow_zeroing(self, ctx_name):
        self.try_to_set_ctx(ctx_name)

        a = nn.NdArray(3)
        assert not a.zeroing

        a.zero()
        assert a.zeroing
        a.fill(3)
        assert not a.zeroing

        a.zero()
        assert a.zeroing
        a.cast(np.float16)
        assert not a.zeroing

        a.zero()
        assert a.zeroing
        a.get_data(mode="rw")
        assert not a.zeroing

        a.zero()
        assert a.zeroing
        a.get_data(mode="r")
        assert a.zeroing

        a.zero()
        assert a.zeroing
        a.data
        assert not a.zeroing

        a = nn.NdArray(3)
        a.data  # make array
        b = a.narrow(0, 0, 1)
        assert not b.zeroing

        b.zero()
        assert b.zeroing
        b.fill(3)
        assert not b.zeroing

        b.zero()
        assert b.zeroing
        b.get_data(mode="r")
        assert b.zeroing

        b.zero()
        assert b.zeroing
        b.data
        assert not b.zeroing

    @pytest.mark.parametrize("ctx_name", [
        "cpu"])
    def test_nd_array_narrow_array_class_cpu(self, ctx_name):
        self.try_to_set_ctx(ctx_name)

        ctx = nn.get_current_context()

        ctx.array_class = "CpuArray"
        a = nn.NdArray(2, 5)
        a.cast(np.float32, ctx=ctx)
        # No error
        b = a.narrow(0, 0, 1)

        ctx.array_class = "CpuCachedArray"
        a = nn.NdArray(2, 5)
        a.cast(np.float32, ctx=ctx)
        # No error
        b = a.narrow(0, 0, 1)

        # CpuDlpackArray
        # Dlpack Array is created using nn.utils.dlpack.to_dlpack, but is not tested here because it is not NdArray.

    @pytest.mark.parametrize("ctx_name", [
        "cuda"])
    def test_nd_array_narrow_array_class_cuda(self, ctx_name):
        self.try_to_set_ctx(ctx_name)

        ctx = nn.get_current_context()

        ctx.array_class = "CudaArray"
        a = nn.NdArray(2, 5)
        a.cast(np.float32, ctx=ctx)
        # No error
        b = a.narrow(0, 0, 1)

        ctx.array_class = "CudaCachedArray"
        a = nn.NdArray(2, 5)
        a.cast(np.float32, ctx=ctx)
        # No error
        b = a.narrow(0, 0, 1)

        # CudaDlpackArray
        # Dlpack Array is created using nn.utils.dlpack.to_dlpack, but is not tested here because it is not NdArray.

        ctx.array_class = "CudaCachedUnifiedArray"
        a = nn.NdArray(2, 5)
        a.cast(np.float32, ctx=ctx)
        with pytest.raises(RuntimeError):
            b = a.narrow(0, 0, 1)

        ctx.array_class = "CudaCachedHostArray"
        a = nn.NdArray(2, 5)
        a.cast(np.float32, ctx=ctx)
        with pytest.raises(RuntimeError):
            b = a.narrow(0, 0, 1)

        ctx.array_class = "CudaCachedVirtualArray"
        a = nn.NdArray(2, 5)
        a.cast(np.float32, ctx=ctx)
        with pytest.raises(RuntimeError):
            b = a.narrow(0, 0, 1)
