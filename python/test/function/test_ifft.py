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

import pytest
import numpy as np
import nnabla.functions as F
from nbla_test_utils import list_context

ctxs = list_context('IFFT')


def ref_ifft(x, signal_ndim, normalized):
    from nbla_test_utils import convert_to_float2_array, convert_to_complex_array

    x_data_complex = convert_to_complex_array(x)
    batch_dims = x_data_complex.shape[0:len(
        x_data_complex.shape) - signal_ndim]
    ref_data_complex = np.fft.ifftn(x_data_complex,
                                    axes=np.arange(signal_ndim) +
                                    len(batch_dims),
                                    norm="ortho" if normalized else None)

    ref_data_float2 = convert_to_float2_array(
        ref_data_complex).astype(np.float32)
    return ref_data_float2


def ref_grad_ifft(x, dy, signal_ndim, normalized, **kw):
    from nbla_test_utils import convert_to_float2_array, convert_to_complex_array

    dy_complex = convert_to_complex_array(dy)
    batch_dims = dy_complex.shape[0:len(dy_complex.shape) - signal_ndim]
    ref_grad_complex = np.fft.fftn(dy_complex,
                                   axes=np.arange(signal_ndim) +
                                   len(batch_dims),
                                   norm="ortho" if normalized else None)
    if not normalized:
        scale = np.prod(ref_grad_complex.shape[len(batch_dims):]) if len(
            batch_dims) > 0 else np.prod(ref_grad_complex.shape)
        ref_grad_complex /= scale
    ref_grad = convert_to_float2_array(ref_grad_complex).astype(np.float32)
    return ref_grad.flatten()


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("batch_dims", [(), (4, ), (8, 4)])
@pytest.mark.parametrize("signal_ndim, dims", [(1, (32,)), (2, (32, 16)), (3, (16, 16, 16))])
@pytest.mark.parametrize("normalized", [True, False])
def test_fft_forward_backward(seed, ctx, func_name, batch_dims,
                              signal_ndim, dims, normalized):

    if func_name == "IFFT":
        pytest.skip("Not implemented in CPU.")

    from nbla_test_utils import function_tester, convert_to_float2_array, convert_to_complex_array
    rng = np.random.RandomState(seed)
    shape = batch_dims + dims
    x_data_complex = rng.rand(*shape) + 1j * rng.rand(*shape)
    x_data = convert_to_float2_array(x_data_complex)
    inputs = [x_data]
    func_args = [signal_ndim, normalized]
    function_tester(rng,
                    F.ifft,
                    ref_ifft,
                    inputs,
                    func_args=func_args,
                    atol_f=1e-4,
                    atol_b=1e-4,
                    backward=[True],
                    ctx=ctx,
                    func_name=func_name,
                    ref_grad=ref_grad_ifft,
                    disable_half_test=True)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("batch_dims", [(), (4, ), (8, 4)])
@pytest.mark.parametrize("signal_ndim, dims", [(1, (8,)), (2, (8, 4)), (3, (8, 4, 2))])
@pytest.mark.parametrize("normalized", [True, False])
def test_fft_double_backward(seed, ctx, func_name, batch_dims,
                             signal_ndim, dims, normalized):

    if func_name == "IFFT":
        pytest.skip("Not implemented in CPU.")

    from nbla_test_utils import backward_function_tester, convert_to_float2_array, convert_to_complex_array
    rng = np.random.RandomState(seed)
    shape = batch_dims + dims
    x_data_complex = rng.rand(*shape) + 1j * rng.rand(*shape)
    x_data = convert_to_float2_array(x_data_complex)
    inputs = [x_data]
    func_args = [signal_ndim, normalized]
    backward_function_tester(rng,
                             F.ifft,
                             inputs,
                             func_args=func_args,
                             atol_f=1e-4,
                             atol_accum=5e-2,
                             backward=[True],
                             ctx=ctx)
