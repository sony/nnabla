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

import pytest
import numpy as np
import nnabla as nn
import nnabla.functions as F
import scipy.signal as sig
from nbla_test_utils import list_context

# Proxy to get the appropriate context.
# Using convolution is natural since stft/istft depends on 1d convolution now.
ctx_list = [ctx_fname[0] for ctx_fname in list_context('Convolution')]


@pytest.mark.parametrize("ctx", ctx_list)
@pytest.mark.parametrize("window_size, stride, fft_size, window_type, center", [
    (256, 128, 512, 'hamming', True),
    (256, 128, 256, 'hanning', False),
    (256, 128, 512, None, True),
    (256, 128, 256, 'rectangular', False),
])
def test_istft(ctx, window_size, stride, fft_size, window_type, center):
    backend = ctx.backend[0].split(":")[0]
    if backend == 'cuda':
        pytest.skip('CUDA Convolution N-D is only supported in CUDNN extension')

    # clear all previous STFT conv/deconv kernels
    nn.clear_parameters()

    # Make sure that iSTFT(STFT(x)) = x
    x = np.random.randn(1, window_size * 10)

    nx = nn.Variable.from_numpy_array(x)
    with nn.context_scope(ctx):
        nyr, nyi = F._stft_v1(nx,
                              window_size=window_size,
                              stride=stride,
                              fft_size=fft_size,
                              window_type=window_type,
                              center=center)
        nz = F._istft_v1(nyr, nyi,
                         window_size=window_size,
                         stride=stride,
                         fft_size=fft_size,
                         window_type=window_type,
                         center=center)
    nz.forward()

    invalid = window_size - stride
    assert(np.allclose(nx.d[:, invalid:-invalid],
                       nz.d[:, invalid:-invalid],
                       atol=1e-5, rtol=1e-5))


@pytest.mark.parametrize("ctx", ctx_list)
@pytest.mark.parametrize("window_size, stride, fft_size, window_type", [
    (256, 128, 256, 'hanning'),
])
def test_stft(ctx, window_size, stride, fft_size, window_type):
    backend = ctx.backend[0].split(":")[0]
    if backend == 'cuda':
        pytest.skip('CUDA Convolution N-D is only supported in CUDNN extension')

    # clear all previous STFT conv/deconv kernels
    nn.clear_parameters()

    # Compare to `scipy.signal.stft` - only done if SciPy available
    x = np.random.randn(1, window_size * 10)

    nx = nn.Variable.from_numpy_array(x)

    with nn.context_scope(ctx):
        nyr, nyi = F._stft_v1(nx,
                              window_size=window_size,
                              stride=stride,
                              fft_size=fft_size,
                              window_type=window_type,
                              center=False)
    nn.forward_all([nyr, nyi])

    stft_nnabla = nyr.d + 1j * nyi.d

    window_type_scipy = window_type
    if window_type == 'rectangular' or window_type is None:
        window_type_scipy = 'boxcar'

    _f, _t, stft_scipy = sig.stft(x,
                                  window=window_type_scipy,
                                  nperseg=window_size,
                                  noverlap=window_size-stride,
                                  nfft=fft_size,
                                  boundary=None,
                                  padded=False)

    # scipy does a different scaling - take care here
    stft_nnabla /= fft_size // 2

    assert(np.allclose(stft_nnabla,
                       stft_scipy,
                       atol=1e-5, rtol=1e-5))


def ref_stft(x, window_size, stride, fft_size, window_type, center, pad_mode):
    x = nn.Variable.from_numpy_array(x)
    y_r, y_i = F._stft_v1(x, window_size, stride, fft_size,
                          window_type, center, pad_mode)
    y_r.forward()
    y_i.forward()

    return y_r.d, y_i.d


@pytest.mark.parametrize("ctx", ctx_list)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("window_size, stride, fft_size", [
    (16, 8, 16), (16, 8, 32),
])
@pytest.mark.parametrize("window_type", ["hanning", "hamming", "rectangular"])
@pytest.mark.parametrize("center", [True, False])
@pytest.mark.parametrize("pad_mode", ["reflect", "constant"])
def test_stft_forward_backward(ctx, seed, window_size, stride, fft_size, window_type, center, pad_mode):
    backend = ctx.backend[0].split(":")[0]
    if backend == 'cuda':
        pytest.skip('CUDA Convolution N-D is only supported in CUDNN extension')

    from nbla_test_utils import function_tester
    rng = np.random.RandomState(seed)

    x_shape = (2, window_size * 10)
    inputs = [rng.randn(*x_shape).astype(np.float32)]

    func_name = "STFTCuda" if backend == 'cudnn' else "STFT"

    function_tester(rng, F.stft, ref_stft, inputs, func_args=[
                    window_size, stride, fft_size, window_type, center, pad_mode], ctx=ctx, func_name=func_name, atol_f=2e-6, atol_b=2e-2, dstep=1e-2)


def ref_istft(y_r, y_i, window_size, stride, fft_size, window_type, center):
    y_r = nn.Variable.from_numpy_array(y_r)
    y_i = nn.Variable.from_numpy_array(y_i)
    x = F._istft_v1(y_r, y_i, window_size, stride,
                    fft_size, window_type, center)
    x.forward()

    return np.array(x.d)


@pytest.mark.parametrize("ctx", ctx_list)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("window_size, stride, fft_size", [
    (16, 8, 16), (16, 8, 32),
])
@pytest.mark.parametrize("window_type", ["hanning", "hamming", "rectangular"])
@pytest.mark.parametrize("center", [True, False])
def test_istft_forward_backward(ctx, seed, window_size, stride, fft_size, window_type, center):
    backend = ctx.backend[0].split(":")[0]
    if backend == 'cuda':
        pytest.skip('CUDA Convolution N-D is only supported in CUDNN extension')

    from nbla_test_utils import function_tester
    rng = np.random.RandomState(seed)

    # generate istft inputs by calling stft
    x_shape = (2, window_size * 10)
    stft_input = rng.randn(*x_shape).astype(np.float32)
    y_r, y_i = ref_stft(stft_input, window_size, stride,
                        fft_size, window_type, center, pad_mode='reflect')
    istft_inputs = [y_r, y_i]

    func_name = "ISTFTCuda" if backend == 'cudnn' else "ISTFT"

    function_tester(rng, F.istft, ref_istft, istft_inputs, func_args=[
                    window_size, stride, fft_size, window_type, center], ctx=ctx, func_name=func_name, atol_f=2e-6, atol_b=2e-2, dstep=1e-2)
