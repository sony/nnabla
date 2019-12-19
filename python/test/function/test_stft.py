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
        nyr, nyi = F.stft(nx,
                          window_size=window_size,
                          stride=stride,
                          fft_size=fft_size,
                          window_type=window_type,
                          center=center)
        nz = F.istft(nyr, nyi,
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
        nyr, nyi = F.stft(nx,
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
