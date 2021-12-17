# Copyright 2019,2020,2021 Sony Corporation.
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
import librosa
from nbla_test_utils import list_context

# Proxy to get the appropriate context.
# Using convolution is natural since stft/istft depends on 1d convolution now.
ctx_list = [ctx_fname[0] for ctx_fname in list_context('Convolution')]


def create_window_func(window_type, window_size):
    if window_type == 'hanning':
        return np.hanning(window_size + 1)[:-1]
    elif window_type == 'hamming':
        return np.hamming(window_size + 1)[:-1]
    elif window_type == 'rectangular' or window_type is None:
        return np.ones(window_size)


def create_inv_window_func(window_type, window_size, stride, fft_size, length):
    w = create_window_func(window_type, window_size)

    # Padding with zero
    if (window_size < fft_size):
        diff = fft_size - window_size
        w = np.pad(
            w, (diff//2, diff - diff//2), mode='constant')

    w = nn.Variable.from_numpy_array(w)
    w = w.reshape((1, 1, w.shape[0]))

    # Overwrap add for window
    ones = F.constant(1, (1, 1, (length - fft_size) // stride + 1))
    iw = F.deconvolution(ones, w * w, stride=(stride,))
    iw.forward()

    # Flatten
    iw = np.reshape(iw.d, (iw.shape[2],))

    return iw


def is_nola_violation(window_type, window_size, stride, fft_size, length, center):
    inv_window = create_inv_window_func(
        window_type, window_size, stride, fft_size, length)

    # Padding region is ignored for NOLA check since division is not occured.
    if center:
        pad = fft_size // 2
        inv_window = inv_window[pad:-pad]

    # True if zero element found.
    return np.any(inv_window < 1e-11)


def ref_stft(x, window_size, stride, fft_size, window_type, center, pad_mode, as_istft_backward):
    if not as_istft_backward:
        # Use librosa.stft as the forward reference.

        # librosa.stft does not support batched input.
        b = x.shape[0]
        ys = []
        for i in range(b):
            y = librosa.stft(x[i], n_fft=fft_size, hop_length=stride, win_length=window_size,
                             window=window_type, center=center, pad_mode=pad_mode)
            ys.append(y)

        # Convert to nnabla stft output format
        ys = np.array(ys)
        y_r = ys.real
        y_i = ys.imag

        return y_r, y_i
    else:
        # Use F.istft backward as the reference

        x = nn.Variable.from_numpy_array(x)

        # Just create istft inputs
        y_r, y_i = F.stft(x, window_size, stride, fft_size,
                          window_type, center, pad_mode)

        # Execute istft backward
        y_r.need_grad = True
        y_i.need_grad = True
        y_r.grad.zero()
        y_i.grad.zero()
        z = F.istft(y_r, y_i, window_size, stride,
                    fft_size, window_type, center, pad_mode)

        z.forward()
        z.backward(x.data)

        return y_r.g, y_i.g


def create_stft_input_shape(window_size):
    return (2, window_size * 10)


@pytest.mark.parametrize("ctx", ctx_list)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("window_size, stride, fft_size", [
    (16, 8, 16), (16, 4, 16), (16, 8, 32),
])
@pytest.mark.parametrize("window_type", ["hanning", "hamming", "rectangular"])
@pytest.mark.parametrize("center", [True, False])
@pytest.mark.parametrize("pad_mode", ["reflect", "constant"])
@pytest.mark.parametrize("as_istft_backward", [False, True])
def test_stft_forward_backward(ctx, seed, window_size, stride, fft_size, window_type, center, pad_mode, as_istft_backward):
    backend = ctx.backend[0].split(":")[0]
    if backend == 'cuda':
        pytest.skip('CUDA Convolution N-D is only supported in CUDNN extension')
    func_name = "STFTCuda" if backend == 'cudnn' else "STFT"

    from nbla_test_utils import function_tester
    rng = np.random.RandomState(seed)

    x_shape = create_stft_input_shape(window_size)
    inputs = [rng.randn(*x_shape).astype(np.float32)]

    # Some conditions are skipped for the reason of use of ISTFT function.
    if as_istft_backward:
        # Ignore violation of NOLA condition
        length = x_shape[1]
        if is_nola_violation(window_type, window_size, stride, fft_size, length, center):
            pytest.skip('NOLA condition violation.')

        if pad_mode != "constant":
            pytest.skip('`pad_mode` must be "constant" when `as_istft_backward == True`. Normal ISTFT never use `pad_mode` and just slice the output. Thus, STFT as a backward of normal ISTFT, STFT must be `pad_mode == constant`')

    function_tester(rng, F.stft, ref_stft, inputs, func_args=[
                    window_size, stride, fft_size, window_type, center, pad_mode, as_istft_backward], ctx=ctx, func_name=func_name, atol_f=2e-6, atol_b=2e-2, dstep=1e-2)


@pytest.mark.parametrize("ctx", ctx_list)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("window_size, stride, fft_size", [
    (16, 8, 16), (16, 8, 32),
])
@pytest.mark.parametrize("window_type", ["hanning", "hamming", "rectangular"])
@pytest.mark.parametrize("center", [True, False])
@pytest.mark.parametrize("pad_mode", ["reflect", "constant"])
@pytest.mark.parametrize("as_istft_backward", [False, True])
def test_stft_double_backward(ctx, seed, window_size, stride, fft_size, window_type, center, pad_mode, as_istft_backward):
    from nbla_test_utils import backward_function_tester

    backend = ctx.backend[0].split(":")[0]
    if backend == 'cuda':
        pytest.skip('CUDA Convolution N-D is only supported in CUDNN extension')

    rng = np.random.RandomState(seed)
    x_shape = create_stft_input_shape(window_size)
    inputs = [rng.randn(*x_shape).astype(np.float32)]

    # Some conditions are skipped for the reason of ISTFT use.
    if as_istft_backward:
        # Ignore violation of NOLA condition
        length = x_shape[1]
        if is_nola_violation(window_type, window_size, stride, fft_size, length, center):
            pytest.skip('NOLA condition violation.')

        if pad_mode != "constant":
            # Normal ISTFT never use `pad_mode` and just slice the output. Thus, `padmode` must be `"constant"` for STFT as a backward of normal ISTFT.`
            pytest.skip(
                '`pad_mode` must be "constant" when `as_istft_backward == True`.')

    func_args = [window_size, stride, fft_size,
                 window_type, center, pad_mode, as_istft_backward]
    backward_function_tester(rng, F.stft,
                             inputs=inputs,
                             func_args=func_args,
                             ctx=ctx)
