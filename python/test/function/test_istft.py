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

from operator import is_
import pytest
import numpy as np
import nnabla as nn
import nnabla.functions as F
import scipy.signal as sig
import librosa
from nbla_test_utils import list_context
from test_stft import create_stft_input_shape, create_window_func, is_nola_violation, ref_stft

# Proxy to get the appropriate context.
# Using convolution is natural since stft/istft depends on 1d convolution now.
ctx_list = [ctx_fname[0] for ctx_fname in list_context('Convolution')]


def ref_istft(y_r, y_i, window_size, stride, fft_size, window_type, center, pad_mode, as_stft_backward):
    if not as_stft_backward:
        # Use librosa.istft as the forward reference.

        # Convert to librosa.istft input format.
        y = y_r + 1j * y_i

        # Get original signal length.
        x_shape = create_stft_input_shape(window_size)
        length = x_shape[1]

        # librosa.istft does not support batched input.
        window_type = 'hann' if window_type == 'hanning' else window_type
        b = y.shape[0]
        xs = []
        for i in range(b):
            x = librosa.istft(y[i], hop_length=stride,
                              win_length=window_size, window=window_type, center=center, length=length)
            xs.append(x)
        return np.array(xs)
    else:
        # Use F.stft backward as the reference

        y_r = nn.Variable.from_numpy_array(y_r)
        y_i = nn.Variable.from_numpy_array(y_i)

        # Just create stft inputs
        x = F.istft(y_r, y_i, window_size, stride, fft_size,
                    window_type, center, pad_mode, True)

        # Execute istft backward
        x.need_grad = True
        x.grad.zero()
        z_r, z_i = F.stft(x, window_size, stride,
                          fft_size, window_type, center, pad_mode)

        z_r.g = y_r.d
        z_i.g = y_i.d
        z = F.sink(z_r, z_i, one_input_grad=False)
        z.forward()
        z.backward()

        return x.g


def check_nola_violation(y_r, y_i, window_size, stride, fft_size, window_type, center, pad_mode, as_stft_backward):
    # Check reference raise
    try:
        # Use PyTorch to check NOLA condition becasue librosa does not raise.
        # If PyTorch is not installed, NOLA condition test for reference is skipped.
        import torch

        def ref_istft_torch(y_r, y_i, window_size, stride, fft_size, window_type, center):
            y_r = np.reshape(y_r, y_r.shape + (1,))
            y_i = np.reshape(y_i, y_i.shape + (1,))
            y = np.concatenate((y_r, y_i), axis=3)

            y = torch.tensor(y)
            y = torch.view_as_complex(y)
            window = torch.tensor(create_window_func(window_type, window_size))

            x_shape = create_stft_input_shape(window_size)
            length = x_shape[1]

            x = torch.istft(y, n_fft=fft_size, hop_length=stride,
                            win_length=window_size, window=window, center=center, length=length)
            return x

        with pytest.raises(RuntimeError, match=r"window overlap add"):
            # NOLA condition is checked during forward execution.
            ref_istft_torch(y_r, y_i, window_size, stride,
                            fft_size, window_type, center)
    except:
        # Install PyTorch to check NOLA condition validation of reference istft.
        pass

    # Check NNabla raise
    y_r = nn.Variable.from_numpy_array(y_r)
    y_i = nn.Variable.from_numpy_array(y_i)
    with pytest.raises(RuntimeError, match=r"NOLA\(Nonzero Overlap Add\) condition is not met."):
        # NOLA condition is checked during setup.
        _ = F.istft(y_r, y_i, window_size, stride,
                    fft_size, window_type, center, pad_mode, as_stft_backward)


'''
@pytest.mark.parametrize("ctx", ctx_list)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("window_size, stride, fft_size", [
    (16, 8, 16), (16, 4, 16), (16, 8, 32),
])
@pytest.mark.parametrize("window_type", ["hanning", "hamming", "rectangular"])
@pytest.mark.parametrize("center", [True, False])
@pytest.mark.parametrize("pad_mode", ["reflect", "constant"])
@pytest.mark.parametrize("as_stft_backward", [False, True])
def test_istft_forward_backward(ctx, seed, window_size, stride, fft_size, window_type, center, pad_mode, as_stft_backward):
    backend = ctx.backend[0].split(":")[0]
    if backend == 'cuda':
        pytest.skip('CUDA Convolution N-D is only supported in CUDNN extension')

    if not as_stft_backward:
        if pad_mode != "constant":
            pytest.skip(
                '`pad_mode != "constant"` is only for `as_stft_backward == True`')

    func_name = "ISTFTCuda" if backend == 'cudnn' else "ISTFT"

    from nbla_test_utils import function_tester
    rng = np.random.RandomState(seed)

    # Generate istft inputs by calling stft
    x_shape = create_stft_input_shape(window_size)
    stft_input = rng.randn(*x_shape).astype(np.float32)
    y_r, y_i = ref_stft(stft_input, window_size, stride,
                        fft_size, window_type, center, pad_mode, False)
    istft_inputs = [y_r, y_i]

    # Check violation of NOLA condition
    if not as_stft_backward:
        length = x_shape[1]
        if is_nola_violation(window_type, window_size, stride, fft_size, length, center):
            check_nola_violation(
                y_r, y_i, window_size, stride, fft_size, window_type, center, pad_mode, as_stft_backward)
            return

    function_tester(rng, F.istft, ref_istft, istft_inputs, func_args=[
                    window_size, stride, fft_size, window_type, center, pad_mode, as_stft_backward], ctx=ctx, func_name=func_name, atol_f=1e-5, atol_b=3e-2, dstep=1e-2)
'''


@pytest.mark.parametrize("ctx", ctx_list)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("window_size, stride, fft_size", [
    (16, 8, 16), (16, 4, 16), (16, 8, 32),
])
@pytest.mark.parametrize("window_type", ["hanning", "hamming", "rectangular"])
@pytest.mark.parametrize("center", [True, False])
@pytest.mark.parametrize("pad_mode", ["reflect", "constant"])
@pytest.mark.parametrize("as_stft_backward", [False, True])
def test_istft_double_backward(ctx, seed, window_size, stride, fft_size, window_type, center, pad_mode, as_stft_backward):
    backend = ctx.backend[0].split(":")[0]
    if backend == 'cuda':
        pytest.skip('CUDA Convolution N-D is only supported in CUDNN extension')

    if not as_stft_backward:
        if pad_mode != "constant":
            pytest.skip(
                '`pad_mode != "constant"` is only for `as_stft_backward == True`')

    from nbla_test_utils import backward_function_tester
    rng = np.random.RandomState(seed)

    # Generate istft inputs by calling stft
    x_shape = create_stft_input_shape(window_size)
    stft_input = rng.randn(*x_shape).astype(np.float32)
    y_r, y_i = ref_stft(stft_input, window_size, stride,
                        fft_size, window_type, center, pad_mode, False)
    istft_inputs = [y_r, y_i]

    if not as_stft_backward:
        # Skip for NOLA condition violation
        length = x_shape[1]
        if is_nola_violation(window_type, window_size, stride, fft_size, length, center):
            pytest.skip('NOLA condition violation.')

    rng = np.random.RandomState(seed)
    func_args = [window_size, stride, fft_size,
                 window_type, center, pad_mode, as_stft_backward]
    backward_function_tester(rng, F.istft,
                             inputs=istft_inputs,
                             func_args=func_args,
                             ctx=ctx,
                             atol_accum=6e-2)


# Make sure that ISTFT(STFT(x)) = x
@pytest.mark.parametrize("ctx", ctx_list)
@pytest.mark.parametrize("window_size, stride, fft_size", [
    (256, 128, 512), (256, 128, 256), (256, 64, 512),
])
@pytest.mark.parametrize("window_type", ["hanning", "hamming", "rectangular"])
@pytest.mark.parametrize("center", [True, False])
@pytest.mark.parametrize("pad_mode", ["reflect", "constant"])
def test_stft_istft_identity(ctx, window_size, stride, fft_size, window_type, center, pad_mode):
    backend = ctx.backend[0].split(":")[0]
    if backend == 'cuda':
        pytest.skip('CUDA Convolution N-D is only supported in CUDNN extension')

    x_shape = create_stft_input_shape(window_size)
    x = np.random.randn(*x_shape)

    # Skip for NOLA condition violation
    length = x_shape[1]
    if is_nola_violation(window_type, window_size, stride, fft_size, length, center):
        pytest.skip('NOLA condition violation.')
        return

    x = nn.Variable.from_numpy_array(x)
    with nn.context_scope(ctx):
        yr, yi = F.stft(x, window_size, stride, fft_size,
                        window_type, center, pad_mode)
        z = F.istft(yr, yi, window_size, stride, fft_size,
                    window_type, center, pad_mode="constant")
    z.forward()

    assert (np.allclose(x.d, z.d, atol=1e-5, rtol=1e-5))
