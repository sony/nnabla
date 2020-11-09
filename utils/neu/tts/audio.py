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

import copy

import librosa as lr
import numpy as np
from scipy import signal


def amp_2_db(x):
    r"""Convert a signal from amplitude to decibel."""
    return 20 * np.log10(np.maximum(1e-5, x))


def db_2_amp(x):
    r"""Convert a signal from decibel to amplitude."""
    return np.power(10.0, x * 0.05)


def preemphasis(x, factor=0.97):
    r"""Apply preemphasis to a signal."""
    return signal.lfilter([1, -factor], [1], x)


def rev_preemphasis(x, factor=0.97):
    r"""Apply inverse preemphasis to a signal."""
    return signal.lfilter([1], [1, -factor], x)


def normalize(x, hp):
    r"""Normalize spectrogram into a range of [0, 1].

    Args:
        x (numpy.ndarray): The input spectrogram of shape (freq x time).
        max_db (float): Maximum intensity in decibel.
        ref_db (float): Reference intensity in decibel.

    Returns:
        numpy.ndarray: An (freq x time) array of values in [0, 1].
    """
    x = amp_2_db(x)
    x = np.clip((x - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)
    return x


def denormalize(x, hp):
    r"""Denormalize spectrogram.

    Args:
        x (numpy.ndarray): The input spectrogram of shape (freq x time).
        hp (HParams): A container for hyperparameters.
            - max_db (float): Maximum intensity in decibel.
            - ref_db (float): Reference intensity in decibel.

    Returns:
        numpy.ndarray: Spectrogram of shape (freq x time).
    """
    x = np.clip(x, 0, 1)
    x = db_2_amp(x * hp.max_db - hp.max_db + hp.ref_db)
    return x


def spec2mel(spectrogram, sr, n_fft, n_mels):
    r"""Convert a spectrogram to a mel-spectrogram.

    Args:
        spectrogram (numpy.ndarray): A spectrogram of shape (freq x time).
        sr (int): Sampling rate of the incoming signal.
        n_fft (int): Number of FFT components.
        n_mels (int): number of Mel bands to generate.

    Returns:
        numpy.ndarray: Mel spectrogram of shape (n_mels x time).
    """
    mel_basis = lr.filters.mel(sr, n_fft, n_mels)  # mel spectrogram
    mel = np.dot(mel_basis, spectrogram)  # (n_mels, t)
    return mel


def wave2spec(wave, hp):
    r"""Convert a waveform to spectrogram.

    Args:
        wave (np.ndarray): An input waveform in 1D array.
        hp (HParams): A container for hyperparameters.
            - n_fft (int): Length of the windowed signal after padding with zeros.
            - hop_length (int): Number of audio samples between adjacent STFT columns.
            - win_length (int): Each frame of audio is windowed by `window()` of
                length `win_length` and then padded with zeros to match `n_fft`.

    Returns:
        numpy.ndarray: Spectrogram of shape (1+n_fft//2, time).
    """
    linear = lr.stft(wave, n_fft=hp.n_fft, hop_length=hp.hop_length,
                     win_length=hp.win_length)
    mag = np.abs(linear)

    return mag


def spec2wave(spectrogram, hp):
    r"""Griffin-Lim's algorithm.

    Args:
        spectrogram (numpy.ndarray): A spectrogram input of shape
            `(1+n_fft//2, time)`.
        hp (HParams): A container for hyperparameters.            
            - n_fft (int): Number of FFT components.
            - hop_length (int): Number of audio samples between adjacent STFT columns.
            - win_length (int): Each frame of audio is windowed by `window()` of
                length `win_length` and then padded with zeros to match `n_fft`.
            - n_iter (int, optional): [description]. Defaults to 50.

    Returns:
        np.ndarray: An 1D array representing the waveform.
    """
    X_best = np.power(copy.deepcopy(spectrogram), hp.power)
    for i in range(hp.n_iter):
        X_t = lr.istft(X_best, hop_length=hp.hop_length,
                       win_length=hp.win_length, window="hann")
        est = lr.stft(X_t, hp.n_fft, hop_length=hp.hop_length,
                      win_length=hp.win_length)
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = spectrogram * phase

    X_t = lr.istft(X_best, hop_length=hp.hop_length,
                   win_length=hp.win_length, window="hann")

    return np.real(X_t)


def synthesize_from_spec(spec, hp):
    r"""Convert a waveform from its spectrogram.

    Args:
        spec (numpy.ndarray): A spectrogram of shape (time, (n_fft//2+1)).
        hp (HParams): A container for hyperparameters.
            - max_db (float): Maximum intensity in decibel.
            - ref_db (float): Reference intensity in decibels.
            - preemphasis (float): A pre-emphasis factor.
            - n_fft (int): Length of the windowed signal after padding with zeros.
            - hop_length (int): Number of audio samples between adjacent STFT columns.
            - win_length (int): Each frame of audio is windowed by `window()` of
                length `win_length` and then padded with zeros to match `n_fft`.
            - n_iter (int, optional): The number of iterations used in the Griffin-Lim
                algorithm. Defaults to 50.

    Returns:
        numpy.ndarray: An 1D array of float values.
    """
    spec = denormalize(spec.T, hp)

    # wave reconstruction
    wave = spec2wave(spec, hp)

    if hasattr(hp, 'preemphasis'):
        wave = rev_preemphasis(wave, factor=hp.preemphasis)  # de-preemphasis
    wave, _ = lr.effects.trim(wave)  # trim

    return wave.astype(np.float32)
