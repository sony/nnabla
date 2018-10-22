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

# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

import argparse
from nnabla import logger
import scipy
import time

from datasets import data_iterator
import matplotlib.pylab as plt
from networks import Generator
from functions import pixel_wise_feature_vector_normalization
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import numpy as np


def get_descriptors_for_minibatch(minibatch, nhood_size, nhoods_per_image):
    S = minibatch.shape  # (minibatch, channel, height, width)
    assert len(S) == 4 and S[1] == 3
    N = nhoods_per_image * S[0]
    H = nhood_size // 2
    nhood, chan, x, y = np.ogrid[0:N, 0:3, -H:H+1, -H:H+1]
    img = nhood // nhoods_per_image
    x = x + np.random.randint(H, S[3] - H, size=(N, 1, 1, 1))
    y = y + np.random.randint(H, S[2] - H, size=(N, 1, 1, 1))
    idx = ((img * S[1] + chan) * S[2] + y) * S[3] + x
    return minibatch.flat[idx]

# ----------------------------------------------------------------------------


def finalize_descriptors(desc):
    if isinstance(desc, list):
        desc = np.concatenate(desc, axis=0)
    assert desc.ndim == 4  # (neighborhood, channel, height, width)
    desc -= np.mean(desc, axis=(0, 2, 3), keepdims=True)
    desc /= np.std(desc, axis=(0, 2, 3), keepdims=True)
    desc = desc.reshape(desc.shape[0], -1)
    return desc

# ----------------------------------------------------------------------------


def _sliced_wasserstein(A, B, dirs_per_repeat):
    # (neighborhood, descriptor_component)
    assert A.ndim == 2 and A.shape == B.shape
    results = []
    # (descriptor_component, direction)
    dirs = np.random.randn(A.shape[1], dirs_per_repeat)
    # normalize descriptor components for each direction
    dirs /= np.sqrt(np.sum(np.square(dirs), axis=0, keepdims=True))
    dirs = dirs.astype(np.float32)
    # (neighborhood, direction)
    projA = np.matmul(A, dirs)
    projB = np.matmul(B, dirs)
    # sort neighborhood projections for each direction
    projA = np.sort(projA, axis=0)
    projB = np.sort(projB, axis=0)
    # pointwise wasserstein distances
    dists = np.abs(projA - projB)
    # average over neighborhoods and directions
    results.append(np.mean(dists))
    # average over repeats
    return results

# ----------------------------------------------------------------------------


def downscale_minibatch(minibatch, lod):
    if lod == 0:
        return minibatch
    t = minibatch.astype(np.float32)
    for i in range(lod):
        t = (t[:, :, 0::2, 0::2] + t[:, :, 0::2, 1::2] +
             t[:, :, 1::2, 0::2] + t[:, :, 1::2, 1::2]) * 0.25
    return np.round(t).clip(0, 255).astype(np.uint8)

# ----------------------------------------------------------------------------


gaussian_filter = np.float32([
    [1, 4, 6, 4, 1],
    [4, 16, 24, 16, 4],
    [6, 24, 36, 24, 6],
    [4, 16, 24, 16, 4],
    [1, 4, 6, 4, 1]]) / 256.0


def pyr_down(minibatch):  # matches cv2.pyrDown()
    assert minibatch.ndim == 4
    return scipy.ndimage.convolve(minibatch, gaussian_filter[np.newaxis, np.newaxis, :, :], mode='mirror')[:, :, ::2, ::2]


def pyr_up(minibatch):  # matches cv2.pyrUp()
    assert minibatch.ndim == 4
    S = minibatch.shape
    res = np.zeros((S[0], S[1], S[2] * 2, S[3] * 2), minibatch.dtype)
    res[:, :, ::2, ::2] = minibatch
    return scipy.ndimage.convolve(res, gaussian_filter[np.newaxis, np.newaxis, :, :] * 4.0, mode='mirror')


def generate_laplacian_pyramid(minibatch, num_levels):
    pyramid = [np.float32(minibatch)]
    for i in range(1, num_levels):
        pyramid.append(pyr_down(pyramid[-1]))
        pyramid[-2] -= pyr_up(pyramid[-1])
    return pyramid


def reconstruct_laplacian_pyramid(pyramid):
    minibatch = pyramid[-1]
    for level in pyramid[-2::-1]:
        minibatch = pyr_up(minibatch) + level
    return minibatch

# ----------------------------------------------------------------------------


def sliced_wasserstein(A, B, dir_repeats, dirs_per_repeat):
    # (neighborhood, descriptor_component)
    assert A.ndim == 2 and A.shape == B.shape
    results = []

    import multiprocessing as mp
    from multiprocessing.pool import ThreadPool
    pool = ThreadPool(processes=mp.cpu_count())
    results = []
    score = []
    for repeat in range(dir_repeats):
        ret = pool.apply_async(_sliced_wasserstein, (A, B, dirs_per_repeat))
        results.append(ret)
    pool.close()
    pool.join()
    for repeat in range(dir_repeats):
        dists_mean = results[repeat].get()
        score.append(dists_mean)
    return np.mean(score)


def compute_metric(di, gen, latent, num_minibatch, nhoods_per_image,
                   nhood_size, level_list, dir_repeats,
                   dirs_per_repeat, hyper_sphere=True):
    logger.info("Generate images")
    st = time.time()
    real_descriptor = [[] for _ in level_list]
    fake_descriptor = [[] for _ in level_list]
    for k in range(num_minibatch):
        logger.info("iter={} / {}".format(k, num_minibatch))
        real, _ = di.next()
        real = np.uint8((real + 1.) / 2. * 255)

        B = len(real)
        z_data = np.random.randn(B, latent, 1, 1)
        z = nn.Variable.from_numpy_array(z_data)
        z = pixel_wise_feature_vector_normalization(z) if hyper_sphere else z
        y = gen(z)
        fake = y.d
        fake = np.uint8((y.d + 1.) / 2. * 255)

        for i, desc in enumerate(generate_laplacian_pyramid(real, len(level_list))):
            real_descriptor[i].append(get_descriptors_for_minibatch(
                desc, nhood_size, nhoods_per_image))
        for i, desc in enumerate(generate_laplacian_pyramid(fake, len(level_list))):
            fake_descriptor[i].append(get_descriptors_for_minibatch(
                desc, nhood_size, nhoods_per_image))
    logger.info(
        "Elapsed time for generating images: {} [s]".format(time.time() - st))

    logger.info("Compute Sliced Wasserstein Distance")
    scores = []
    for i, level in enumerate(level_list):
        st = time.time()
        real = finalize_descriptors(real_descriptor[i])
        fake = finalize_descriptors(fake_descriptor[i])
        scores.append(sliced_wasserstein(
            real, fake, dir_repeats, dirs_per_repeat))
        logger.info("Level: {}, dist: {}".format(level, scores[-1]))
        logger.info(
            "Elapsed time: {} [s] at {}-th level".format(time.time() - st, i))
    return scores
