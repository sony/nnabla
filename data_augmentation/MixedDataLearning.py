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

import numpy as np
import nnabla as nn
import nnabla.functions as F


def softmax_cross_entropy_loss_vlabel(pred, vlabel):
    # The shape of vlabel is supposed to be (batch_size, n_class)
    logp = F.log_softmax(pred)
    loss = -1.0 * F.mean(F.sum(vlabel * logp, axis=1))
    return loss

# Abstract class for learning with mixed data


class MixedDataLearning(object):
    def __init__(self):
        # Set params for mixing data
        return NotImplemented

    def set_mix_ratio():
        return NotImplemented

    def mix_data(self, x, y):
        # Mix data given x and y, and return mix_x and mix_y
        # x, y, mix_x, and mix_y are supposed to be nn.Variable
        return NotImplemented

    def loss(self, pred, mix_y):
        # Calculate a classification loss given mix_y and prediction results of mix_x.
        # Both pred and mix_y are supposed to be a nn.Variable
        return NotImplemented


# Mixup
class MixupLearning(MixedDataLearning):
    def __init__(self, batch_size, alpha=0.5):
        # Set params for mixing data
        # For mixup, set alpha for the beta distribution that generates interpolation ratios.
        self._batch_size = batch_size
        self._alpha = alpha
        self._lam = nn.Variable((batch_size, 1))

    def set_mix_ratio(self):
        if self._alpha > 0.0:
            self._lam.d = np.random.beta(
                self._alpha, self._alpha, self._batch_size).reshape((self._batch_size, 1))
        else:
            self._lam.d = np.ones((self._batch_size, 1))

    def mix_data(self, x, y):
        # Mix data given x and y, and return mix_x and mix_y
        # Both y and mix_y are supposed to be nn.Variable((batch_size, n_class))
        batch_size = x.shape[0]
        ind = np.random.permutation(batch_size)
        x0 = x
        y0 = y
        x1 = x0[ind]
        y1 = y0[ind]
        mix_x = self._lam.reshape((-1, 1, 1, 1)) * x0 + \
            (1.0-self._lam.reshape((-1, 1, 1, 1))) * x1
        mix_y = self._lam * y0 + (1.0-self._lam) * y1
        return mix_x, mix_y

    def loss(self, pred, mix_y):
        # Calculate a classification loss given mix_y and prediction results of mix_x.
        # Both pred and mix_y are supposed to be a nn.Variable
        return softmax_cross_entropy_loss_vlabel(pred, mix_y)


# Cutmix
class CutmixLearning(MixedDataLearning):
    def __init__(self, shape_of_batch, alpha=0.5, cutmix_prob=0.5):
        # Set params for mixing data
        # For cutmix, set alpha for the beta distribution that generates cutting area ratio.
        # cutmix_prob controls a probablity to conduct cutmix for each batch
        # shape_of_batch should be (batch_size, n_channels, height, width)
        self._batch_size = shape_of_batch[0]
        self._alpha = alpha
        self._cutmix_prob = cutmix_prob
        self._mask = nn.Variable(shape_of_batch)
        self._lam = nn.Variable((shape_of_batch[0], 1))

    def set_mix_ratio(self):
        # How to get a random bounding box
        def rand_bbox(shape_of_x, lam):
            width = shape_of_x[3]
            height = shape_of_x[2]
            cut_ratio = np.sqrt(1.0 - lam)
            cut_w = np.int(width * cut_ratio)
            cut_h = np.int(height * cut_ratio)

            cx = np.random.randint(width)
            cy = np.random.randint(height)

            bbx0 = np.clip(cx - cut_w//2, 0, width)
            bby0 = np.clip(cy - cut_h//2, 0, height)
            bbx1 = np.clip(cx + cut_w//2, 0, width)
            bby1 = np.clip(cy + cut_h//2, 0, height)

            return bbx0, bby0, bbx1, bby1

        def get_mask(shape_of_x, bbx0, bby0, bbx1, bby1):
            mask = np.zeros(shape_of_x)
            mask[:, :, bby0:bby1, bbx0:bbx1] = 1.0
            return mask

        if self._alpha > 0.0 and np.random.rand() <= self._cutmix_prob:
            lam_tmp = np.random.beta(self._alpha, self._alpha)
            bbx0, bby0, bbx1, bby1 = rand_bbox(self._mask.shape, lam_tmp)
            self._mask.d = get_mask(self._mask.shape, bbx0, bby0, bbx1, bby1)
            self._lam.d = (1.0 - ((bbx1-bbx0)*(bby1-bby0)/(
                self._mask.shape[2]*self._mask.shape[3]))) * np.ones((self._batch_size, 1))
        else:
            self._mask.d = np.zeros(self._mask.shape)
            self._lam.d = np.ones((self._batch_size, 1))

    def mix_data(self, x, y):
        # Mix data given x and y, and return mix_x and mix_y
        # Both y and mix_y are supposed to be nn.Variable((batch_size, n_class))
        batch_size = x.shape[0]
        ind = np.random.permutation(batch_size)
        x0 = x
        y0 = y
        x1 = x0[ind]
        y1 = y0[ind]
        mix_x = (1.0 - self._mask) * x0 + self._mask * x1
        mix_y = self._lam * y0 + (1.0 - self._lam) * y1
        return mix_x, mix_y

    def loss(self, pred, mix_y):
        # Calculate a classification loss given mix_y and prediction results of mix_x.
        # Both pred and mix_y are supposed to be a nn.Variable
        return softmax_cross_entropy_loss_vlabel(pred, mix_y)


# VH-Mixup
class VHMixupLearning(MixedDataLearning):
    def __init__(self, shape_of_batch, alpha=0.5):
        # Set params for mixing data
        # For vh-mixup, set alpha for the beta distribution that generates interpolation ratios.
        # shape_of_batch should be (batch_size, n_channels, height, width)
        self._batch_size = shape_of_batch[0]
        self._maskv = nn.Variable(shape_of_batch)
        self._maskh = nn.Variable(shape_of_batch)
        self._lamx = nn.Variable([shape_of_batch[0], 1])
        self._lamy = nn.Variable([shape_of_batch[0], 1])
        self._alpha = alpha

    def set_mix_ratio(self):
        # How to concatenate images
        def get_maskv(alpha):
            if alpha <= 0.0:
                return np.ones(self._maskv.shape), 1.0
            mask = np.zeros(self._maskv.shape)
            lam = np.random.beta(self._alpha, self._alpha)
            lh = np.int(lam * self._maskv.shape[2])
            mask[:, :, 0:lh, :] = 1.0
            return mask, lam

        def get_maskh(alpha):
            if alpha <= 0.0:
                return np.ones(self._maskh.shape), 1.0
            mask = np.zeros(self._maskh.shape)
            lam = np.random.beta(self._alpha, self._alpha)
            lw = np.int(lam * self._maskh.shape[3])
            mask[:, :, :, 0:lw] = 1.0
            return mask, lam

        self._maskv.d, lam1 = get_maskv(self._alpha)
        self._maskh.d, lam2 = get_maskh(self._alpha)
        if self._alpha > 0.0:
            self._lamx.d = np.random.beta(
                self._alpha, self._alpha, self._batch_size).reshape((self._batch_size, 1))
        else:
            self._lamx.d = np.ones((self._batch_size, 1))
        self._lamy.d = lam1 * self._lamx.d + lam2 * (1.0 - self._lamx.d)

    def mix_data(self, x, y):
        # Mix data given x and y, and return mix_x and mix_y
        # Both y and mix_y are supposed to be nn.Variable((batch_size, n_class))
        batch_size = x.shape[0]
        ind = np.random.permutation(batch_size)
        x0 = x
        y0 = y
        x1 = x0[ind]
        y1 = y0[ind]
        x_hcat = self._maskh * x0 + (1.0 - self._maskh) * x1
        x_vcat = self._maskv * x0 + (1.0 - self._maskv) * x1
        mix_x = self._lamx.reshape(
            (-1, 1, 1, 1)) * x_hcat + (1.0-self._lamx.reshape((-1, 1, 1, 1))) * x_vcat
        mix_y = self._lamy * y0 + (1.0-self._lamy) * y1
        return mix_x, mix_y

    def loss(self, pred, mix_y):
        # Calculate a classification loss given mix_y and prediction results of mix_x.
        # Both pred and mix_y are supposed to be a nn.Variable
        return softmax_cross_entropy_loss_vlabel(pred, mix_y)
