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


from __future__ import absolute_import
from six.moves import range

import nnabla as nn
import nnabla.functions as F


def recon_loss(x, y):
    return F.mean(F.absolute_error(x, y))


def lsgan_loss(feat, target_is_real=True, persistent=True):
    if target_is_real:
        label = F.constant(1, shape=feat.shape)
    else:
        label = F.constant(0, shape=feat.shape)
    loss = F.mean(F.pow_scalar(feat - label, 2.0))
    loss.persistent = persistent
    return loss


def context_preserving_loss(xa, yb):

    def mask_weight(a, b):
        # much different from definition in the paper
        merged_mask = F.concatenate(a, b, axis=1)
        summed_mask = F.sum((merged_mask + 1) / 2, axis=1, keepdims=True)
        clipped = F.clip_by_value(summed_mask,
                                  F.constant(0, shape=summed_mask.shape),
                                  F.constant(1, shape=summed_mask.shape))
        z = clipped*2 - 1
        mask = (1 - z) / 2
        return mask

    x = xa[:, :3, :, :]
    a = xa[:, 3:, :, :]
    y = yb[:, :3, :, :]
    b = yb[:, 3:, :, :]

    assert x.shape == y.shape and a.shape == b.shape
    W = mask_weight(a, b)
    return F.mean(F.mul2(F.absolute_error(x, y), W))
