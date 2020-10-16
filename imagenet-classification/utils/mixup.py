# Copyright (c) 2020 Sony Corporation. All Rights Reserved.
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


def get_nnabla_version_integer():
    from nnabla import __version__
    import re
    r = list(map(int, re.match('^(\d+)\.(\d+)\.(\d+)', __version__).groups()))
    return r[0] * 10000 + r[1] * 100 + r[2]


class MixUp(object):

    def __init__(self, alpha, num_classes, rng=None):
        self.alpha = alpha
        self.num_classes = num_classes
        self.lam = None
        if rng is None:
            rng = np.random.RandomState(726)
        self.rng = rng

    def mix_data(self, image, label):
        '''
        Define mixed data Variables.

        Args:
            image(Variable): (B, C, H, W) or (B, H, W, C)
            label(Variable): (B, 1) of integers in [0, num_classes)

        Returns:
            image(Variable): mixed data
            label(Variable): mixed label with (B, num_clases)

        '''
        if image.shape[0] % 2 != 0:
            raise ValueError(
                'Please use an even number of batch size with this implementation of mixup regularization. Given {}.'.format(image.shape[0]))
        image2 = image[::-1]
        label = F.one_hot(label, (self.num_classes,))
        label2 = label[::-1]
        self.lam = nn.Variable((image.shape[0], 1, 1, 1))
        if get_nnabla_version_integer() < 10700:
            raise ValueError(
                'This does not work with nnabla version less than 1.7.0 due to [a bug](https://github.com/sony/nnabla/pull/608). Please update the nnabla version.')
        llam = F.reshape(self.lam, (-1, 1))
        self.reset_mixup_ratio()  # Call it for safe.
        mimage = self.lam * image + (1 - self.lam) * image2
        mlabel = llam * label + (1 - llam) * label2
        return mimage, mlabel

    def reset_mixup_ratio(self):
        assert self.lam is not None, 'mix_data() must be called before calling this method.'
        self.lam.d = self.rng.beta(self.alpha, self.alpha, size=self.lam.shape)
