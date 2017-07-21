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

"""
# Usage

```
digits = tiny_digits.load_digits()

```
"""

from __future__ import print_function

from nnabla.monitor import tile_images
from nnabla.utils.data_iterator import data_iterator_simple

import sys
import numpy as np

# Scikit
try:
    from sklearn.datasets import load_digits  # Only for dataset
except ImportError:
    print("Require scikit-learn", file=sys.stderr)
    raise

# Matplotlib
try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Require matplotlib", file=sys.stderr)
    raise

imshow_opt = dict(cmap='gray', interpolation='nearest')


def plot_stats(digits):
    print("Num images:", digits.images.shape[0])
    print("Image shape:", digits.images.shape[1:])
    print("Labels:", digits.target[:10])
    plt.imshow(tile_images(digits.images[:64, None]), **imshow_opt)


def data_iterator_tiny_digits(digits, batch_size=64, shuffle=False, rng=None):
    def load_func(index):
        """Loading an image and its label"""
        img = digits.images[index]
        label = digits.target[index]
        return img[None], np.array([label]).astype(np.int32)
    return data_iterator_simple(load_func, digits.target.shape[0], batch_size, shuffle, rng, with_file_cache=False)
