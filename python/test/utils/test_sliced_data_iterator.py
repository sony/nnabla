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


import pytest
import numpy as np

from nnabla.utils.data_source_loader import load_image
from nnabla.utils.data_iterator import data_iterator_simple

from .test_data_iterator import check_data_iterator_result


@pytest.mark.parametrize("num_of_slices", [2, 3, 5])
@pytest.mark.parametrize("size", [50])
@pytest.mark.parametrize("batch_size", [1, 5, 11])
@pytest.mark.parametrize("shuffle", [False, True])
def test_sliced_data_iterator(test_data_csv_png_10, num_of_slices, size, batch_size, shuffle):

    def test_load_func(position):
        return np.full((1), position, dtype=np.float32)

    di = data_iterator_simple(test_load_func, size,
                              batch_size, shuffle=shuffle)

    import six
    if six.PY2:
        from fractions import gcd
    else:
        from math import gcd

    def lcm(a, b):
        return abs(a * b) / gcd(a, b) if a and b else 0

    max_epoch = lcm(batch_size, size) / size

    all_data = []
    for slice_pos in range(num_of_slices):
        sliced_di = di.slice(
            rng=None, num_of_slices=num_of_slices, slice_pos=slice_pos)
        sliced_data = {}
        while True:
            current_epoch = sliced_di.epoch
            if current_epoch > max_epoch + 1:
                break
            data = sliced_di.next()
            if current_epoch not in sliced_data:
                sliced_data[current_epoch] = []
            for dat in data:
                for d in dat:
                    sliced_data[current_epoch].append(d)
        all_data.append(sliced_data)

    epochs = {}
    for slice_pos, sliced_data in enumerate(all_data):
        for epoch in sorted(sliced_data.keys()):
            if epoch not in epochs:
                epochs[epoch] = []
            epochs[epoch].append(set(sliced_data[epoch]))

    for epoch in sorted(epochs.keys()):
        x0 = epochs[epoch][0]
        acceptable_size = batch_size
        amount = size // num_of_slices
        if acceptable_size < amount:
            acceptable_size = amount
        for dup in [x0 & x for x in epochs[epoch][1:]]:
            assert len(dup) < amount
