# Copyright 2017,2018,2019,2020,2021 Sony Corporation.
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


import pytest
import numpy as np
import math
from functools import partial

from nnabla.utils.data_source_loader import load_image
from nnabla.utils.data_iterator import data_iterator_simple

from .test_data_iterator import check_data_iterator_result
from .conftest import generate_cache_dir


@pytest.mark.parametrize("num_of_slices", [7])
@pytest.mark.parametrize("size", [55, 31])
@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("shuffle", [False])
@pytest.mark.parametrize("drop_last", [True, False])
def test_sliced_data_iterator_equivalence(test_data_csv_png_10, num_of_slices, size, batch_size, shuffle, drop_last):

    def lcm(a, b):
        return abs(a * b) / math.gcd(a, b) if a and b else 0

    max_epoch = lcm(batch_size, size) / size

    def test_load_func(position):
        return np.full((1), position, dtype=np.int32)

    def simple_load_func(data_set, position):
        return data_set[position]

    def get_data(iter_list, iter_num):
        total = 0
        for it in iter_list:
            for _ in range(iter_num):
                yield it.next()
                total += 1
            yield total
        yield total

    iter_num = int((max_epoch * size) / (num_of_slices * batch_size) + 0.5)

    sliced_di_list = []
    di = data_iterator_simple(test_load_func, size,
                              batch_size, shuffle=shuffle)

    for slice_pos in range(num_of_slices):
        sliced_di = di.slice(
            rng=None, num_of_slices=num_of_slices, slice_pos=slice_pos, drop_last=drop_last)
        sliced_di_list.append(sliced_di)

    ref_di_list = []
    all_data = [np.full((1), position, dtype=np.int32)
                for position in range(size)]
    slice_block_size = size // num_of_slices
    if not drop_last:
        slice_block_size += 1 if size % num_of_slices != 0 else 0
    for slice_pos in range(num_of_slices):
        start_index = slice_pos * slice_block_size
        end_index = start_index + slice_block_size
        if end_index > size:
            end_index = size
            start_index = end_index - slice_block_size
        sliced_data = all_data[start_index: end_index]
        di = data_iterator_simple(
            partial(simple_load_func, sliced_data), slice_block_size, batch_size, shuffle=shuffle)
        ref_di_list.append(di)

    set_a = set()
    set_b = set()
    for ref, t in zip(get_data(ref_di_list, iter_num), get_data(sliced_di_list, iter_num)):
        if isinstance(ref, tuple):
            ref, t = ref[0], t[0]
        if isinstance(ref, np.ndarray):
            # print(f"{ref} <--> {t}")
            set_a = set_a.union(set(ref))
            set_b = set_b.union(set(t))
        else:
            # print("-" * 30)
            assert ref == t
    # str_a = ','.join([str(f) for f in set_a])
    # str_b = ','.join([str(f) for f in set_b])
    # print(f"{str_a}  <--> {str_b}")
    assert set_a == set_b

    di_all = ref_di_list + sliced_di_list
    for di in di_all:
        di.close()


@pytest.mark.parametrize("num_of_slices", [2, 3, 5])
@pytest.mark.parametrize("size", [50])
@pytest.mark.parametrize("batch_size", [1, 5, 11])
@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("drop_last", [True])
def test_sliced_data_iterator_duplicated_unexpectedly(test_data_csv_png_10, num_of_slices, size, batch_size, shuffle, drop_last):
    # As the discussion in this: https://app.slack.com/client/TAVSDRN92/CCWT3ENA2/thread/CCWT3ENA2-1653356233.062649
    # Each slice might have overlapped items after the random order of each epoch is updated.
    # It only assumes that each slice at each epoch have different dataset.
    def test_load_func(position):
        return np.full((1), position, dtype=np.int32)

    di = data_iterator_simple(test_load_func, size,
                              batch_size, shuffle=shuffle)

    def lcm(a, b):
        return abs(a * b) / math.gcd(a, b) if a and b else 0

    max_epoch = int(lcm(batch_size, size) / size)

    epoch = []
    data_per_epoch = []

    slice_di_list = []
    for slice_pos in range(num_of_slices):
        sliced_di = di.slice(
            rng=None, num_of_slices=num_of_slices, slice_pos=slice_pos, drop_last=drop_last)
        slice_di_list.append(sliced_di)

    offset = 0
    for _ in range(max_epoch):
        data_per_batch = []
        for sliced_di in slice_di_list:
            batch = sliced_di.next()
            data_per_batch.append([d for dat in batch for d in dat])
            data_per_epoch.append(data_per_batch[-1][:])
        for i in range(len(data_per_batch)-1):
            for j in range(i + 1, len(data_per_batch)):
                ok = len(set(data_per_batch[i]) & set(data_per_batch[j])) == 0
                if not ok:
                    print("=" * 50)
                    for x in range(len(data_per_batch)):
                        print(
                            f"{len(epoch)}: {i} <--> {j}: {sorted(data_per_batch[x])}")
                    print(f"{set(data_per_batch[i]) & set(data_per_batch[j])}")
                    print("=" * 50)
                assert ok, "data overlapped in same epoch!"
        if offset + batch_size >= size:
            epoch.append(data_per_epoch)
            data_per_epoch = []
            offset += batch_size
            offset -= size
        else:
            offset += batch_size

    if shuffle:
        for i in range(len(epoch) - 1):
            for j in range(i + 1, len(epoch)):
                for k in range(len(epoch[i])):
                    ok = set(epoch[i][k]) != set(epoch[j][k])
                    if not ok:
                        print("-" * 50)
                        print(f"{k}: {i} {sorted(epoch[i][k])}")
                        print(f"{k}: {j} {sorted(epoch[j][k])}")
                        print("-" * 50)
                    assert ok, "Not allow duplicated data set occurs in shuffle mode."

    for sliced_di in slice_di_list:
        sliced_di.close()
    di.close()


@pytest.mark.parametrize("num_of_slices", [2, 3, 5])
@pytest.mark.parametrize("size", [197, 124])
@pytest.mark.parametrize("batch_size", [1, 20])
@pytest.mark.parametrize("shuffle", [False, True])
def test_sliced_data_iterator_race_condition(num_of_slices, size, batch_size, shuffle):
    from nnabla.utils.data_source_implements import CacheDataSource
    from nnabla.utils.data_iterator import data_iterator_cache

    with generate_cache_dir(size) as cache_dir:
        rng = np.random.RandomState(313)
        iterator = data_iterator_cache(cache_dir, batch_size, shuffle=True)
        sliced_it = iterator.slice(rng, num_of_slices, 1)

        for i in range(size + 5):
            d = sliced_it.next()
        sliced_it.close()
        iterator.close()
