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

from __future__ import absolute_import

from nnabla.utils.data_source import SlicedDataSource


def get_slice_start_end(size, n_slices, rank):
    _size = size // n_slices
    amount = size % n_slices
    slice_start = _size * rank
    if rank < amount:
        slice_start += rank
    else:
        slice_start += amount

    slice_end = slice_start + _size
    if slice_end > size:
        slice_start -= (slice_end - size)
        slice_end = size

    return slice_start, slice_end


def _get_sliced_data_source(ds, comm, shuffle=True):
    if comm is not None and comm.n_procs > 1:
        start, end = get_slice_start_end(ds._size, comm.n_procs, comm.rank)
        ds = SlicedDataSource(ds, shuffle=shuffle,
                              slice_start=start, slice_end=end)

    return ds
