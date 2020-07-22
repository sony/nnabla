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

from .._context cimport CContext
from .._array cimport dtypes
from .._nd_array cimport CNdArray, NdArrayPtr

cdef extern from "nbla/utils/dlpack_utils.hpp":
    cdef struct CDLManagedTensor "DLManagedTensor":
        pass

cdef extern from "nbla/utils/dlpack_utils.hpp" namespace "nbla":
    NdArrayPtr from_dlpack(CDLManagedTensor *dltensor) nogil except+
    void from_dlpack(CDLManagedTensor *dltensor, CNdArray *a) nogil except+
    CDLManagedTensor* to_dlpack(CNdArray *a) nogil except+
    CDLManagedTensor* to_dlpack(CNdArray *a, const dtypes dtype, const CContext &ctx) nogil except+
    void call_deleter(CDLManagedTensor *dlp) nogil except+
