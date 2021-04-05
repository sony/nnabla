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

from libcpp cimport bool as cpp_bool
from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr
from _common cimport *
from _array cimport *
from _context cimport *


cdef extern from "nbla/synced_array.hpp" namespace "nbla":

    cdef cppclass CSyncedArray "nbla::SyncedArray":
        CSyncedArray(Size_t size) except +
        CArray * cast(dtypes dtype, const CContext & ctx) nogil except+
        ArrayPtr cast_sp(dtypes dtype, const CContext & ctx) nogil except+
        const CArray * get(dtypes dtype, const CContext & ctx) nogil except+
        dtypes dtype() except+
        Size_t size() except+
        void zero() except+
        void fill(float value) except+
        void copy_from(const CSyncedArray *src) except+
        size_t modification_count() except+
        cpp_bool clear_called() except+
        cpp_bool zeroing() const
        void clear() except+

    ctypedef shared_ptr[CSyncedArray] SyncedArrayPtr


cdef extern from "nbla/nd_array.hpp" namespace "nbla":

    cdef cppclass CNdArray "nbla::NdArray":

        CNdArray(const Shape_t & shape) except+
        CNdArray(SyncedArrayPtr array, const Shape_t & shape) except+
        void reshape(const Shape_t & shape, cpp_bool force=false)except+
        shared_ptr[CNdArray] view(const Shape_t & shape) except+
        Shape_t shape() const
        Shape_t strides() const
        Size_t size(Size_t) const
        Size_t ndim() const
        SyncedArrayPtr array()
        void set_array(SyncedArrayPtr array) except+
        void zero() except+
        void fill(double v) except+
        const CArray * get(dtypes dtype, const CContext & ctx) nogil except+
        shared_ptr[const CArray] get_sp(dtypes dtype, const CContext & ctx) nogil except+
        unsigned long data_ptr(dtypes dtype, const CContext & ctx, cpp_bool write_only) nogil except+
        CArray * cast(dtypes dtype, const CContext & ctx, cpp_bool write_only) nogil except +
        ArrayPtr cast_sp(dtypes dtype, const CContext & ctx, cpp_bool write_only) nogil except +

    ctypedef shared_ptr[CNdArray] NdArrayPtr


cdef class NdArray:
    cdef NdArrayPtr arr
    cdef CNdArray * arrp

    @staticmethod
    cdef object create(NdArrayPtr arr)
