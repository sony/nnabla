# Copyright 2017,2018,2019,2020,2021 Sony Corporation.
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

from libcpp.memory cimport shared_ptr

cdef extern from "nbla/array.hpp" namespace "nbla":
    cdef cppclass dtypes:
        pass

    cdef cppclass CArray "nbla::Array":
        void * pointer "nbla::Array::pointer<void>" ()
        const void * const_pointer "nbla::Array::const_pointer<const void>" () const

    ctypedef shared_ptr[CArray] ArrayPtr
    ctypedef const CArray ConstArray
    ctypedef shared_ptr[ConstArray] ConstArrayPtr

cdef class Array:
    cdef shared_ptr[const CArray] arr

    @staticmethod
    cdef object create(shared_ptr[const CArray] arr)
