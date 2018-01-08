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

from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from libcpp.pair cimport pair
from libcpp.string cimport string
from libcpp.memory cimport shared_ptr
from libcpp cimport bool as cpp_bool
cimport _variable
from _variable cimport CVariable, CContext, dtypes
from _nd_array cimport CNdArray


cdef extern from "nbla/communicator.hpp" namespace "nbla":
    cdef cppclass CCommunicator "nbla::Communicator":
        string name() except +
        void add_context_and_parameters(
            const pair[CContext, vector[pair[string, shared_ptr[CVariable]]]] & ctx_params) except +

        void remove_context_parameters(
            const pair[CContext, vector[string]] & ctx_keys) except +
        void clear_context_parameters() except +

        void init() except +
        int size() except +
        int rank() except +
        int local_rank() except +

        void reduce(cpp_bool division) except +
        void allreduce(cpp_bool division, cpp_bool inplace) nogil except +
        void all_reduce(vector[shared_ptr[CNdArray]] ndarray_list, cpp_bool division, cpp_bool inplace) nogil except +
        void all_reduce(shared_ptr[CNdArray] data, cpp_bool division, cpp_bool inplace) nogil except +
        void reducescatter(cpp_bool division) nogil except +
        void bcast() nogil except +
        void allgather() nogil except +

        void reduce_async(cpp_bool division) nogil except +
        void allreduce_async(cpp_bool division, cpp_bool inplace) nogil except +
        void reducescatter_async(cpp_bool division) nogil except +
        void bcast_async() nogil except +
        void allgather_async() nogil except +

cdef extern from "nbla/communicator/data_parallel_communicator.hpp" namespace "nbla":
    shared_ptr[CCommunicator] create_DataParallelCommunicatorCommunicator(const CContext & ) except +

cdef extern from "nbla/communicator/multi_process_data_parallel_communicator.hpp" namespace "nbla":
    shared_ptr[CCommunicator] create_MultiProcessDataParallelCommunicatorCommunicator(const CContext & ) except +

cdef class Communicator:

    cdef shared_ptr[CCommunicator] communicator
    cdef CCommunicator * communicatorp

    @staticmethod
    cdef create(shared_ptr[CCommunicator] communicator)
