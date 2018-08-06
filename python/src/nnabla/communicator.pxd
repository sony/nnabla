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
from _variable cimport CVariable, CContext, dtypes, CommunicatorBackwardCallbackPtr
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
        void barrier() except +
        void abort() except +

        int size() except +
        int rank() except +
        int local_rank() except +

        string new_group(pair[string, vector[int]] name_ranks) except +
        unordered_map[string, vector[int]] list_groups() except +
        vector[int] find_group(const string & group) except +

        void reduce(const vector[shared_ptr[CNdArray]] & ndarray_list, int dst, cpp_bool division, cpp_bool inplace, const string & group) nogil except +
        void reduce(shared_ptr[CNdArray] data, int dst, cpp_bool division, cpp_bool inplace, const string & group) nogil except +
        void allreduce(cpp_bool division, cpp_bool inplace) nogil except +
        void all_reduce(const vector[shared_ptr[CNdArray]] & ndarray_list, cpp_bool division, cpp_bool inplace, const string & group) nogil except +
        void all_reduce(shared_ptr[CNdArray] data, cpp_bool division, cpp_bool inplace, const string & group) nogil except +
        CommunicatorBackwardCallbackPtr all_reduce_callback(const vector[shared_ptr[CNdArray]] & ndarray_list, size_t pack_size, cpp_bool division, const string & group) except +
        CommunicatorBackwardCallbackPtr all_reduce_callback(shared_ptr[CNdArray] data, size_t pack_size, cpp_bool division, const string & group) except +
        void reduce_scatter(const vector[shared_ptr[CNdArray]] & ndarray_list, shared_ptr[CNdArray] ndarray, cpp_bool division, const string & group) nogil except +
        void bcast(const vector[shared_ptr[CNdArray]] & ndarray_list, int src, cpp_bool inplace, const string & group) nogil except +
        void bcast(shared_ptr[CNdArray] ndarray, int src, cpp_bool inplace, const string & group) nogil except +
        void all_gather(shared_ptr[CNdArray] ndarray, const vector[shared_ptr[CNdArray]] & ndarray_list, const string & group) nogil except +

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
