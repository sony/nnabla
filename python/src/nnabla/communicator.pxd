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


cdef extern from "nbla/communicator.hpp" namespace "nbla":
    cdef cppclass CCommunicator "nbla::Communicator":
        string name() except +
        void add_context_and_parameters(
            const pair[CContext, vector[pair[string, shared_ptr[CVariable]]]] & ctx_params) except +

        void remove_context_parameters(
            const pair[CContext, vector[string]] &ctx_keys) except +
        void clear_context_parameters() except +
        
        void init() except +

        void reduce(cpp_bool division) except +
        void allreduce(cpp_bool division) except +
        void reducescatter(cpp_bool division) except +
        void bcast() except +
        void allgather() except +
        void ireduce(cpp_bool division) except +
        void iallreduce(cpp_bool division) except +
        void ireducescatter(cpp_bool division) except +
        void ibcast() except +
        void iallgather() except +
        
        void reduce_async(cpp_bool division) except +
        void allreduce_async(cpp_bool division) except +
        void reducescatter_async(cpp_bool division) except +
        void bcast_async() except +
        void allgather_async() except +
        void ireduce_async(cpp_bool division) except +
        void iallreduce_async(cpp_bool division) except +
        void ireducescatter_async(cpp_bool division) except +
        void ibcast_async() except +
        void iallgather_async() except +
        
cdef extern from "nbla/communicator/data_parallel_communicator.hpp" namespace "nbla":
    shared_ptr[CCommunicator] create_DataParallelCommunicatorCommunicator(const CContext &) except +        

cdef class Communicator:

    cdef shared_ptr[CCommunicator] communicator
    cdef CCommunicator * communicatorp

    @staticmethod
    cdef create(shared_ptr[CCommunicator] communicator)
    
    