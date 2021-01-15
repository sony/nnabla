# Copyright (c) 2020 Sony Corporation. All Rights Reserved.
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
from libcpp.memory cimport shared_ptr
from nnabla._context cimport CContext
from nnabla.function cimport CgFunctionPtr


cdef extern from "nbla/lms/swap_in_out_scheduler.hpp" namespace "nbla":
    cdef cppclass CSwapInOutScheduler "nbla::SwapInOutScheduler":
        CSwapInOutScheduler(const CContext &h_ctx, const CContext &d_ctx, 
                            const size_t s, const size_t l,
                            const cpp_bool save_host_mem,
                            const cpp_bool save_host_mem_no_abort) except +
        void start_scheduling() except +
        void end_scheduling() except +
        void pre_function_callback(const CgFunctionPtr &ptr) except +
        void post_function_callback(const CgFunctionPtr &ptr) except +
        void pre_update_callback() except +
        void post_update_callback() except +


cdef class SwapInOutScheduler:
    cdef shared_ptr[CSwapInOutScheduler] scheduler

