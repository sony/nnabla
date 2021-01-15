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

from libcpp.string cimport string
from _variable cimport function_hook_type
from solver cimport update_hook_type

cdef extern from "nbla/global_function_callback.hpp" namespace "nbla":
    void c_set_function_pre_hook "nbla::set_function_pre_hook" (const string & key, const function_hook_type& cb) nogil except+
    void c_set_function_post_hook "nbla::set_function_post_hook" (const string & key, const function_hook_type& cb) nogil except+
    void c_unset_function_pre_hook "nbla::unset_function_pre_hook" (const string & key) nogil except+
    void c_unset_function_post_hook "nbla::unset_function_post_hook" (const string & key) nogil except+


cdef extern from "nbla/global_solver_callback.hpp" namespace "nbla":
    void c_set_solver_pre_hook "nbla::set_solver_pre_hook" (const string & key, const update_hook_type& cb) nogil except+
    void c_set_solver_post_hook "nbla::set_solver_post_hook" (const string & key, const update_hook_type& cb) nogil except+
    void c_unset_solver_pre_hook "nbla::unset_solver_pre_hook" (const string & key) nogil except+
    void c_unset_solver_post_hook "nbla::unset_solver_post_hook" (const string & key) nogil except+

