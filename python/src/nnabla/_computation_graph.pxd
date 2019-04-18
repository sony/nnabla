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
from libcpp cimport bool as cpp_bool
from _nd_array cimport *
from function cimport *
from _variable cimport *


cdef extern from "nbla/computation_graph/computation_graph.hpp" namespace "nbla":
    vector[CgVariablePtr] connect(
        CgFunctionPtr,
        vector[CgVariablePtr] & ,
        int,
        vector[NdArrayPtr],
        cpp_bool) except+
    void steal_variable_from_to(CgVariablePtr f, CgVariablePtr t) except+
    void forward_all(const vector[CgVariablePtr] &,
                     cpp_bool,
                     cpp_bool,
		     function_hook_type, function_hook_type) nogil except+
