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
from libcpp.pair cimport pair
from libcpp cimport bool as cpp_bool

cdef extern from "nbla/computation_graph/variable.hpp" namespace "nbla":
    void c_activate_clear_called_flag_recorder() except +
    void c_deactivate_clear_called_flag_recorder() except +
    vector[vector[pair[cpp_bool, cpp_bool]]] c_get_input_clear_called_flags() except +
    vector[vector[pair[cpp_bool, cpp_bool]]] c_get_output_clear_called_flags() except +
