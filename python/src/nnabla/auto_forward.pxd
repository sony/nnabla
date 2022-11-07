# Copyright 2022 Sony Group Corporation.
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


cdef extern from "nbla/auto_forward.hpp" namespace "nbla":
    cpp_bool c_get_auto_forward "nbla::SingletonManager::get<nbla::AutoForward>()->get_auto_forward" () except +
    void c_set_auto_forward "nbla::SingletonManager::get<nbla::AutoForward>()->set_auto_forward" (const cpp_bool autoforward) except +
