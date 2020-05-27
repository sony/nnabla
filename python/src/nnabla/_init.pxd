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
from libcpp.string cimport string
from libcpp cimport bool as cpp_bool


cdef extern from "nbla/init.hpp" namespace "nbla":
    void init_cpu() except +
    void clear_cpu_memory_cache() except+
    void print_cpu_memory_cache_map() except+
    vector[string] cpu_array_classes() except +
    void _cpu_set_array_classes(const vector[string] & a) except +
    void cpu_device_synchronize(const string & device) except +
    int cpu_get_device_count() except +
    vector[string] cpu_get_devices() except +

cdef extern from "nbla/garbage_collector.hpp" namespace "nbla":
    void register_gc "nbla::SingletonManager::get<nbla::GarbageCollector>()->register_collector" (void() nogil) except +

cdef extern from "nbla/cpu.hpp" namespace "nbla":
    vector[string] _cpu_array_classes "nbla::SingletonManager::get<nbla::Cpu>()->array_classes" () except +
    void _cpu_set_array_classes "nbla::SingletonManager::get<nbla::Cpu>()->_set_array_classes" (const vector[string] & a) except +


cdef extern from "nbla/singleton_manager.hpp" namespace "nbla":
    cdef cppclass SingletonManager:
        @staticmethod
        void clear() except +
