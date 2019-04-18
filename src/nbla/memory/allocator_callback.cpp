// Copyright (c) 2017 Sony Corporation. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <nbla/exception.hpp>
#include <nbla/memory/allocator_callback.hpp>

#include <iostream>
#include <typeinfo>

namespace nbla {

PrintingAllocatorCallback::PrintingAllocatorCallback(const string &name)
    : name_(name) {}
void PrintingAllocatorCallback::on_alloc(size_t bytes,
                                         const string &device_id) {
  std::cout << format_string(
                   "PrintingAllocatorCallback::on_alloc(%s, %s) in %s.",
                   std::to_string(bytes).c_str(), device_id.c_str(),
                   name_.c_str())
            << std::endl;
}
void PrintingAllocatorCallback::on_free(size_t bytes, const string &device_id) {
  std::cout << format_string(
                   "PrintingAllocatorCallback::on_free(%s, %s) in %s.",
                   std::to_string(bytes).c_str(), device_id.c_str(),
                   name_.c_str())
            << std::endl;
}
void PrintingAllocatorCallback::on_free_unused_device_caches(
    const string &device_id, size_t freed_bytes) {
  std::cout << format_string("PrintingAllocatorCallback::on_free_unused_device_"
                             "caches(%s, %s) in %s.",
                             device_id.c_str(),
                             std::to_string(freed_bytes).c_str(), name_.c_str())
            << std::endl;
}
void PrintingAllocatorCallback::on_allocation_failure() {
  std::cout << format_string(
                   "PrintingAllocatorCallback::on_allocation_failure() in %s.",
                   name_.c_str())
            << std::endl;
}
}
