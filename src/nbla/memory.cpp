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

#include <nbla/memory.hpp>

namespace nbla {
Memory::Memory(Size_t bytes, const string &device)
    : size_(bytes), ptr_(nullptr), device_(device) {}

Size_t Memory::size() const { return size_; }

string Memory::device() const { return device_; }

void *Memory::ptr() {
  if (!ptr_) {
    NBLA_CHECK(allocate(), error_code::memory,
               "Failed to allocate %d bytes of memory on %s.", size_,
               device_.c_str());
  }
  return ptr_;
}
}
