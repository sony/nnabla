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

#include <nbla/cpu_memory.hpp>
#include <nbla/garbage_collector.hpp>
#include <nbla/memory-internal.hpp>

#include <memory>
#include <vector>

namespace nbla {

using std::vector;
using std::shared_ptr;
using std::unique_ptr;
using std::make_shared;

/////////////////////////////
// CPU Memory implementation
/////////////////////////////
CpuMemory::CpuMemory(Size_t bytes, const string &device)
    : Memory(bytes, device) {}

CpuMemory::~CpuMemory() {
  if (!ptr_)
    return;
  ::free(ptr_);
}

bool CpuMemory::allocate() {
  if (ptr_)
    return true;
  ptr_ = ::malloc(size_);
  if (!ptr_) {
    // Garbage collection and retry to allocate.
    SingletonManager::get<GarbageCollector>()->collect();
    ptr_ = ::malloc(size_);
  }
  return ptr_;
}

template class MemoryCache<CpuMemory>;
} // End of namespace nbla
