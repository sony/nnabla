// Copyright 2019,2020,2021 Sony Corporation.
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

#pragma once

#include <nbla/memory/allocator.hpp>

#include <map>
#include <tuple>
#include <typeinfo>
#include <unordered_map>

namespace nbla {

/** Memory allocator wihout caching.

    This always creates a new instance of Memory with allocation
   (Memory::alloc), and frees when it's returned.

   @tparam MemoryType A Memory class implementation such as CpuMemory and
   CudaMemory.
 */
template <typename MemoryType> class NaiveAllocator : public Allocator {
public:
  typedef MemoryType memory_type;
  NaiveAllocator() {
#if 0
    callback_.reset(new PrintingAllocatorCallback(typeid(*this).name()));
#endif
  }

private:
  void free_impl(shared_ptr<Memory> memory) override {
    // Do nothing.
  }

  shared_ptr<Memory> alloc_impl(size_t orig_bytes,
                                const string &device_id) override {
    // Always create a newly allocated memory.
    auto mem = make_shared<memory_type>(orig_bytes, device_id);
    this->alloc_retry(mem);
    return mem;
  }

  size_t free_unused_device_caches_impl(const string &device_id) {
    // Do nothing.
    return 0;
  }
};
}
