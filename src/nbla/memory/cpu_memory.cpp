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
#include <nbla/memory/cpu_memory.hpp>

#include <memory>

#if 0
#include <cstdio>
#define DEBUG_LOG(...) printf(__VA_ARGS__);
#else
#define DEBUG_LOG(...)
#endif

namespace nbla {
using std::make_shared;
// ----------------------------------------------------------------------
// CpuMemory implementation
// ----------------------------------------------------------------------
CpuMemory::CpuMemory(size_t bytes, const string &device_id)
    : Memory(bytes, device_id) {}
CpuMemory::CpuMemory(size_t bytes, const string &device_id, void *ptr)
    : Memory(bytes, device_id) {
  ptr_ = ptr;
}

CpuMemory::~CpuMemory() {
  if (!ptr_) {
    return;
  }
  NBLA_FORCE_ASSERT(!prev(),
                    "Trying to free memory which has a prev (allocated "
                    "by another memory and split previously).");
  DEBUG_LOG("%s: %zu at %p\n", __func__, this->bytes(), ptr_);
  ::free(ptr_);
}

bool CpuMemory::alloc_impl() {
  ptr_ = ::malloc(this->bytes());
  DEBUG_LOG("%s: %zu at %p\n", __func__, this->bytes(), ptr_);
  return bool(ptr_);
}

shared_ptr<Memory> CpuMemory::divide_impl(size_t second_start) {
  /*
    Create a right sub-block which starts at second_start of this->ptr_. This
    instance doesn't have to be modified because it already points a start of a
    left sub-block.
   */
  size_t out_bytes = this->bytes() - second_start;
  void *out_ptr = (void *)((uint8_t *)ptr_ + second_start);
  return shared_ptr<Memory>(
      new CpuMemory(out_bytes, this->device_id(), out_ptr));
}

void CpuMemory::merge_next_impl(Memory *from) {
  /*
    this->ptr_ already points a start of a merged memory block.
   */
}

void CpuMemory::merge_prev_impl(Memory *from) {
  /*
    Use a start pointer as this->ptr_ which reprensents a start of a merged
    memory block.
   */
  ptr_ = from->pointer();
}
}
