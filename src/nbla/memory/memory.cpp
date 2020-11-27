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
#include <nbla/memory/memory.hpp>

#include <algorithm>
#include <typeinfo>

namespace nbla {

// ----------------------------------------------------------------------
// Memory interface
// ----------------------------------------------------------------------
Memory::Memory(size_t bytes, const string &device_id)
    : device_id_(device_id), bytes_(std::max(bytes, (size_t)1)),
      requested_bytes_(bytes_) {}

Memory::~Memory() {}

void Memory::alloc() {
  /*
    Check double allocation, and call derived class alloc_impl to allocate
    memory.
   */
  NBLA_CHECK(!ptr_, error_code::value, "Allocation called twice.");
  NBLA_CHECK(this->alloc_impl(), error_code::memory, "%s allocation failed.",
             typeid(*this).name());
}

size_t Memory::bytes_active() {
  /*
    By the following condition, we know that this is not a memory block
    originally allocated by alloc.
   */
  if (!ptr_ || next_ || prev_) {
    return 0;
  }
  return bytes_;
}

void Memory::associate_consecutive(Memory *left, Memory *right) {
  if (left) {
    left->next_ = right;
  }
  if (right) {
    right->prev_ = left;
  }
}

shared_ptr<Memory> Memory::divide(size_t second_start) {
  /*
    Split memory at second_start and return second one.

    This instance will be responsible for the memory up to second_start - 1.
   */
  NBLA_CHECK(second_start > 0, error_code::value,
             "`second_start` must be > 0. Given %zd.", second_start);
  NBLA_CHECK(second_start < bytes_, error_code::value,
             "`second_start` must be < bytes_. Given %zd was not < %zd.",
             second_start, bytes_);
  auto new_mem = divide_impl(second_start);
  bytes_ = second_start;
  associate_consecutive(new_mem.get(), this->next_);
  associate_consecutive(this, new_mem.get());
  return new_mem;
}

void Memory::try_merge(Memory *from) {
  /*
    Try to merge another memory instance.
    Merging if from is next or prev of this, and if not locked.
    The merged memory is set as disabled, and cannot be used any more.
   */
  if (!from || from->locked()) {
    /*
      memory is in use. This is set when CacheAllocator creates AllocatorMemory
      which wraps memroy instance.
    */
    return;
  }
  if (next_ == from) {
    this->merge_next_impl(from);
    associate_consecutive(this, from->next_);
  } else if (prev_ == from) {
    this->merge_prev_impl(prev_);
    associate_consecutive(from->prev_, this);
  }
  bytes_ += from->bytes_;
  from->disable();
}

void Memory::bind() {
  /*
    Bind physical memories (p_memories_) to virtual address, and make
    ptr_ available as a data pointer.
    This method can be executed only if memory_type_ == MemoryType::Virtual.
    Actual implementation should be implemented as bind_impl()
    in a derived class whose memory_type == MemoryType::Virtual.
   */
  NBLA_CHECK(memory_type_ == MemoryType::Virtual, error_code::memory,
             "This Memory instance is not Virtual Memory. Calling bind() is "
             "prohibited.");
  this->bind_impl();
}

void Memory::unbind() {
  /*
    Unbind physical memories (p_memories_) from virtual address, and
    make ptr_ disable.
    This method can be executed only if memory_type_ == MemoryType::Virtual.
    Actual implementation should be implemented as unbind_impl()
    in a derived class whose memory_type == MemoryType::Virtual.
   */
  NBLA_CHECK(memory_type_ == MemoryType::Virtual, error_code::memory,
             "This Memory instance is not Virtual Memory. Calling unbind() is "
             "prohibited.");

  this->unbind_impl();
  ptr_ = nullptr; // reset ptr as nullptr just in case.
}

bool Memory::grow(VecPhysicalMemoryPtr &p_mems) {
  /**
    Grow virtual memory.
    This method can be executed only if memory_type_ == MemoryType::Virtual.
    ctual implementation should be implemented as unbind_impl()
    in a derived class whose memory_type == MemoryType::Virtual.
   */

  NBLA_CHECK(memory_type_ == MemoryType::Virtual, error_code::memory,
             "This Memory instance is not Virtual Memory. Calling grow() is "
             "prohibited.");

  return this->grow_impl(p_mems);
}
}
