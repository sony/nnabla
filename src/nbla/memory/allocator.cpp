// Copyright (c) 2017-2020 Sony Corporation. All Rights Reserved.
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

#include <iostream>
#include <nbla/memory/allocator.hpp>

namespace nbla {

Allocator::Allocator() {}
Allocator::~Allocator() {}

AllocatorMemory Allocator::alloc(size_t bytes, const string &device_id) {
  // Ensuring at least 1 byte. Workaround while knowing that it's in efficient.
  std::lock_guard<std::mutex> lock(mutex_);
  bytes = std::max(bytes, (size_t)1);
  auto mem = this->alloc_impl(bytes, device_id);
  device_memory_used_in_bytes_.insert(
      {device_id, (size_t)0}); // insert if not exists.
  if (callback_) {
    callback_->on_alloc(mem->bytes(), mem->device_id());
  }
  // NOTE: Allocator is always instantiated as a shared_ptr.
  return AllocatorMemory(mem, this->shared_from_this());
}
void Allocator::free(shared_ptr<Memory> memory) {
  std::lock_guard<std::mutex> lock(mutex_);
  memory->release();
  size_t bytes = memory->bytes();
  string device_id = memory->device_id();
  this->free_impl(memory);
  if (callback_) {
    callback_->on_free(bytes, device_id);
  }
}
size_t Allocator::free_unused_caches() {
  size_t freed_bytes = 0;
  for (auto &i : device_memory_used_in_bytes_) {
    freed_bytes += free_unused_device_caches(i.first);
  }
  return freed_bytes;
}
size_t Allocator::free_unused_device_caches(const string &device_id) {
  size_t freed_bytes = this->free_unused_device_caches_impl(device_id);
  device_memory_used_in_bytes_[device_id] -= freed_bytes;
  if (callback_) {
    callback_->on_free_unused_device_caches(device_id, freed_bytes);
  }
  return freed_bytes;
}

size_t Allocator::device_memory_used_in_bytes(const string &device_id) {
  auto it = device_memory_used_in_bytes_.find(device_id);
  if (it == device_memory_used_in_bytes_.end()) {
    return 0;
  }
  return it->second;
}

void Allocator::alloc_retry(shared_ptr<Memory> mem) {
  /*
    Try to allocate memory. If exception raised, free all available caches and
    retry to allocate.
   */
  try {
    mem->alloc();
  } catch (...) {
    // TODO: Move the following verbose to callback?
    std::cout << "Failed to allocate. Freeing memory cache and retrying."
              << std::endl;
    if (callback_) {
      callback_->on_allocation_failure();
    }
    this->free_unused_device_caches(mem->device_id());
    try {
      mem->alloc();
    } catch (...) {
      // TODO: Move the following verbose to callback?
      std::cerr << "Failed to allocate again." << std::endl;
      throw;
    }
  }
  device_memory_used_in_bytes_[mem->device_id()] += mem->bytes();
}

AllocatorMemory::AllocatorMemory(shared_ptr<Memory> memory,
                                 shared_ptr<Allocator> allocator)
    : memory_(memory), allocator_(allocator) {
  memory->lock();
}

AllocatorMemory::AllocatorMemory() : memory_(nullptr), allocator_(nullptr) {}

void AllocatorMemory::release() {
  if (!memory_) {
    return;
  }
  // memory_->release(); Move it to lock protected scope
  allocator_->free(memory_);
  memory_ = nullptr;
}

AllocatorMemory::~AllocatorMemory() { this->release(); }

AllocatorMemory::AllocatorMemory(AllocatorMemory &&rhs) {
  *this = std::move(rhs);
}

AllocatorMemory &AllocatorMemory::operator=(AllocatorMemory &&rhs) {
  this->release();
  memory_ = rhs.memory_;
  allocator_ = rhs.allocator_;
  rhs.memory_ = nullptr; // Avoid freeing.
  return *this;
}
}
