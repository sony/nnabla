// Copyright (c) 2020 Sony Corporation. All Rights Reserved.
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

#include <queue>
#include <memory>

#include <nbla/memory/allocator.hpp>
#include <nbla/global_context.hpp>
#include <nbla/backend_registry.hpp>

namespace nbla {
  using std::queue;
  using std::make_shared;

  class NBLA_API VirtualCachingAllocatorBase : public Allocator {
  public:
    // Types realted to memory cache
    typedef queue<PhysicalMemoryPtr> PhysicalMemoryCache;
    typedef unordered_map<string, PhysicalMemoryCache> CacheMap;

  private:
    // Memory cache
    CacheMap small_device_cache_;
    CacheMap large_device_cache_;

    // Size of each single memory chunk.
    static constexpr size_t small_chunk_size_ = 2ULL << 20; // 2MB
    static constexpr size_t large_chunk_size_ = 20ULL << 20; // 20MB

    // Waiting memory list to be cleared.
    queue<shared_ptr<Memory>> waiting_list_ = {};

    void sync_waiting_list();

    void alloc_physical_memory(size_t orig_bytes,
                               size_t chunk_size,
                               const string& device_id,
                               size_t& allocated_bytes,
                               vector<PhysicalMemoryPtr>& p_mems);

    void free_impl(shared_ptr<Memory> memory) override;

    shared_ptr<Memory> alloc_impl(size_t orig_bytes,
                                  const string &device_id) override;

  protected:
    /** CachingAllocatorWithBuckets implements this with Memory class template.
     */
    virtual PhysicalMemoryPtr create_physical_memory_impl(size_t bytes,
                                                          const string &device_id) = 0;

    virtual shared_ptr<Memory> create_virtual_memory_impl(size_t bytes, const string &device_id,
                                                          const VecPhysicalMemoryPtr &p_memories) = 0;

    size_t free_unused_device_caches_impl(const string &device_id) override;

    void print_memory_cache_map_impl() override;

  public:
    VirtualCachingAllocatorBase() {};
  };

  template<class PhysicalMemoryType, class VirtualMemoryType>
  class NBLA_API VirtualCachingAllocator : public VirtualCachingAllocatorBase {
    typedef PhysicalMemoryType p_memory_type;
    typedef VirtualMemoryType v_memory_type;

    PhysicalMemoryPtr create_physical_memory_impl(size_t bytes,
                                                  const string &device_id) override {
      return make_shared<p_memory_type>(bytes, device_id);
    }

    shared_ptr<Memory> create_virtual_memory_impl(size_t bytes,
                                                  const string &device_id,
                                                  const VecPhysicalMemoryPtr &p_memories) override {
      return make_shared<v_memory_type>(bytes, device_id, p_memories);
    }

  public:
    VirtualCachingAllocator()
    : VirtualCachingAllocatorBase() {}
  };
}
