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

#include <chrono>
#include <map>
#include <memory>
#include <queue>

#include <nbla/backend_registry.hpp>
#include <nbla/global_context.hpp>
#include <nbla/memory/allocator.hpp>

namespace nbla {
using std::queue;
using std::make_shared;
using std::multimap;

typedef std::chrono::system_clock::time_point Tp;

class NBLA_API VirtualCachingAllocatorBase : public Allocator {
public:
  // Types realted to memory cache
  typedef queue<PhysicalMemoryPtr> PhysicalMemoryCache;
  typedef unordered_map<string, PhysicalMemoryCache> PhysicalMemoryCacheMap;

  typedef pair<Tp, shared_ptr<Memory>> MemPtrWithTime;
  typedef multimap<size_t, MemPtrWithTime> MappedCache;
  typedef std::priority_queue<MemPtrWithTime, vector<MemPtrWithTime>,
                              std::greater<MemPtrWithTime>>
      Wlist;

  void set_chunk_size(size_t size);

  // Size of each single memory chunk.
  size_t chunk_size_ = 20ULL << 20; // 20MB

private:
  // Memory cache
  PhysicalMemoryCacheMap physical_memory_cache_;
  MemCountMap memory_counter_;
  unordered_map<string, long long> fragmentation_bytes_;

  // Waiting memory list to be cleared.
  Wlist waiting_list_;

  // Mapped deviceptr cache
  unordered_map<string, MappedCache> mapped_ptr_cache_;

  void sync_waiting_list();

  void alloc_physical_memory(size_t alloc_bytes, const string &device_id,
                             size_t &p_mem_bytes,
                             vector<PhysicalMemoryPtr> &p_mems);

  void alloc_physical_memory_with_retry(size_t alloc_bytes,
                                        const string &device_id,
                                        size_t &p_mem_bytes,
                                        vector<PhysicalMemoryPtr> &p_mems);

  void transfer_memory_from_cache(MemPtrWithTime &from,
                                  vector<PhysicalMemoryPtr> &to,
                                  size_t alloc_bytes, size_t &p_mem_bytes);

  void free_impl(shared_ptr<Memory> memory) override;

  shared_ptr<Memory> alloc_impl(size_t orig_bytes,
                                const string &device_id) override;

protected:
  /** CachingAllocatorWithBuckets implements this with Memory class template.
   */
  virtual PhysicalMemoryPtr
  create_physical_memory_impl(size_t bytes, const string &device_id) = 0;

  virtual shared_ptr<Memory>
  create_virtual_memory_impl(size_t bytes, const string &device_id,
                             const VecPhysicalMemoryPtr &p_memories) = 0;

  size_t free_unused_device_caches_impl(const string &device_id) override;

  void print_memory_cache_map_impl() override;

  size_t get_total_cache_bytes_impl(const string &device_id);

public:
  VirtualCachingAllocatorBase() = default;

  size_t get_fragmentation_bytes(const string &device_id) override;

  size_t get_max_available_bytes(const string &device_id) override;

  vector<int> get_used_memory_counts(const string &device_id) override;
};

template <class PhysicalMemoryType, class VirtualMemoryType>
class NBLA_API VirtualCachingAllocator : public VirtualCachingAllocatorBase {
  typedef PhysicalMemoryType p_memory_type;
  typedef VirtualMemoryType v_memory_type;

  PhysicalMemoryPtr
  create_physical_memory_impl(size_t bytes, const string &device_id) override {
    return make_shared<p_memory_type>(bytes, device_id);
  }

  shared_ptr<Memory>
  create_virtual_memory_impl(size_t bytes, const string &device_id,
                             const VecPhysicalMemoryPtr &p_memories) override {
    return make_shared<v_memory_type>(bytes, device_id, p_memories);
  }

public:
  VirtualCachingAllocator() : VirtualCachingAllocatorBase() {}
};
}
