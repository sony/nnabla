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

#include <nbla/memory/virtual_caching_allocator.hpp>

namespace nbla {
  inline VirtualCachingAllocatorBase::PhysicalMemoryCache &
  get_device_cache_map(unordered_map<string, VirtualCachingAllocatorBase::PhysicalMemoryCache> &m,
                       const string &device_id) {
    auto it = m.find(device_id);
    if (it == m.end()) {
      // Create a new DeviceCacheMap
      it = m.emplace(device_id, VirtualCachingAllocatorBase::PhysicalMemoryCache()).first;
    }
    return it->second;
  }

  void VirtualCachingAllocatorBase::set_chunk_size(size_t size, int ct_flag) {
    (ct_flag == int(chunk_type::SMALL) ? small_chunk_size_ : large_chunk_size_) = size;
  }

  //----------------------------------------------------------------------
  // Overriding member functions
  //----------------------------------------------------------------------

  void VirtualCachingAllocatorBase::free_impl(shared_ptr<Memory> memory) {
    // large or small
    // Currently assume either small or large physical memories are used for virtual memory.
    bool is_small = memory->get_physical_memory()[0]->bytes() < large_chunk_size_;
    auto &cache = is_small ? small_device_cache_ : large_device_cache_;

    // get device_cache
    auto &p_mem_cache = get_device_cache_map(cache, memory->device_id());

    // Return physical memories to cache.
    // todo: trace whether a physical memory is used or not by any virtual memory.
    // Currently both waiting_list and cache have shared_ptr<PhysicalMemory>
    // to prevent keep memory from being freed by free_unused_device_caches_impl.
    // This is just a workaround and not so efficient.
    for (auto &e: memory->get_physical_memory()) {
      p_mem_cache.push(e);
    }

    // Request to change device memory state
    memory->lock_device_memory();

    // Keep this memory in waiting_list
    waiting_list_.push(memory);
  }

  void VirtualCachingAllocatorBase::sync_waiting_list() {
    // Lazily unbind virtual memory to keep memory region safe in asynchronous computation.

    // todo: binary search?
    while (!waiting_list_.empty()) {
      auto &m = waiting_list_.front();
      if (m->get_device_memory_state() == DeviceMemoryState::Locked) break;
      m->unbind();
      waiting_list_.pop();
    }
  }

  void VirtualCachingAllocatorBase::alloc_physical_memory(size_t orig_bytes,
                                                          size_t chunk_size,
                                                          const string& device_id,
                                                          size_t &allocated_bytes,
                                                          std::vector<PhysicalMemoryPtr> &p_mems) {
    while (allocated_bytes < orig_bytes) {
      auto pm = create_physical_memory_impl(chunk_size, device_id);
      p_mems.push_back(pm);
      allocated_bytes += p_mems.back()->bytes();
    }
  }

  shared_ptr<Memory>
  VirtualCachingAllocatorBase::alloc_impl(size_t orig_bytes,
                                          const string &device_id) {
    /*
     * write me
     */
    sync_waiting_list();

    // Decide large or small depending on request bytes
    bool is_small = orig_bytes < large_chunk_size_;
    auto &cache = is_small ? small_device_cache_ : large_device_cache_;
    auto cs = is_small ? small_chunk_size_ : large_chunk_size_;
    auto &p_mem_cache = get_device_cache_map(cache, device_id);

#if 0
    print_memory_cache_map_impl()
#endif

    vector<PhysicalMemoryPtr> p_mems;

    // Get physical memories from cache.
    // todo: more efficient way.
    size_t p_mem_bytes = 0;
    while (!p_mem_cache.empty() && orig_bytes > p_mem_bytes) {
      auto p_mem = p_mem_cache.front();
      p_mem_cache.pop();

      p_mems.push_back(p_mem);
      p_mem_bytes += p_mem->bytes();
    }

    // note: p_mem_bytes is always a multiple of chunk_size_, since it's sum of a multiple of chunk_size_.
    // Retry allocation logic.
    try {
      // Additionally allocate physical memory if needed.
      alloc_physical_memory(orig_bytes, cs, device_id, p_mem_bytes, p_mems);

    } catch (...) {
      std::cout << "[VirtualCachingAllocatorBase] Failed to allocate physical memory. Free cache and try again." << std::endl;
      // If memory allocation is failed, it is possible that memory cache causes this problem.
      // So freeing all caches once might help to allocate a new memory.
      free_unused_caches();

      try {
        // Additionally allocate physical memory if needed.
        alloc_physical_memory(orig_bytes, cs, device_id, p_mem_bytes, p_mems);

      } catch (...) {
        std::cerr << "[VirtualCachingAllocatorBase] Failed to allocate physical memory again." << std::endl;
        throw;
      }
    }

    try {
      auto mem = create_virtual_memory_impl(p_mem_bytes, device_id, p_mems);
      mem->bind();
      return mem;
    } catch (...) {
      std::cerr << "[VirtualCachingAllocatorBase] Failed to map virtual memory." << std::endl;
      throw;
    }
  }

  size_t VirtualCachingAllocatorBase::free_unused_device_caches_impl(
          const string &device_id) {

    // VirtualCachingAllocator has 2 kinds of memory cache, large and small.
    // If various size of memories are requested time to time,
    // the number of caching memories can glow.
    // It is possible that Selection algorithm selects the one which is not cached and create new ones
    // in spite of selecting opposite ones even if a lot of memories are cached for the opposite.
    // In that case, this method could be helpful.

    // todo: Clear not all memories but partial ones?
    sync_waiting_list();

    auto &ms = small_device_cache_[device_id];
    auto &ml = large_device_cache_[device_id];

    // sum all memory bytes sizes up
    size_t freed = ms.size() * large_chunk_size_ + ml.size() * small_chunk_size_;

    // clear all
    PhysicalMemoryCache().swap(ms);
    PhysicalMemoryCache().swap(ml);

    return freed;
  }

  void VirtualCachingAllocatorBase::print_memory_cache_map_impl() {
    printf("[VirtualCachingAllocatorBase]");

    for (auto &p: small_device_cache_) {
      string device_id = p.first;
      printf("===== device_id: %s =====\n# waiting memory: %s\ncached bytes (small): %s\ncached bytes (large): %s\n",
              device_id.c_str(), waiting_list_.size(),
              byte_to_human_readable(small_device_cache_[device_id].size() * small_chunk_size_).c_str(),
              byte_to_human_readable(large_device_cache_[device_id].size() * large_chunk_size_).c_str()
      );
    }
  }
}
