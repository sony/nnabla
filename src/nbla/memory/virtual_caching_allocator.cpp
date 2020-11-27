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

using std::cout;
using std::endl;
using std::cerr;

inline VirtualCachingAllocatorBase::PhysicalMemoryCache &get_device_cache_map(
    unordered_map<string, VirtualCachingAllocatorBase::PhysicalMemoryCache> &m,
    const string &device_id) {
  auto it = m.find(device_id);
  if (it == m.end()) {
    // Create a new DeviceCacheMap
    it =
        m.emplace(device_id, VirtualCachingAllocatorBase::PhysicalMemoryCache())
            .first;
  }
  return it->second;
}

void VirtualCachingAllocatorBase::set_chunk_size(size_t size) {
  chunk_size_ = size;
}

//----------------------------------------------------------------------
// Overriding member functions
//----------------------------------------------------------------------

void VirtualCachingAllocatorBase::free_impl(shared_ptr<Memory> memory) {
  // Request to change device memory state
  memory->lock_device_memory();

  // get time_point
  auto tp = std::chrono::high_resolution_clock::now();

  // Keep this memory in mapped_ptr_cache_
  mapped_ptr_cache_[memory->device_id()].emplace(memory->bytes(),
                                                 make_pair(tp, memory));
}

void VirtualCachingAllocatorBase::sync_waiting_list() {
  // Lazily unbind virtual memory to keep memory region safe in asynchronous
  // computation.

  // todo: binary search?
  while (!waiting_list_.empty()) {
    auto &m = waiting_list_.top().second;
    if (m->get_device_memory_state() == DeviceMemoryState::Locked)
      break;

    string dev = m->device_id();

    // decrease memory counts
    memory_counter_[dev]--;

    fragmentation_bytes_[dev] -= m->bytes() - m->requested_bytes();

    m->unbind();
    waiting_list_.pop();
  }
}

void VirtualCachingAllocatorBase::alloc_physical_memory(
    size_t alloc_bytes, const string &device_id, size_t &p_mem_bytes,
    vector<PhysicalMemoryPtr> &p_mems) {
  while (alloc_bytes > p_mem_bytes) {
    auto pm = create_physical_memory_impl(chunk_size_, device_id);
    p_mems.push_back(pm);
    p_mem_bytes += p_mems.back()->bytes();
  }
}

void VirtualCachingAllocatorBase::alloc_physical_memory_with_retry(
    size_t alloc_bytes, const string &device_id, size_t &p_mem_bytes,
    std::vector<PhysicalMemoryPtr> &p_mems) {
  // Retry allocation logic.
  try {
    // Additionally allocate physical memory if needed.
    alloc_physical_memory(alloc_bytes, device_id, p_mem_bytes, p_mems);

  } catch (...) {
    std::cout << "[VirtualCachingAllocatorBase] Failed to allocate physical "
                 "memory. Free cache and try again."
              << std::endl;
    // If memory allocation is failed,
    // freeing all caches once might help to allocate a new memory.
    free_unused_caches();

    try {
      // Additionally allocate physical memory if needed.
      alloc_physical_memory(alloc_bytes, device_id, p_mem_bytes, p_mems);

    } catch (...) {
      std::cerr << "[VirtualCachingAllocatorBase] Failed to allocate physical "
                   "memory again."
                << std::endl;
      throw;
    }
  }
}

void VirtualCachingAllocatorBase::transfer_memory_from_cache(
    MemPtrWithTime &from, vector<PhysicalMemoryPtr> &ext_pmems,
    size_t alloc_bytes, size_t &p_mem_bytes) {
  // Unbind previous virtual address
  waiting_list_.push(from);

  auto mem = from.second;

  auto &p_mems = mem->get_physical_memory();
  string dev = mem->device_id();

  int i = 0;
  // Move physical memory
  for (; i < p_mems.size() && alloc_bytes > p_mem_bytes; ++i) {
    ext_pmems.emplace_back(p_mems[i]);
    p_mem_bytes += p_mems[i]->bytes();
  }

  // Move back unecessary memories to cache.
  for (; i < p_mems.size(); ++i) {
    physical_memory_cache_[dev].push(p_mems[i]);
  }
}

shared_ptr<Memory>
VirtualCachingAllocatorBase::alloc_impl(size_t orig_bytes,
                                        const string &device_id) {

  sync_waiting_list();

  // Round up to allocataion bytes
  size_t alloc_bytes =
      (orig_bytes + chunk_size_ - 1) / chunk_size_ * chunk_size_;
  auto &mapped_cache = mapped_ptr_cache_[device_id];
  auto &p_cache = physical_memory_cache_[device_id];

  // 1. There is no cache in memory. Allocate a new one.
  if (mapped_cache.size() == 0) {
    vector<PhysicalMemoryPtr> p_mems;

    size_t p_mem_bytes = 0;
    while (!p_cache.empty() && alloc_bytes > p_mem_bytes) {
      auto p_mem = p_cache.front();
      p_cache.pop();

      p_mems.push_back(p_mem);
      p_mem_bytes += p_mem->bytes();
    }

    try {
      alloc_physical_memory_with_retry(orig_bytes, device_id, p_mem_bytes,
                                       p_mems);

      // can allocate new physical memory
      auto mem = create_virtual_memory_impl(p_mem_bytes, device_id, p_mems);
      mem->bind();

      // [debug] increment memory counts
      memory_counter_[device_id]++;

      fragmentation_bytes_[device_id] += mem->bytes() - orig_bytes;
      mem->set_requested_bytes(orig_bytes);

      return mem;
    } catch (...) {
      NBLA_ERROR(error_code::memory,
                 "[VirtualCachingAllocatorBase] Failed to map virtual memory.")
    }
  }

  // 2. If exactly the same size memory exists, return it.
  auto it = mapped_cache.lower_bound(alloc_bytes);
  if (it != mapped_cache.end() && it->first == alloc_bytes) {
    // Just reuse memory.
    auto ret = it->second.second;
    mapped_cache.erase(it);
    return ret;
  }

  // there is no exact match
  // 3. There is at least one smaller memory than request. Can grow memory.
  if (it != mapped_cache.begin()) {
    // Use maximum memory among smaller memoryies than request as a base.

    // lower_bound()-- is the maximum smaller memory. Remove from cache to use.
    it--;
    auto ret = it->second.second;
    auto prev = it->second; // temporary
    mapped_cache.erase(it);

    size_t p_mem_bytes = ret->bytes();
    vector<PhysicalMemoryPtr> ext_p_mems;

    // Get physical memories from physical memory cache first.
    while (!p_cache.empty() && alloc_bytes > p_mem_bytes) {
      auto p_mem = p_cache.front();
      p_cache.pop();

      p_mem_bytes += p_mem->bytes();
      ext_p_mems.push_back(p_mem);
    }

    // When physical memories are additionally needed.
    if (alloc_bytes > p_mem_bytes) {
      auto it2 = mapped_cache.lower_bound(alloc_bytes - p_mem_bytes);

      if (it2 != mapped_cache.end()) {
        // There is a larger memory in chace than remaining bytes.
        transfer_memory_from_cache(it2->second, ext_p_mems, alloc_bytes,
                                   p_mem_bytes);

        mapped_cache.erase(it2);
      } else {
        // There is no larger memories in chace than remaining bytes.
        // Unbind from larger ones.
        auto it3 = mapped_cache.rbegin();
        for (; it3 != mapped_cache.rend() && alloc_bytes > p_mem_bytes; it3++) {
          transfer_memory_from_cache(it3->second, ext_p_mems, alloc_bytes,
                                     p_mem_bytes);
        }
        mapped_cache.erase(it3.base(), mapped_cache.end());

        // create new physical memory
        alloc_physical_memory(alloc_bytes, device_id, p_mem_bytes, ext_p_mems);
      }
    }

    // grow memory
    size_t prev_bytes = ret->bytes();
    bool status = ret->grow(ext_p_mems);

    if (status) {
      // Success.
      // In this case, previous virtual memory is just moved to new one.
      // Thus update debug info here.
      memory_counter_[device_id]--;
      fragmentation_bytes_[device_id] -= prev_bytes - ret->requested_bytes();

      ret->append_physical_memories(ext_p_mems);
    } else {
      // Fail, fall back to slower impl.

      // Previous virtual address is not used any more.
      waiting_list_.push(prev); // pair<cpu_time, ret>

      // create new physical memory vector
      auto p_mems_base = ret->get_physical_memory();
      vector<PhysicalMemoryPtr> p_mems_new;
      p_mems_new.reserve(p_mems_base.size() + ext_p_mems.size());
      p_mems_new.insert(p_mems_new.end(), p_mems_base.begin(),
                        p_mems_base.end());
      p_mems_new.insert(p_mems_new.end(), ext_p_mems.begin(), ext_p_mems.end());

      // create new virtual memory
      ret = create_virtual_memory_impl(p_mem_bytes, device_id, p_mems_new);
      ret->bind();
    }

    // debug info for the new memroy.
    memory_counter_[device_id]++;
    fragmentation_bytes_[device_id] += alloc_bytes - orig_bytes;

    return ret;
  }

  // 4. There is no smaller memory than request.
  // Unmap minimum memory among larger memories than request.
  auto prev = it->second;
  vector<PhysicalMemoryPtr> p_mems;
  size_t p_mem_bytes = 0;

  transfer_memory_from_cache(prev, p_mems, alloc_bytes, p_mem_bytes);
  mapped_cache.erase(it);

  auto ret = create_virtual_memory_impl(alloc_bytes, device_id, p_mems);

  ret->bind();

  // increment memory counts
  memory_counter_[device_id]++;

  fragmentation_bytes_[device_id] += alloc_bytes - orig_bytes;
  ret->set_requested_bytes(orig_bytes);

  return ret;
}

size_t VirtualCachingAllocatorBase::free_unused_device_caches_impl(
    const string &device_id) {

  // Try to free all caches.
  // todo: Clear not all memories but partial ones?
  auto &mapped_cache = mapped_ptr_cache_[device_id];
  for (auto &p : mapped_cache) {
    waiting_list_.push(p.second);
  }

  MappedCache().swap(mapped_cache);

  sync_waiting_list();

  // Try to free all cached physical memories.
  auto &ms = physical_memory_cache_[device_id];

  // sum all memory bytes sizes up.
  size_t freed = ms.size() * chunk_size_;

  // clear all.
  PhysicalMemoryCache().swap(ms);

  return freed;
}

void VirtualCachingAllocatorBase::print_memory_cache_map_impl() {
  for (auto &p : physical_memory_cache_) {
    string device_id = p.first;
    string bytes = byte_to_human_readable(p.second.size() * chunk_size_);
    printf("===== device_id: %s =====\n"
           " waiting memory: %lu\n"
           " cached bytes : %s\n",
           device_id.c_str(), waiting_list_.size(), bytes.c_str());
  }
}

size_t VirtualCachingAllocatorBase::get_total_cache_bytes_impl(
    const string &device_id) {
  size_t ret = 0;

  // physical memory cache
  auto p_cache = physical_memory_cache_[device_id];
  if (p_cache.empty())
    ret += p_cache.size() * chunk_size_;

  // mapped cache
  for (auto &p : mapped_ptr_cache_[device_id]) {
    ret += p.first;
  }

  return ret;
}

size_t
VirtualCachingAllocatorBase::get_fragmentation_bytes(const string &device_id) {
  return fragmentation_bytes_[device_id];
}

size_t
VirtualCachingAllocatorBase::get_max_available_bytes(const string &device_id) {
  return get_total_cache_bytes_impl(device_id);
}

vector<int>
VirtualCachingAllocatorBase::get_used_memory_counts(const string &device_id) {
  return {memory_counter_[device_id]};
}
}
