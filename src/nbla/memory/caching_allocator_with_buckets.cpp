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

#include <nbla/memory/caching_allocator_with_buckets.hpp>

#if 0
#include <cstdio>
#define DEBUG_LOG(...) printf(__VA_ARGS__)
#define DEBUG_PRINT_CACHES(M, S) print_device_cache_map(M, S)
#else
#define DEBUG_LOG(...)
#define DEBUG_PRINT_CACHES(M, S)
#endif

namespace nbla {

//----------------------------------------------------------------------
// CTOR/DTOR
//----------------------------------------------------------------------
CachingAllocatorWithBucketsBase::CachingAllocatorWithBucketsBase() {}

//----------------------------------------------------------------------
// Inline local functions
//----------------------------------------------------------------------

inline CachingAllocatorWithBucketsBase::Key create_key_by_memory(Memory *mem) {
  return CachingAllocatorWithBucketsBase::Key{mem->bytes(), mem};
}

inline void try_erase_cache(
    CachingAllocatorWithBucketsBase::DeviceCacheMap &device_cache_map,
    Memory *mem) {
  /*
    Erase from cache if it memory instance has been mergd to another
   (i.e., disabled).
  */
  if (!mem || !mem->disabled()) {
    return;
  }
  auto key = create_key_by_memory(mem);
  device_cache_map.erase(key);
}

inline CachingAllocatorWithBucketsBase::DeviceCacheMap &
get_device_cache_map(CachingAllocatorWithBucketsBase::CacheMap &m,
                     const string &device_id) {
  auto it = m.find(device_id);
  if (it == m.end()) {
    // Create a new DeviceCacheMap
    it = m.emplace(device_id, CachingAllocatorWithBucketsBase::DeviceCacheMap())
             .first;
  }
  return it->second;
}

inline size_t free_unused_device_caches_in_map(
    CachingAllocatorWithBucketsBase::CacheMap &cache_map,
    const string &device_id) {
  auto &device_cache_map = get_device_cache_map(cache_map, device_id);
  auto it = device_cache_map.begin();
  auto end = device_cache_map.end();
  size_t total_freed_bytes{0};
  while (it != end) {
    size_t freed_bytes = it->second->bytes_active();
    if (freed_bytes) {
      total_freed_bytes += freed_bytes;
      // Memory will be freed at destructor of a Memory instance.
      it = device_cache_map.erase(it);
      continue;
    }
    ++it;
  }
  return total_freed_bytes;
}

//----------------------------------------------------------------------
//
//----------------------------------------------------------------------
size_t CachingAllocatorWithBucketsBase::round_size(size_t bytes) const {
  if (bytes < round_small_) {
    bytes = round_small_;
  } else if (bytes < small_alloc_) {
    bytes = ((bytes + round_small_ - 1) / round_small_) * round_small_;
  } else {
    bytes = ((bytes + round_large_ - 1) / round_large_) * round_large_;
  }
  return bytes;
}

static void
print_device_cache_map(const CachingAllocatorWithBucketsBase::DeviceCacheMap &m,
                       bool small) {
  std::vector<size_t> sizes;
  for (auto &i : m) {
    sizes.push_back(i.second->bytes());
  }
  printf("device_cache_map(%d): [%s]\n", (int)small,
         string_join(sizes, ", ").c_str());
}

//----------------------------------------------------------------------
// Overriding member functions
//----------------------------------------------------------------------
shared_ptr<Memory>
CachingAllocatorWithBucketsBase::alloc_impl(size_t orig_bytes,
                                            const string &device_id) {
  /*
    Find the minimum size memory with >bytes from a cache set which is
    dedicated for large or small memory size depending on the requested bytes.


    If memory is not found from cache, it will try to creat a new memory.

    The found/created memory is split if the remaining size is >= round_small_
    and small_alloc_ + 1 for small and large respectively.
   */

  // Rounding memory block size
  size_t bytes = round_size(orig_bytes);
  bool small = bytes <= small_alloc_;

  // Find a memory block from cache or create
  auto &cache_map = small ? small_cache_map_ : large_cache_map_;
  auto &device_cache_map = get_device_cache_map(cache_map, device_id);
  DEBUG_PRINT_CACHES(device_cache_map, small);
  Key key{bytes, nullptr};
  shared_ptr<Memory> mem;
  auto it = device_cache_map.lower_bound(key);
  if (it != device_cache_map.end()) {
    mem = it->second;
    device_cache_map.erase(it);
    DEBUG_LOG("Found: %zu\n", mem->bytes());
  } else {
    size_t alloc_bytes = small ? small_alloc_ : bytes;
    mem = this->make_memory(alloc_bytes, device_id);
    DEBUG_LOG("Alloc: %zu\n", alloc_bytes);
    alloc_retry(mem);
  }

  // Split obtained memory if it is too large.
  if (mem->bytes() - bytes >= (small ? round_small_ : small_alloc_ + 1)) {
    DEBUG_LOG("Split (%d): %zu at %zu\n", (int)small, mem->bytes(), bytes);
    shared_ptr<Memory> remaining = mem->divide(bytes);
    device_cache_map[create_key_by_memory(remaining.get())] = remaining;
  }
  DEBUG_PRINT_CACHES(device_cache_map, small);
  return mem;
}

shared_ptr<Memory>
CachingAllocatorWithBucketsBase::make_memory(size_t bytes,
                                             const string &device_id) {
  return this->make_memory_impl(bytes, device_id);
}

void CachingAllocatorWithBucketsBase::free_impl(shared_ptr<Memory> memory) {
  /*
    Return a given memory to cache set.

    The next and previous memory instances will be merged if they are not used
    (locked).
   */
  bool small = memory->bytes() <= small_alloc_;

  auto &cache_map = small ? small_cache_map_ : large_cache_map_;
  auto &device_cache_map = get_device_cache_map(cache_map, memory->device_id());

  // Try to merge prev and next
  Memory *n = memory->next();
  memory->try_merge(n);
  Memory *p = memory->prev();
  memory->try_merge(p);
  try_erase_cache(device_cache_map, n);
  try_erase_cache(device_cache_map, p);

  // Cache
  device_cache_map[create_key_by_memory(memory.get())] = memory;
  DEBUG_LOG("cache_impl\n");
  DEBUG_PRINT_CACHES(device_cache_map, small);
}

size_t CachingAllocatorWithBucketsBase::free_unused_device_caches_impl(
    const string &device_id) {
  /*
    Remove all unused caches from a specified device.
   */
  size_t freed_bytes{0};
  freed_bytes += free_unused_device_caches_in_map(small_cache_map_, device_id);
  freed_bytes += free_unused_device_caches_in_map(large_cache_map_, device_id);
  return freed_bytes;
}
}
