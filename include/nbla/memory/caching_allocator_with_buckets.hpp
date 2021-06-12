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
#include <memory>
#include <tuple>
#include <unordered_map>

namespace nbla {

using std::map;
using std::unordered_map;
using std::tuple;
using std::make_shared;

/** A base class of CachingAllocatorWithBuckets.

    This implements a caching logic but it leaves an instantiation of Memory
    class as a virtual function
    CachingAllocatorWithBucketsBase::make_memory_impl. It
    enables an easy realization of this allocator with any Memory class
    implementation such as CpuMemory and CudaMemory. The caching algorithm is
    described as following.

    ## Caching a previously requested memory into a memory pool

    This allocator maintains a memory pool as a map from a requested memory
    configuration to a Memory instance previously created. A created memory
    block is re-used without allocation and free, which significantly reduces
    overhead due to memory allocation and deallocation, and implicit
    synchronization of device execution queues in CUDA for example.

    ## Size dependent memory pool

    A memory pool is maintained as two separate pools for small size and large
    size memory respectively. By default, memory size less than 1MB is
    considered as a small block, otherwise large.

    ## Rounding rules of memory size

    A requested memory size is rounded to a multiple of round_small_ (512B by
    default) or round_large_ (128KB by default) for small or large blocks
    respectively.

    ## Creation rules

    If any of previously created memory block larger than a requested size
    is not found, a new Memory instance is created. If found, a minimum size
    memory block is used after applying the following split rule.

    ## Split rules

    If the size of the found memory block is greater than or equal to
    round_small_ (512B by default) or small_alloc_ (1MB by default) + 1 for
    small or large respectively, the found memory block is split into two at an
    offset position by a requested size after rounding, then the second one is
    returned to the pool, and the first one is used.

    @sa CachingAllocatorWithBuckets
 */
class NBLA_API CachingAllocatorWithBucketsBase : public Allocator {
public:
  typedef tuple<size_t, Memory *> Key;
  typedef map<Key, shared_ptr<Memory>> DeviceCacheMap;
  typedef unordered_map<string, DeviceCacheMap> CacheMap;

private:
  CacheMap small_cache_map_;
  CacheMap large_cache_map_;
  MemCountMap small_memory_counter_;
  MemCountMap large_memory_counter_;
  static constexpr int round_small_ = 512;       // 512B
  static constexpr int round_large_ = 128 << 10; // 128KB
  static constexpr int small_alloc_ = 1 << 20;   // 1MB

  size_t round_size(size_t bytes) const;

  void free_impl(shared_ptr<Memory> memory) override;

  shared_ptr<Memory> alloc_impl(size_t orig_bytes,
                                const string &device_id) override;

  size_t free_unused_device_caches_impl(const string &device_id) override;

  void print_memory_cache_map_impl() override;

  size_t get_max_cache_bytes(const string &device_id);
  size_t get_total_cache_bytes(const string &device_id);

protected:
  /** Make Memory instance with a given configuration.

      This is called from alloc_impl if no re-usable memory is found in pool.
   */
  shared_ptr<Memory> make_memory(size_t bytes, const string &device_id);

  /** CachingAllocatorWithBuckets implements this with Memory class template.
   */
  virtual shared_ptr<Memory> make_memory_impl(size_t bytes,
                                              const string &device_id) = 0;

public:
  CachingAllocatorWithBucketsBase();

  size_t get_fragmentation_bytes(const string &device_id) override;

  size_t get_max_available_bytes(const string &device_id) override;

  vector<int> get_used_memory_counts(const string &device_id) override;
};

/** A realization of CachingAllocatorWithBuckets.

@tparam MemoryType A Memory class implementation such as CpuMemory and
CudaMemory.
@sa CachingAllocatorWithBucketsBase
@ingroup AllocatorImplGrp
*/
template <class MemoryType>
class CachingAllocatorWithBuckets : public CachingAllocatorWithBucketsBase {
  typedef MemoryType memory_type;
  shared_ptr<Memory> make_memory_impl(size_t bytes,
                                      const string &device_id) override {
    return make_shared<memory_type>(bytes, device_id);
  }

public:
  CachingAllocatorWithBuckets() : CachingAllocatorWithBucketsBase() {
#if 0
    this->callback_.reset(new PrintingAllocatorCallback(
        typeid(CachingAllocatorWithBuckets<memory_type>).name()));
#endif
  }
};
}
