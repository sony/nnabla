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

#ifndef __NBLA_MEMORY_HPP__
#define __NBLA_MEMORY_HPP__
#include <nbla/common.hpp>

#include <memory>
#include <mutex>
#include <unordered_map>

namespace nbla {

using std::shared_ptr;
using std::make_shared;

/** Memory interface class.
 */
class NBLA_API Memory {
protected:
  Size_t size_;
  void *ptr_;
  string device_;

public:
  /** Constructor
      @param size Size of memory allocated in bytes.
      @param device Device ID.
   */
  Memory(Size_t bytes, const string &device);

  /** Allocate memory.

      @return Return false if it fails to allocate memory.
   */
  virtual bool allocate() = 0;

  /// Size
  Size_t size() const;

  /// Device ID
  string device() const;

  /** Returns a void pointer to device memory.

      Note: This will implicitly call allocate() if pointer is not ready, and
     raise if allocation failed.
   */
  void *ptr();
  DISABLE_COPY_AND_ASSIGN(Memory);
};

/** Template class Memory cache.

    @tparam M Memory class which has the same interface with Memory class.
 */
template <class M> class MemoryCache {
  mutable std::mutex mtx_;

public:
  typedef M memory_type; ///< Memory class.
  /* Each vector element has memory pool on a specific CUDA device which ID
     corresponds the index in vector. */
  typedef std::unordered_multimap<Size_t, shared_ptr<memory_type>> cache_type;

private:
  int blocksize_; // Minimum unit of cache match.
  std::unordered_map<string, cache_type> pools_;

public:
  /** Pop a cached memory from pool or create.

     @param bytes Memory size in bytes.
     @param device Device ID.
     @return Shared pointer holding a memory instance
   */
  shared_ptr<memory_type> pop_or_create(Size_t bytes, const string &device);

  /** Cache a memory instance back to pool.

      @param mem Shared memory of a memory instance.
   */
  void cache(shared_ptr<memory_type> mem);

  /** Count the number of cached memories in pool.

      @param device_id DeviceID.
   */
  size_t count(const string &device_id) const;

  /** Clear memory for all device.
   */
  void clear();

  /** Clear memory caches in the specified device.

      @param device_id Device ID.
   */
  void clear(const string &device_id);

  /** Constructor.

      @param blocksize Unit size of memory block. Memory will be allocated
     with
                       lowerbound of a multiple of the blocksize.
   */
  MemoryCache(int blocksize = 512);
  ~MemoryCache();
  DISABLE_COPY_AND_ASSIGN(MemoryCache);
};
}

#endif
