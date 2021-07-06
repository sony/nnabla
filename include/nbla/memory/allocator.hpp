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

#include <nbla/common.hpp>
#include <nbla/memory/allocator_callback.hpp>
#include <nbla/memory/memory.hpp>

#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace nbla {

using std::unordered_map;
using std::unique_ptr;

// Forward declaration
class Allocator;

/** \addtogroup NNablaCoreGrp */
/*@{*/
/**
A RAII class for Memory allocated by a Allocator class.

At a destructor, the borrowed Memory will be returned to a Allocator
instance registered at a constructor.

 */
class AllocatorMemory {
  shared_ptr<Memory> memory_;

  /**
     The allocator used to allocate memory_ must be retained as a shared_ptr,
     otherwise allocator_ may be destroyed before destroying memory_, which
     results in accessing a destructed allocator.
   */
  shared_ptr<Allocator> allocator_;

  void release();

public:
  /** Get a Memory intance owned by this.
   */

  inline shared_ptr<Memory> memory() { return memory_; }
  /** Get bytes.

      @see Memory::bytes
   */
  inline size_t bytes() const { return memory_->bytes(); }

  /** Get device ID.

      @see Memory::device_id
   */
  inline string device_id() const { return memory_->device_id(); }

  /** Get a raw pointer retaining a memory block defined in a Memory
      implementation class.
   */
  inline void *pointer() { return memory_->pointer(); }
  /** Get as const.

      @see pointer
   */
  inline const void *const_pointer() const { return memory_->const_pointer(); }

  /** Constructor.

      This is called from Allocator::alloc to wrap a Memory instance.

      @param[in] memory A Memory instance wrapped.
      @param[in] allocator Used to return the wrapped memory to it
                          when this instance is destroyed.
   */
  NBLA_API AllocatorMemory(shared_ptr<Memory> memory,
                           shared_ptr<Allocator> allocator);

  /** Constructor.

      This is called to create an epmpty instance.
  */
  NBLA_API AllocatorMemory();

  /** Destructor returns a Memory instance given at a constructor to an
      allocator.
   */
  NBLA_API ~AllocatorMemory();

  // Move constructor/operator
  NBLA_API AllocatorMemory(AllocatorMemory &&rhs);
  NBLA_API AllocatorMemory &operator=(AllocatorMemory &&rhs);

  // Disable copy and assign.
  AllocatorMemory(const AllocatorMemory &) = delete;
  void operator=(const AllocatorMemory &) = delete;
};

/** Allocator interface class.

    A derived class implements logics that manage cached device memory blocks.

    A derived class NaiveAllocator which takes a Memory class as a
    template argument implements an memory allocator without caching.

    @note A Allocator instance must be instantiated as a shared_ptr.
    @todo Support streams or events.
 */
class NBLA_API Allocator : public std::enable_shared_from_this<Allocator> {
protected:
  unique_ptr<AllocatorCallback>
      callback_; ///< Callback could be set in a derived class.
  unordered_map<string, size_t> device_memory_used_in_bytes_;

  std::mutex mutex_;

public:
  typedef unordered_map<string, int> MemCountMap;

  std::function<void(void)> callback_tmp_ = nullptr;

  /** Constructor does nothing.
   */
  Allocator();

  // API
  /** Request a memory block wrapped by AllocatorMemory.

      A derived class must implement alloc_impl where a memory is newly created
     or
      obtained from memory pool.

      @param[in] bytes Number of bytes of memory block requested.
      @param[in] device_id Device specifier string. A format of allow specifier
                           is determined depending on Memory class
                           implementation.
      @return A memory block wrapped by AllocatorMemory. It's assumed that
     compiler
              RVO is enabled to prevent copying a return value. The returned
              AllocatorMemory must be moved to Array instance using std::move.

   */
  AllocatorMemory alloc(size_t bytes, const string &device_id);

  /** User should call this. This is called from destructor of AllocatorMemory.

      A given memory block is returned to this allocator instance. A derived
      class must implement Allocator::free_impl where a memory is
      returned to pool or simply freed if it's non-caching allocator.
   */
  void free(shared_ptr<Memory> memory);

  /** Free all unused memory blocks from all devices.

      Allocator::free_unused_device_caches is called for all device IDs
      previously passed to Allocator::alloc function.
   */
  size_t free_unused_caches();

  /** Free all unused memory blocks in a specified device.

      A derived class must implement
      Allocator::free_unused_device_caches_impl where all
      unused memory blocks that are origianally created, i.e. not divided, are
      discarded from pool.

      @param[in] device_id Specifies device by a string.
      @return Size of freed memory in bytes.
   */
  size_t free_unused_device_caches(const string &device_id);

  /** Get currently allocated memory size in a specifie device.

      @param[in] device_id Specifies device by a string.
      @return Total size of memory currently allocated by this allocator in
     bytes.
   */
  size_t device_memory_used_in_bytes(const string &device_id);

  /** Destructor does nothing.
   */
  virtual ~Allocator();

  /** APIs for memory cache analysis
   */
  void print_memory_cache_map() { print_memory_cache_map_impl(); }

  virtual size_t get_fragmentation_bytes(const string &device_id) { return 0; }

  virtual size_t get_max_available_bytes(const string &device_id) { return 0; }

  virtual vector<int> get_used_memory_counts(const string &device_id) {
    return {};
  }

protected:
  /** Call mem's Memory::alloc with retry.

      If first call of alloc is failed, it performs
      Allocator::free_unused_device_caches,
      then call Memory::alloc() of `mem` again.

      This function is designed to be used from a Allocator::alloc_impl
      function in a derived class.

      @param[in] mem A new instantiated Memory instance which is not yet
                     allocated by Memory::alloc yet.
   */
  void alloc_retry(shared_ptr<Memory> mem);

  // Virtual functions
  /** Request a memory block.

      Implementation must perform;

      - Create a new Memory instance
      - Invoke memory allocation of the created memory instance by
        Allocator::alloc_retry function.
      - Return the allocated Memory.

      @param[in] bytes Number of bytes of memory block requested.
      @param[in] device_id Device specifier string. A string format of specifier
                 is determined depending on Memory class implementation.
      @return An allocated Memory instance.
   */
  virtual shared_ptr<Memory> alloc_impl(size_t bytes,
                                        const string &device_id) = 0;

  /** Return memory to pool.

      In implementation of a derived class, the given memory should be returned
      to pool or simply freed if this is non-caching allocator.
   */
  virtual void free_impl(shared_ptr<Memory> memory) = 0;

  /** Free all unused memory blocks in a specified device.

      A derived class should discard Memory instances from pool which are not
      used or have been divided from an originally allocated memory block.

      @param[in] device_id Specifies device by a string.
      @return Size of freed memory in bytes.

   */
  virtual size_t free_unused_device_caches_impl(const string &device_id) = 0;

  virtual void print_memory_cache_map_impl(){};

  DISABLE_COPY_AND_ASSIGN(Allocator);
};
/*@}*/
/** \defgroup AllocatorImplGrp Allocator list */
}
