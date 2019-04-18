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

#pragma once

#include <nbla/defs.hpp>

#include <cstddef>
#include <memory>
#include <string>

namespace nbla {

using std::string;
using std::size_t;
using std::shared_ptr;

/** \addtogroup NNablaCoreGrp */
/*@{*/
/** Memory interface class.

The Memory class is designed to be managed by a Allocator instance.
This is extended to implement a device memory class which is responsible for
allocating/freeing a device memory block, and splitting into/merging blocks.
 */
class Memory {
private:
  size_t bytes_{0};
  string device_id_;
  bool locked_{false};
  Memory *next_{nullptr};
  Memory *prev_{nullptr};

protected:
  void *ptr_{nullptr};

private:
  inline void disable() { ptr_ = nullptr; }

public:
  /** Constructor will be called from a inherited class constructor to set
   bytes
   and device_id.
   */
  NBLA_API Memory(size_t bytes, const string &device_id);

  /** A derived class must implements destructor.

     This destructor does nothing.

     The memory allocated in Memory::alloc_impl function of a derived class

     must be freed in destuctor of the derived class. It is recommended to
     insert assertion ensuring that this->prev() returns false to check
     runtime fatal issue.

     @note If ptr_ is nullptr, a Memory has previously been merged to another,
     and is not retaining a memory block that should be freed.
   */
  virtual NBLA_API ~Memory();

  // Getter

  /** Get memory block size in bytes.
   */
  inline size_t bytes() const { return bytes_; }

  /** Get device id as a string.
   */
  inline string device_id() const { return device_id_; }

  /** Get a raw pointer retaining a memory block instance.

      In CpuMemory realization, it is a raw memory block allocated by malloc.
   */
  inline void *pointer() { return ptr_; }

  /** @copydoc pointer
   */
  inline const void *const_pointer() const { return ptr_; }

  /** This returns true if this Memory instance is in use.

      The lock state is managed by AllocatorMemory instance in a RAII way.
     Allocator::alloc function returns a AllocatorMemory instance retaining
     a Memory instance with lock. At the end of life of the AllocatorMemory
     instance,
     the lock is released to inform whether the memory instance can be merged
     with a consecutive (next_ or prev_) memory block at Memory::try_merge
     function.
   */
  inline bool locked() const { return locked_; }

  /** This returns whether the Memory instance was disabled by merging to
      another memory block.
   */
  inline bool disabled() { return !ptr_; }

  /** Returns the next memory block which has previously been split from an
      originally allocated memory block.
   */
  inline Memory *next() const { return next_; }

  /** Returns the previous memory block.

      @sa Memory::next
   */
  inline Memory *prev() const { return prev_; }

  // Setter
  /** Get a lock of this Memory instance when it's used to prevent merging
     with
     another.

     The lock is obtained by AllocatorMemory in its initialization.

   */
  inline void lock() { locked_ = true; }

  /** Relase a lock when it's not used.

      The lock is released by AllocatorMemory in its destructor.
   */
  inline void release() { locked_ = false; }

  // Logic
  /** Allocate memory by Memory::alloc_impl implemented in an implementation
      class.

      This should be called before using this instance, and is designed to be
      called via Allocator::alloc_retry which should be called in
      Allocator::alloc_impl in an implementation class.
   */
  void alloc();

  /** Returns number of bytes of memory block this owns.

      This returns 0 if this is not originally allocated memory instance,
      i.e., divided by another memory.
   */
  size_t bytes_active();

  /** This splits memory at an offset specified by second_start, and returns a
      second block. A first block is retained in this instance.
   */
  shared_ptr<Memory> divide(size_t second_start);

  /** Merge another memory block specified by from if it's possible.

      Merging is performed if the following conditions are met;

      - `from` is a valid pointer.
      - `from` is not locked.
      - `from` is next or previous of this.

      When merged, `from` will be disabled by Memory::disable.
   */
  void try_merge(Memory *from);

  /** Set Memory::prev_/Memory::next_ pointers of `left` and `right` as
      connected.
   */
  static void associate_consecutive(Memory *left, Memory *right);

  // Virtual functions
protected:
  /** Implementation must perform;

      - Allocate a memory block
      - Set an allocated memory to ptr_ as a void*.

      This is called from Memory::alloc.
      @note An implementation class must implement a destructor which frees
            the allocated memory block if it is originally allocated and still
            active. See CpuMemory::~CpuMemory().
      @return true if allocation succeeds.
   */
  virtual bool alloc_impl() = 0;

  /** Implementation must perform creating a new Memory object which retains a
      sub-block of memory which is previously allocated by Memory::alloc_impl.

      The byte size of this instance and memory continuity of this instance
      and the divided memory instance are modified by the interface function
      Memory::divide.

      @param[in] second_start Offset position where memory sub-block starts.
      @return A new created memory.
   */
  virtual shared_ptr<Memory> divide_impl(size_t second_start) = 0;

  /** Implementation must perform merging consecutive memory blocks (this and
      from).

      After merging, this instance must be modified to behave as a merged
      memory block. Memory continuity and byte size (prev_, next_, and bytes_)
      are modified by the interface function Memory::try_merge.
   */
  virtual void merge_next_impl(Memory *from) = 0;
  /** Implementation must perform merging consecutive memory blocks (from and
      this).

      @see Memory::merge_next_impl
   */
  virtual void merge_prev_impl(Memory *from) = 0;
};
/*@}*/
/** \defgroup MemoryImplGrp Memory list */
}
