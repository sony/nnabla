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

#include <nbla/defs.hpp>

#include <cstddef>
#include <memory>
#include <string>

namespace nbla {

using std::string;
using std::size_t;
using std::shared_ptr;

/** Memory status on the device.
 */

enum DeviceMemoryState {
  Locked =
      0, ///< This memory will be used on device in future. Cannot clear it.
  Unlocked = 1, ///< This memory is no longer used on device. Can clear it.
};

/** Physical memory interface class
 */

class PhysicalMemory {
protected:
  bool allocated_;
  size_t bytes_;
  string device_id_;

public:
  explicit PhysicalMemory(size_t bytes, const string &device_id)
      : allocated_(false), bytes_{bytes}, device_id_{device_id} {};

  ~PhysicalMemory() = default;

  /** Allocate physical memory at least larger than bytes_.
   *  This method must be implemented in derived class.
   * @return Size of allocated bytes. (Requested bytes and actual allocated
   * bytes would be different.)
   */
  virtual size_t alloc() = 0;

  inline size_t bytes() { return bytes_; };

  inline bool allocated() { return allocated_; };
};

typedef shared_ptr<PhysicalMemory> PhysicalMemoryPtr;
typedef std::vector<PhysicalMemoryPtr> VecPhysicalMemoryPtr;

/** \addtogroup NNablaCoreGrp */
/*@{*/
/** Memory interface class.

The Memory class is designed to be managed by a Allocator instance.
This is extended to implement a device memory class which is responsible for
allocating/freeing a device memory block, and splitting into/merging blocks.
 */
class Memory {
private:
  string device_id_;
  bool locked_{false};
  Memory *next_{nullptr};
  Memory *prev_{nullptr};

protected:
  /** Type of Memory Flags.
  */
  enum MemoryType {
    Normal = 0,  ///< Default memory type.
    Virtual = 1, ///< Virtual memory having physical memories internally.
  };

  size_t bytes_{0};
  long long requested_bytes_{-1};
  void *ptr_{nullptr};
  MemoryType memory_type_{MemoryType::Normal};
  VecPhysicalMemoryPtr p_memories_;

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

  /** Get requested allocation memory size in bytes.
   */
  inline size_t requested_bytes() const {
    return requested_bytes_ < 0 ? bytes_ : requested_bytes_;
  }

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

  /** Release a lock when it's not used.

      The lock is released by AllocatorMemory in its destructor.
   */
  inline void release() { locked_ = false; }

  /** Get physical memory as reference. **/
  inline VecPhysicalMemoryPtr &get_physical_memory() { return p_memories_; }

  /** Append physical memories **/
  inline void append_physical_memories(VecPhysicalMemoryPtr &p_mems) {
    for (auto &m : p_mems)
      p_memories_.emplace_back(m);
  }

  /** Clear physical memory. **/
  inline void clear_physical_memory() { p_memories_.clear(); }

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

  /** Bind physical memories on virtual address.
   *  This method can be executed only if memory_type_ == MemType::Virtual.
   */
  void bind();

  /** Unbind physical memories from corresponding virtual address, and return
   * physical memory list as vector.
   *  This method can be executed only if memory_type_ == MemType::Virtual.
   */
  void unbind();

  /** Glow virtual memory from already allocated virtual address.
   */
  bool grow(VecPhysicalMemoryPtr &p_mems);

  /** Get device memory state.
   *  In default this function does noting and return
   * DeviceMemoryState::Unlocked.
   */
  virtual DeviceMemoryState get_device_memory_state() {
    return DeviceMemoryState::Unlocked;
  }

  /** Request device memory state to `request`.
   * `request` must be DeviceMemoryState.
   * In default this function does nothing.
   */
  virtual void lock_device_memory(){};

  inline void set_requested_bytes(size_t bytes) { requested_bytes_ = bytes; }

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

  /** Implementation must perform following things:
   * - Make sure physical memory is already allocated. (and alloc physical
   * memory if needed.)
   * - Reserve virtual address whose bytes size is larger than bytes_.
   * - Map physical memory to virtual address and set virtual address to ptr_ as
   * void*.
   * In default, this function raises not implemented error.
   */
  virtual void bind_impl() {
    NBLA_ERROR(error_code::not_implemented, "bind_impl() is not implemented.");
  };

  /** Implementation must perform following things:
   * - Unmap virtual address from physical memory.
   * - Release virtual address.
   * In default, this function raises not implemented error.
   */
  virtual void unbind_impl() {
    NBLA_ERROR(error_code::not_implemented,
               "unbind_impl() is not implemented.");
  };

  /** Implementation must perform following things:
   * - Grow virtual memory to match its physical memory size.
   *  (If the size of virtual memory is alredy the same as physical, do nothing)
   * - If fail to allocate virtual address aligned with current virtual address,
   * fall back to slower implementation. (Simply retry unbind->bind)
   * - Return false if failed to grow.
   * In default, this function raises not implemented error.
   */
  virtual bool grow_impl(VecPhysicalMemoryPtr &p_mems) {
    NBLA_ERROR(error_code::not_implemented, "grow_impl() is not implemented.");
  };
};
/*@}*/
/** \defgroup MemoryImplGrp Memory list */
}
