// Copyright 2017,2018,2019,2020,2021 Sony Corporation.
// Copyright 2021 Sony Group Corporation.
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

/** Synced array
 */
#ifndef __NBLA_SYNCED_ARRAY_HPP__
#define __NBLA_SYNCED_ARRAY_HPP__
#include <nbla/array.hpp>
#include <nbla/common.hpp>
#include <nbla/context.hpp>

#include <map>
#include <memory>

namespace nbla {

using std::pair;

/** Synchronized array interface that implicitly transfers and cast arrays
over devices and data types.
\ingroup NNablaCoreGrp
*/
class NBLA_API SyncedArray : public enable_shared_from_this<SyncedArray> {
  struct ArrayDesc {
    string key;
    string array_class;
    dtypes dtype;
  };
  ArrayDesc head_;         ///< Head desc for transferring content.
  bool zeroing_lazy_eval_; ///< Flag for lazy evaluation of zero() function.
  bool filling_lazy_eval_; ///< Flag for lazy evaluation of fill() function.
  bool zeroing_;           ///< Flag for updated or not after zero().
  float fill_value_; ///< Filling value used in lazy eval of fill() function.
  Size_t size_;      ///< Size.
  Size_t offset_;    ///< Offset of the allocated memory.
  /// < key: array_class and memory range, value:<Array instance,is_head>
  map<string, pair<shared_ptr<Array>, bool>> array_;
  size_t modification_count_; ///< Number of modification count.
  bool clear_called_; ///< clear called flag. If clear_all_array is called, it
                      /// turns to true. When cast_sp, get_sp, zero, or fill are
                      /// called, it turns to false.

  /** Parameters of the linked list.
   * NOTE : Narrow method exists in NdArray to divide the memory.
   * If used narrow, SyncedArray has a tree-like connected relationship such
   * that it has a single parent and multiple children.
   */
  shared_ptr<SyncedArray> parent_; ///< Pointer of the parent SyncedArray
  std::vector<weak_ptr<SyncedArray>>
      children_; ///< list of pointers of child SyncedArrays

  // Reference counts of this instance from Python user
  int python_user_reference_counts = 0;

public:
  SyncedArray(const Size_t size);
  SyncedArray(shared_ptr<SyncedArray> parent, const Size_t size,
              const Size_t offset);
  ~SyncedArray();

  /** Cast and get array with dtype context

  This will return an array object with specified dtype and device with implicit
  synchronization over different dtypes/devices.

  @param[in] write_only When true, just returns an Array instance requested
  without synchronization.
  @param[in] async_flags AsyncFlag::NONE  -> Synchronous synchronization
                         AsyncFlag::ASYNC -> Asynchronous synchronization
                         AsyncFlag::UNSAFE  -> No synchronization to host
                         AsyncFlag::ASYNC | AsyncFlag::UNSAFE ->
                         The memory region of the source array of an
                         asynchronous data transfer is not guranteed to be
                         kept safe until the end of the transfer.

  */
  Array *cast(dtypes dtype, const Context &ctx, bool write_only = false,
              const int async_flags = AsyncFlag::NONE);

  /** Cast and get array as a shared_ptr

      @sa cast

   */
  shared_ptr<Array> cast_sp(dtypes dtype, const Context &ctx,
                            bool write_only = false,
                            const int async_flags = AsyncFlag::NONE);

  /** Get array with dtype context.

  This will return an array object with specified dtype and device with implicit
  synchronization. Note that this function call does not move the head in array
  list.

  TODO: Is "const" member function appropriate? This implicitly creates or
  modify array contents of specified dtype-context.
  */
  const Array *get(dtypes dtype, const Context &ctx,
                   const int async_flags = AsyncFlag::NONE);

  /** Get array as a shared pointer.
   */
  shared_ptr<const Array> get_sp(dtypes dtype, const Context &ctx,
                                 const int async_flags = AsyncFlag::NONE);

  /** Get the head array.
   */
  Array *head_array();

  /** Get the head array as a shared pointer.
   */
  shared_ptr<Array> head_array_sp();

  /** Get array's ptr.

      @param[in] dtype Enum of data type.
      @param[in] ctx Descriptor of array backend.
      @param[in] write_only No synchronization happens.
      @param[in] async_flags:
        AsyncFlag::NONE  -> Synchronous synchronization happens.
        AsyncFlag::ASYNC -> Asynchronous synchronization happens.
        AsyncFlag::SAFE  -> Same as AsyncFlag::NONE.
        AsyncFlag::ASYNC | AsyncFlag::SAFE -> Asynchronous synchronization
     happens
                                              and the synchronized source array
                                              keeps safe against the host
     operation.
   */
  const void *data_ptr(dtypes dtype, const Context &ctx,
                       bool write_only = false,
                       const int async_flags = AsyncFlag::NONE);

  /** Get dtype
   */
  inline dtypes dtype() const {
    NBLA_CHECK(!head_.key.empty(), error_code::unclassified,
               "Array is not initialized.");
    return head_.dtype;
  };

  /** Returns true if this SyncedArray has the head array.
   */
  bool has_head_array() { return !head_.key.empty() && !array_.empty(); }

  /** Get the array class of the head.
   *
   */
  inline string head_array_class() { return head_.array_class; }

  /** Get the number of arrays
   */
  inline Size_t get_num_arrays() const { return array_.size(); }

  /** Size. */
  inline Size_t size() const { return size_; }

  /** Fill all element with 0.

  Note: This is lazily evaluated at calling of get() or cast().
  */
  void zero();

  /** Fill all element with given float value.
   */
  void fill(float value);

  /** Get number of modification count.

      Modification accounts for calling either cast, zero or fill.
  */
  size_t modification_count() const;

  /** Get clear called flag.
   */
  bool clear_called() const;

  /** Copy values from another SynedArray.

      @note The copy is happening in a device and a dtype of source array.
   */
  void copy_from(const SyncedArray *src);

  void clear();

  /** Get whether or not it fills array values obtained in cast/get call later.

      This is provided to determine gradient accumulation flags in our
     computation graph engine, as well as to determine whether or not solver and
     communicator execute their operations by depending on whether gradients are
     updated.

   */
  bool zeroing() const;

  /** Get a new narrowed SyncedArray.
   */
  shared_ptr<SyncedArray> narrow(const Size_t narrow_size, const Size_t offset);

  /** Returns true if this SyncedArray is narrowed.
   */
  bool is_narrowed() const { return parent_ != nullptr; }

  /** Returns true if this SyncedArray has the Parent.
   */
  inline bool is_child() const { return parent_ != nullptr; }

  /** Returns true if this SyncedArray has not the Parent.
   */
  inline bool is_root() const { return parent_ == nullptr; }

  /** Returns true if this SyncedArray has a parent or child SyncedArrays.
   */
  bool has_family() const {
    return static_cast<bool>(parent_ || children_.size() > 0);
  }

  /** Get the refernce counts from a Python user.
   */
  int get_python_user_reference_counts() const;

  /** Update the refernce counts from a Python user.
   */
  void update_python_user_reference_counts(const int diff);

private:
  ArrayDesc sync(dtypes dtype, const Context &ctx, bool write_only = false,
                 const int async_flags = AsyncFlag::NONE);

  void clear_all_array();

  /** Call clear_all_array for descendants than itself.
   * If keep_head is true, clear all array that are not parent-child
   * relationship. For example, get array(NdArray, SyncedArray) with a different
   * type will produce Array with no parent-child relationship.
   */
  void clear_all_array_descendants(bool keep_head);

  // Clearing zero and fill flags for lazy evaluation.
  void clear_flags();

  /** Clear flags of all child arrays for the zero function.
   */
  void propagate_zeroing_flag_descendants(bool flag);

  /** Clear flags for the zero function (including
   * propagate_zeroing_flag_descendants()).
   */
  void propagate_zeroing_flag(bool flag);

  /** Remove a child SyncedArray in the children_.
   */
  void remove_child(const SyncedArray *child);

  /** Get offset of the Array.
   */
  Size_t offset() const { return offset_; }

  /** Get root of parent-child SyncedArrays.
   */
  shared_ptr<SyncedArray> get_root();

  /** Traverses the tree of parent-child SyncedArrays and calls zero() and
   * fill().
   */
  void traverse_zero_fill();

  /** Traverses child SyncedArrays and returns true if any of zeroing or filling
   * flags is true.
   */
  bool check_zeroing_filling_descendants();

  /** Traverses the tree of parent-child SyncedArrays and returns true if any of
   * zeroing or filling flags is true.
   */
  bool check_zeroing_filling();

  /** Clears flags in child SyncedArrays.
   */
  void clear_flags_descendants();

  /** Create arrays of descendants from itself.
   */
  void create_array_descendants();

  /** Create child Array.
   */
  void create_array_from_parent();

  DISABLE_COPY_AND_ASSIGN(SyncedArray);
};

///< Shared pointer of SyncedArray.
typedef shared_ptr<SyncedArray> SyncedArrayPtr;

class SingletonManager; // Forward declaration for friend

/// Get, cast, or clear
enum SyncedArrayCallbackTag { GET, CAST, CLEAR };

/// Type of callback function for get, cast, and clear of SyncedArray.
using synced_array_callback_func_type =
    function<void(SyncedArrayPtr saptr, const SyncedArrayCallbackTag func_name,
                  const dtypes dtype, const Context &ctx, const bool write_only,
                  const bool first_creation, const bool off_recording)>;

/**
Singleton class to store a callback function for get, cast, and clear of
SyncedArray.
*/
class NBLA_API SyncedArrayCallback {
  synced_array_callback_func_type callback_func_;

public:
  ~SyncedArrayCallback();

  /** Check if callback function is not set. */
  bool empty();

  /** Set a callback function */
  void set_callback_func(synced_array_callback_func_type f);

  /** Call callback */
  void call_callback(SyncedArrayPtr saptr,
                     const SyncedArrayCallbackTag func_name, const dtypes dtype,
                     const Context &ctx, const bool write_only,
                     const bool first_creation, const bool off_recording);

private:
  friend SingletonManager; // needs forward declaration
                           // Never called by users.
  SyncedArrayCallback();
  DISABLE_COPY_AND_ASSIGN(SyncedArrayCallback);
};
} // namespace nbla
#endif
