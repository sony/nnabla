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

using std::map;
using std::shared_ptr;
using std::pair;

/** Synchronized array interface that implicitly transfers and cast arrays
over devices and data types.
\ingroup NNablaCoreGrp
*/
class NBLA_API SyncedArray {
  struct ArrayDesc {
    string key;
    string array_class;
    dtypes dtype;
  };
  ArrayDesc head_;   ///< Head desc for transferring content.
  bool zeroing_;     ///< Flag for lazy evaluation of zero() function.
  bool filling_;     ///< Flag for lazy evaluation of fill() function.
  float fill_value_; ///< Filling value used in lazy eval of fill() function.
  Size_t size_;      ///< Size.
  map<string, pair<shared_ptr<Array>, bool>>
      array_; ///< key: array_class, value: <Array instance, is_head>
  size_t modification_count_; ///< Number of modification count.

public:
  SyncedArray(const Size_t size);
  ~SyncedArray();

  /** Cast and get array with dtype context

  This will return an array object with specified dtype and device with implicit
  synchronization over different dtypes/devices.

  @param[in] write_only When true, just returns an Array instance requested
  without synchronization.
  */
  Array *cast(dtypes dtype, const Context &ctx, bool write_only = false);

  /** Cast and get array as a shared_ptr

      @sa cast

   */
  shared_ptr<Array> cast_sp(dtypes dtype, const Context &ctx,
                            bool write_only = false);

  /** Get array with dtype context.

  This will return an array object with specified dtype and device with implicit
  synchronization. Note that this function call does not move the head in array
  list.

  TODO: Is "const" member function appropriate? This implicitly creates or
  modify array contents of specified dtype-context.
  */
  const Array *get(dtypes dtype, const Context &ctx);

  /** Get array as a shared pointer.
   */
  shared_ptr<const Array> get_sp(dtypes dtype, const Context &ctx);

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
   */
  const void *data_ptr(dtypes dtype, const Context &ctx,
                       bool write_only = false);

  /** Get dtype
  */
  inline dtypes dtype() const {
    NBLA_CHECK(!head_.key.empty(), error_code::unclassified,
               "Array is not initialized.");
    return head_.dtype;
  };

  /** Get the array class of the head.
   *
   */
  inline std::string head_array_class() { return head_.array_class; }

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

private:
  ArrayDesc sync(dtypes dtype, const Context &ctx, bool write_only = false);

  void clear_all_array();

  // Clearing zero and fill flags for lazy evaluation.
  void clear_flags();

  DISABLE_COPY_AND_ASSIGN(SyncedArray);
};

///< Shared pointer of SyncedArray.
typedef shared_ptr<SyncedArray> SyncedArrayPtr;
}
#endif
