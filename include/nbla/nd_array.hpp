// Copyright 2017,2018,2019,2020,2021 Sony Corporation.
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

#ifndef __NBLA_ND_ARRAY_HPP__
#define __NBLA_ND_ARRAY_HPP__

#include <nbla/synced_array.hpp>

#include <memory>

namespace nbla {

using std::make_shared;

/** Dtype and backend agnostic multi-dimensional array.
 */
class NdArray {
  SyncedArrayPtr array_;
  Shape_t shape_;
  Shape_t strides_;
  Size_t size_;
  Size_t ndim_;

  /** Update shape info by shape.
   */
  void update_shape_info();

public:
  typedef shared_ptr<NdArray> Ptr;

  /** Create a shared_ptr instance of NdArray.
   */
  template <typename... Args> static Ptr create(Args... args) {
    return make_shared<NdArray>(args...);
  }

  /** Ctor given shape.

      Create an SyncedArray instance with corresponding size.

      @param[in] shape Shape of creating array.
   */
  NBLA_API NdArray(const Shape_t &shape = {});

  /** Ctor given previously created SyncedArray and shape.

      @param[in] array Previously created SyncedArray.
      @param[in] shape Shape of N-d array. Total size must be the same as
     SyncedArray.
   */
  NBLA_API NdArray(SyncedArrayPtr array, const Shape_t &shape = {});

  /** Reshape.

      @param[in] shape N-d array will be reshaped to this shape.
      @param[in] force If total size doesn't match and true is given, array will
     reset to total size, which means the content of array will become totally
     different one.
   */
  NBLA_API void reshape(const Shape_t &shape, bool force = false);

  /** Create an another instance with different shape but sharing array content.

      @param[in] shape N-d array reshaped to.
   */
  NBLA_API Ptr view(const Shape_t &shape);

  /** Get shape.
   */
  NBLA_API Shape_t shape() const;

  /** Get strides.
   */
  NBLA_API Shape_t strides() const;

  /** Get total size (product of shape array).

  @param axis Size followed by given axis is computed.
   */
  NBLA_API Size_t size(Size_t axis = -1) const;

  /** Get number of dimensions.
   */
  NBLA_API Size_t ndim() const;

  /** Get SyncedArray instance held by this instance.

      @note This is not copying data. Modifying content affects this.
   */
  NBLA_API SyncedArrayPtr array();

  /** Replace SyncedArray instance with previously created another one.
   */
  NBLA_API void set_array(SyncedArrayPtr array);

  /** Set all value to zero.

      @note This will be lazily evaluated when data is used.
  */
  NBLA_API void zero();

  /** Set all value to given value.

      @param[in] value Value array filled with.

      @note This will be lazily evaluated when data is used.
  */
  NBLA_API void fill(double v);

  /** Get const array with specified dtype and backend description.

      @param[in] dtype Enum of data type.
      @param[in] ctx Descriptor of array backend.
   */
  NBLA_API const Array *get(dtypes dtype, const Context &ctx);

  /** Get const array as a shared_ptr.

      @sa get
   */
  NBLA_API shared_ptr<const Array> get_sp(dtypes dtype, const Context &ctx);

  /** Get array's ptr.

      @param[in] dtype Enum of data type.
      @param[in] ctx Descriptor of array backend.
      @param[in] write_only No synchronization happens.
   */
  NBLA_API unsigned long data_ptr(dtypes dtype, const Context &ctx,
                                  bool write_only = false);

  /** Get mutable array with specified dtype and backend description.

      @param[in] dtype Enum of data type.
      @param[in] ctx Descriptor of array backend.
      @param[in] write_only No synchronization happens.
   */
  NBLA_API Array *cast(dtypes dtype, const Context &ctx,
                       bool write_only = false);

  /** Get a mutable array as a shared_ptr

      @sa cast
   */
  NBLA_API shared_ptr<Array> cast_sp(dtypes dtype, const Context &ctx,
                                     bool write_only = false);

  DISABLE_COPY_AND_ASSIGN(NdArray);
};

///< Shared pointer of NdArray
typedef NdArray::Ptr NdArrayPtr;
}
#endif
