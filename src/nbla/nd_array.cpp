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

#include <nbla/nd_array.hpp>

#include <memory>

namespace nbla {

using std::make_shared;

void NdArray::update_shape_info() {
  size_ = compute_size_by_shape(shape_);
  strides_ = get_c_contiguous_strides(shape_);
  ndim_ = shape_.size();
}

NdArray::NdArray(const Shape_t &shape) : shape_(shape) {
  update_shape_info();
  array_.reset(new SyncedArray(size_));
}

NdArray::NdArray(SyncedArrayPtr array, const Shape_t &shape) : shape_(shape) {
  update_shape_info();
  NBLA_CHECK(array->size() == size_, error_code::value,
             "The total size of array must be the same as the shape. "
             "Array size: %d, shape size: %d.",
             array->size(), size_);
  array_ = array;
}

void NdArray::reshape(const Shape_t &shape, bool force) {
  if (shape_ == shape)
    return;
  const Size_t size = compute_size_by_shape(shape);
  if (size_ == size) {
    shape_ = shape;
    update_shape_info();
    return;
  }
  NBLA_CHECK(force, error_code::value, "Total dimensions not match. Set "
                                       "force=true if you want to resize array "
                                       "(clearing data).");
  shape_ = shape;
  update_shape_info();
  array_.reset(new SyncedArray(size_));
}

NdArrayPtr NdArray::view(const Shape_t &shape) {
  return make_shared<NdArray>(array_, shape);
}

Shape_t NdArray::shape() const { return shape_; }
Shape_t NdArray::strides() const { return strides_; }

Size_t NdArray::size(Size_t axis) const {
  if (axis <= 0) {
    return size_;
  }
  return compute_size_by_shape(shape_, axis);
}

Size_t NdArray::ndim() const { return ndim_; }
SyncedArrayPtr NdArray::array() { return array_; }
void NdArray::set_array(SyncedArrayPtr array) {
  NBLA_CHECK(size_ == array->size(), error_code::value, "Size must match.");
  array_ = array;
}
void NdArray::zero() { array_->zero(); }
void NdArray::fill(double v) { array_->fill(v); }
const Array *NdArray::get(dtypes dtype, const Context &ctx) {
  return array_->get(dtype, ctx);
}
shared_ptr<const Array> NdArray::get_sp(dtypes dtype, const Context &ctx) {
  return array_->get_sp(dtype, ctx);
}

unsigned long NdArray::data_ptr(dtypes dtype, const Context &ctx,
                                bool write_only) {
  return (unsigned long)reinterpret_cast<uintptr_t>(
      array_->data_ptr(dtype, ctx, write_only));
}

Array *NdArray::cast(dtypes dtype, const Context &ctx, bool write_only) {
  return array_->cast(dtype, ctx, write_only);
}

shared_ptr<Array> NdArray::cast_sp(dtypes dtype, const Context &ctx,
                                   bool write_only) {
  return array_->cast_sp(dtype, ctx, write_only);
}
}
