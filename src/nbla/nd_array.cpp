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

#include <nbla/nd_array.hpp>

#include <memory>

namespace nbla {

using std::make_shared;

SyncedArrayPtr BaseNdArray::array() { return array_; }

void BaseNdArray::update_python_user_reference_counts(const int diff) {
  python_user_reference_counts += diff;
  array_->update_python_user_reference_counts(diff);
}

void BaseNdArray::set_array(SyncedArrayPtr array) {
  NBLA_CHECK(size_ == array->size(), error_code::value, "Size must match.");

  // New SyncedArray traces the NdArray tree.
  array->update_python_user_reference_counts(python_user_reference_counts);

  if (array_) {
    // Old SyncedArray purges the NdArray tree.
    array_->update_python_user_reference_counts(-python_user_reference_counts);
  }

  array_ = array;
}

BaseNdArray::~BaseNdArray() {
  array_->update_python_user_reference_counts(-python_user_reference_counts);
}

void NdArray::update_shape_info() {
  size_ = compute_size_by_shape(shape_);
  strides_ = get_c_contiguous_strides(shape_);
  ndim_ = shape_.size();
}

NdArray::NdArray(const Shape_t &shape) : shape_(shape) {
  update_shape_info();
  this->set_array(make_shared<SyncedArray>(size_));
}

NdArray::NdArray(SyncedArrayPtr array, const Shape_t &shape) : shape_(shape) {
  update_shape_info();
  NBLA_CHECK(array->size() == size_, error_code::value,
             "The total size of array must be the same as the shape. "
             "Array size: %d, shape size: %d.",
             array->size(), size_);
  this->set_array(array);
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
  NBLA_CHECK(!array()->is_narrowed(), error_code::value,
             "Narrowed NdArray does not allow reshape to change size.");

  shape_ = shape;
  update_shape_info();
  this->set_array(make_shared<SyncedArray>(size_));
}

NdArrayPtr NdArray::view(const Shape_t &shape) {
  return make_shared<NdArray>(this->array(), shape);
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
void NdArray::zero() { this->array()->zero(); }
void NdArray::fill(double v) { this->array()->fill(v); }
const Array *NdArray::get(dtypes dtype, const Context &ctx) {
  return this->array()->get(dtype, ctx);
}
shared_ptr<const Array> NdArray::get_sp(dtypes dtype, const Context &ctx) {
  return this->array()->get_sp(dtype, ctx);
}

unsigned long NdArray::data_ptr(dtypes dtype, const Context &ctx,
                                bool write_only) {
  return (unsigned long)reinterpret_cast<uintptr_t>(
      this->array()->data_ptr(dtype, ctx, write_only));
}

Array *NdArray::cast(dtypes dtype, const Context &ctx, bool write_only) {
  return this->array()->cast(dtype, ctx, write_only);
}

shared_ptr<Array> NdArray::cast_sp(dtypes dtype, const Context &ctx,
                                   bool write_only) {
  return this->array()->cast_sp(dtype, ctx, write_only);
}

NdArrayPtr NdArray::narrow(const Size_t dim, const Size_t start,
                           const Size_t length) {

  // TODO: Corresponds to any dimension.
  NBLA_CHECK(dim == 0, error_code::value,
             "dim is out of range (expected to be 0, but got %d)", dim);
  Size_t narrow_dim_size = shape_[dim];

  NBLA_CHECK(start >= 0, error_code::value,
             "start is out of range (expected to be [0, %d], but got %d)",
             narrow_dim_size - 1, start);
  NBLA_CHECK(length >= 0, error_code::value,
             "negative number for length (%d) is not permitted", length);
  NBLA_CHECK(start + length <= narrow_dim_size, error_code::value,
             "start (%d) + length (%d) exceeds dimension size (%d)", start,
             length, narrow_dim_size);

  auto narrowed_shape = this->shape();
  narrowed_shape[0] = length;

  const auto offset = start * compute_size_by_shape(narrowed_shape, 1);
  const auto narrowed_size = compute_size_by_shape(narrowed_shape);

  auto narrowed_array = this->array()->narrow(narrowed_size, offset);

  return make_shared<NdArray>(narrowed_array, narrowed_shape);
}
}
