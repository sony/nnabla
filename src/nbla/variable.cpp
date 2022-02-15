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

// Variable.cpp
#include <nbla/variable.hpp>

#include <functional>
#include <memory>
#include <string>

namespace nbla {

using std::make_shared;

void BaseVariable::update_python_user_reference_counts(const int diff) {
  python_user_reference_counts += diff;
  data_->update_python_user_reference_counts(diff);
}

void BaseVariable::set_data(NdArrayPtr data) {
  NBLA_CHECK(data->shape() == shape_, error_code::value, "Shape must match.");

  // New NdArray traces the Variable tree.
  data->update_python_user_reference_counts(python_user_reference_counts);

  if (data_) {
    // Old NdArray purges the Variable tree.
    data_->update_python_user_reference_counts(-python_user_reference_counts);
  }

  data_ = data;
}

BaseVariable::~BaseVariable() {
  data_->update_python_user_reference_counts(-python_user_reference_counts);
}

void BaseVariable::set_grad(NdArrayPtr grad) {
  NBLA_CHECK(grad->shape() == shape_, error_code::value, "Shape must match.");
  grad_ = grad;
}

void Variable::update_shape_info() {
  size_ = compute_size_by_shape(shape_);
  strides_ = get_c_contiguous_strides(shape_);
  ndim_ = shape_.size();
}

Variable::Variable(const Shape_t &shape) {
  this->shape_ = shape;
  update_shape_info();
  this->set_data(make_shared<NdArray>(shape_));
  this->set_grad(make_shared<NdArray>(shape_));
}

Variable::Variable(NdArrayPtr data) {
  shape_ = data->shape();
  update_shape_info();
  this->set_data(data);
  this->set_grad(make_shared<NdArray>(shape_));
}

void Variable::reshape(const vector<int64_t> &shape, bool force) {
  if (shape_ == shape)
    return;
  const Size_t size = compute_size_by_shape(shape);
  if (size_ == size) {
    shape_ = shape;
    update_shape_info();
    this->data()->reshape(shape);
    this->grad()->reshape(shape);
    return;
  }
  NBLA_CHECK(force, error_code::value,
             "Total dimensions not match. Set force=true if you want to "
             "resize array (clearing data). Given: %d != current: %d.",
             size, size_);
  shape_ = shape;
  update_shape_info();
  this->data()->reshape(shape_, true);
  this->grad()->reshape(shape_, true);
}

VariablePtr Variable::view() {
  auto v = make_shared<Variable>(shape_);
  v->set_data(this->data());
  v->set_grad(this->grad());
  return v;
}

VariablePtr Variable::view(const Shape_t &shape) {
  const Size_t size = compute_size_by_shape(shape);
  NBLA_CHECK(size == size_, error_code::value,
             "The total size must be the same as the variable. "
             "Given: %d != current: %d.",
             size, size_);
  auto v = make_shared<Variable>(shape);
  v->set_data(this->data()->view(shape));
  v->set_grad(this->grad()->view(shape));
  return v;
}

Size_t Variable::size(Size_t axis) const {
  if (axis <= 0) {
    return size_;
  }
  return compute_size_by_shape(shape_, axis);
}
}
