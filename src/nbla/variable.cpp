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

// Variable.cpp
#include <nbla/variable.hpp>

#include <functional>
#include <memory>
#include <string>

namespace nbla {

using std::make_shared;

void Variable::update_shape_info() {
  size_ = compute_size_by_shape(shape_);
  strides_ = get_c_contiguous_strides(shape_);
  ndim_ = shape_.size();
}

Variable::Variable(const Shape_t &shape) : shape_(shape) {
  update_shape_info();
  data_.reset(new NdArray(shape_));
  grad_.reset(new NdArray(shape_));
}

Variable::Variable(NdArrayPtr data) {
  shape_ = data->shape();
  update_shape_info();
  data_ = data;
  grad_.reset(new NdArray(shape_));
}

void Variable::reshape(const vector<int64_t> &shape, bool force) {
  if (shape_ == shape)
    return;
  const Size_t size = compute_size_by_shape(shape);
  if (size_ == size) {
    shape_ = shape;
    update_shape_info();
    data_->reshape(shape);
    grad_->reshape(shape);
    return;
  }
  NBLA_CHECK(force, error_code::value,
             "Total dimensions not match. Set force=true if you want to "
             "resize array (clearing data). Given: %d != current: %d.",
             size, size_);
  shape_ = shape;
  update_shape_info();
  data_->reshape(shape_, true);
  grad_->reshape(shape_, true);
}

VariablePtr Variable::view() {
  auto v = make_shared<Variable>(shape_);
  v->set_data(data_);
  v->set_grad(grad_);
  return v;
}

VariablePtr Variable::view(const Shape_t &shape) {
  const Size_t size = compute_size_by_shape(shape);
  NBLA_CHECK(size == size_, error_code::value,
             "The total size must be the same as the variable. "
             "Given: %d != current: %d.",
             size, size_);
  auto v = make_shared<Variable>(shape);
  v->set_data(data_->view(shape));
  v->set_grad(grad_->view(shape));
  return v;
}

Size_t Variable::size(Size_t axis) const {
  if (axis <= 0) {
    return size_;
  }
  return compute_size_by_shape(shape_, axis);
}

void Variable::set_data(NdArrayPtr data) {
  NBLA_CHECK(data->shape() == shape_, error_code::value, "Shape must match.");
  data_ = data;
}

void Variable::set_grad(NdArrayPtr grad) {
  NBLA_CHECK(grad->shape() == shape_, error_code::value, "Shape must match.");
  grad_ = grad;
}
}
