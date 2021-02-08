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

// crelu.cpp

#include <nbla/array.hpp>
#include <nbla/function/crelu.hpp>
#include <nbla/variable.hpp>

#include <algorithm>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(CReLU, int);

template <typename T>
void CReLU<T>::setup_impl(const Variables &inputs, const Variables &outputs) {
  Shape_t in_shape = inputs[0]->shape();
  if (axis_ < 0)
    axis_ += in_shape.size();
  NBLA_CHECK(axis_ >= 0, error_code::value,
             "axis may not be less than zero, got %d", axis_);
  auto axis = static_cast<Shape_t::size_type>(axis_);
  NBLA_CHECK(axis < in_shape.size(), error_code::value,
             "axis must be less than ndim of inputs[0]. "
             "axis: %d >= ndim of input: %d.",
             axis_, in_shape.size());
  in_shape[axis] *= 2;
  outputs[0]->reshape(in_shape, true);
  Size_t size = inputs[0]->size();
  size0_ = inputs[0]->size(axis);
  size1_ = size / size0_;
  NBLA_CHECK(size0_ * size1_ == size, error_code::unclassified,
             "An error occurred during setup CReLU function.");
}

template <class T>
void CReLU<T>::forward_impl(const Variables &inputs, const Variables &outputs) {
  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  T *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_, true);
  for (int i1 = 0; i1 < size1_; ++i1) {
    for (int i0 = 0; i0 < size0_; ++i0) {
      y[i1 * size0_ * 2 + i0] = std::max(T(0), x[i1 * size0_ + i0]);
      y[i1 * size0_ * 2 + size0_ + i0] =
          std::max(T(0), -1 * x[i1 * size0_ + i0]);
    }
  }
}

template <class T>
void CReLU<T>::backward_impl(const Variables &inputs, const Variables &outputs,
                             const vector<bool> &propagate_down,
                             const vector<bool> &accum) {
  if (!propagate_down[0]) {
    return;
  }
  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  T *dx = inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_, !accum[0]);
  const T *dy = outputs[0]->get_grad_pointer<T>(this->ctx_);
  for (int i1 = 0; i1 < size1_; ++i1) {
    for (int i0 = 0; i0 < size0_; ++i0) {
      T &rdx = dx[i1 * size0_ + i0];
      if (x[i1 * size0_ + i0] > 0)
        rdx = (accum[0] ? rdx : (T)0) + dy[i1 * size0_ * 2 + i0];
      else
        rdx = (accum[0] ? rdx : (T)0) - dy[i1 * size0_ * 2 + size0_ + i0];
    }
  }
}
}
