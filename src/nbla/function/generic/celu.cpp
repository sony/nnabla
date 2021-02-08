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

/** CELU
*/
#include <nbla/array.hpp>
#include <nbla/function/celu.hpp>
#include <nbla/variable.hpp>

#include <algorithm>
#include <cmath>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(CELU, double, int);

template <typename T>
void CELU<T>::setup_impl(const Variables &inputs, const Variables &outputs) {
  Shape_t in_shape = inputs[0]->shape();
  if (axis_ < 0)
    axis_ += in_shape.size();
  NBLA_CHECK(axis_ >= 0, error_code::value,
             "axis must not be less than zero, got %d", axis_);
  auto axis = static_cast<Shape_t::size_type>(axis_);
  NBLA_CHECK(axis < in_shape.size(), error_code::value,
             "axis must be less than ndim of inputs[0]. "
             "axis: %d >= ndim of input: %d.",
             axis_, in_shape.size());
  in_shape[axis_] *= 2;
  outputs[0]->reshape(in_shape, true);
  Size_t size = inputs[0]->size();
  size0_ = inputs[0]->size(axis_);
  size1_ = size / size0_;
  NBLA_CHECK(size0_ * size1_ == size, error_code::unclassified,
             "An error occurred during setup CELU function.");
}

template <typename T>
void CELU<T>::forward_impl(const Variables &inputs, const Variables &outputs) {
  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  T *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_, true);
  for (int i1 = 0; i1 < size1_; ++i1) {
    for (int i0 = 0; i0 < size0_; ++i0) {
      const int j0 = i1 * size0_ * 2 + i0;
      const T &xk = x[i1 * size0_ + i0];
      y[j0] = 0 <= xk ? xk : (T)alpha_ * (std::exp(xk) - 1);
      y[j0 + size0_] = xk <= 0 ? -xk : (T)alpha_ * (std::exp(-xk) - 1);
    }
  }
}

template <typename T>
void CELU<T>::backward_impl(const Variables &inputs, const Variables &outputs,
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
      const int j0 = i1 * size0_ * 2 + i0;
      const int j1 = j0 + size0_;
      const int k = i1 * size0_ + i0;
      const T &dyj0 = dy[j0];
      const T &dyj1 = dy[j1];
      const T &xk = x[k];
      const T d = (0 <= xk ? dyj0 : dyj0 * (T)alpha_ * std::exp(xk)) -
                  (xk <= 0 ? dyj1 : dyj1 * (T)alpha_ * std::exp(-xk));
      dx[k] = (accum[0] ? dx[k] : (T)0) + d;
    }
  }
}

} // namespace nbla
