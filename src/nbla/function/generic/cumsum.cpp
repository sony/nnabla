// Copyright (c) 2020 Sony Corporation. All Rights Reserved.
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

#include <nbla/array.hpp>
#include <nbla/common.hpp>
#include <nbla/function/cumsum.hpp>
#include <nbla/variable.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(CumSum, int, bool, bool);

template <typename T>
void CumSum<T>::setup_impl(const Variables &inputs, const Variables &outputs) {
  Shape_t in_shape = inputs[0]->shape();
  Size_t size = inputs[0]->size();
  if (axis_ < 0) {
    axis_ += in_shape.size();
    NBLA_CHECK(axis_ >= 0, error_code::value,
               "Absolute value of axis must be less than that of input ndim. "
               "axes[%d]: %d >= ndim of input: %d.",
               abs(axis_ - static_cast<int>(in_shape.size())), in_shape.size());
  }
  NBLA_CHECK(static_cast<unsigned>(axis_) < in_shape.size(), error_code::value,
             "axis must be less than ndim of inputs[0]. "
             "axis: %d >= ndim of inputs[0]: %d.",
             axis_, in_shape.size());

  Size_t size_axis = inputs[0]->size(axis_);

  size_ = inputs[0]->size();       // Total size
  size0_ = size / size_axis;       // Batch size.
  size1_ = in_shape[axis_];        // Size of specified axis.
  size2_ = size / size0_ / size1_; // Size of rest.

  outputs[0]->reshape(in_shape, true);
}

template <typename T>
void CumSum<T>::forward_impl(const Variables &inputs,
                             const Variables &outputs) {
  typedef typename force_float<T>::type AccumType;

  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  AccumType *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_, true);

  for (int i0 = 0; i0 < size0_; ++i0) {
    for (int i2 = 0; i2 < size2_; ++i2) {
      const int j = i0 * size1_ * size2_ + i2;

      for (int idx = 0; idx < size1_; ++idx) {

        const int i1 = reverse_ ? size1_ - idx - 1 : idx;
        const int y_k = i1 * size2_ + j;
        if (idx == 0) {
          // To prevent accessing out-of-bounds.
          y[y_k] = exclusive_ ? 0 : x[y_k];
          continue;
        }
        const int d = reverse_ ? -1 : 1;
        const int y_k_prev = y_k - d * size2_;
        const int x_k = exclusive_ ? y_k_prev : y_k;

        y[y_k] = y[y_k_prev] + x[x_k];
      }
    }
  }
}

template <typename T>
void CumSum<T>::backward_impl(const Variables &inputs, const Variables &outputs,
                              const vector<bool> &propagate_down,
                              const vector<bool> &accum) {
  if (!(propagate_down[0])) {
    return;
  }
  typedef typename force_float<T>::type AccumType;

  const T *g_y = outputs[0]->get_grad_pointer<T>(this->ctx_);

  if (propagate_down[0]) {
    T *g_x = inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_, !accum[0]);

    for (int i0 = 0; i0 < size0_; ++i0) {
      for (int i2 = 0; i2 < size2_; ++i2) {
        const int j = i0 * size1_ * size2_ + i2;

        AccumType cum_sum = 0.0;
        for (int idx = 0; idx < size1_; ++idx) {

          const int i1 = reverse_ ? idx : size1_ - idx - 1;
          const int x_k = i1 * size2_ + j;

          cum_sum += g_y[x_k];
          auto cur = exclusive_ ? cum_sum - g_y[x_k] : cum_sum;
          if (accum[0])
            g_x[x_k] += cur;
          else
            g_x[x_k] = cur;
        }
      }
    }
  }
}
}