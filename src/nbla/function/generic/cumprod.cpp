// Copyright 2021 Sony Corporation.
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
#include <nbla/function/cumprod.hpp>
#include <nbla/variable.hpp>

#include <iostream>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(CumProd, int, bool, bool);

template <typename T>
void CumProd<T>::setup_impl(const Variables &inputs, const Variables &outputs) {
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
  size0_ = size / size_axis;       // Batch size.
  size1_ = in_shape[axis_];        // Size of specified axis.
  size2_ = size / size0_ / size1_; // Size of rest.

  outputs[0]->reshape(in_shape, true);
}

template <typename T>
void CumProd<T>::forward_impl(const Variables &inputs,
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
          y[y_k] = exclusive_ ? 1 : x[y_k];
          continue;
        }
        const int d = reverse_ ? -1 : 1;
        const int y_k_prev = y_k - d * size2_;
        const int x_k = exclusive_ ? y_k_prev : y_k;

        y[y_k] = y[y_k_prev] * x[x_k];
      }
    }
  }
}

template <typename T>
void CumProd<T>::backward_impl(const Variables &inputs,
                               const Variables &outputs,
                               const vector<bool> &propagate_down,
                               const vector<bool> &accum) {
  if (!(propagate_down[0])) {
    return;
  }
  typedef typename force_float<T>::type AccumType;

  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  const T *y = outputs[0]->get_data_pointer<T>(this->ctx_);
  const T *g_y = outputs[0]->get_grad_pointer<T>(this->ctx_);

  if (propagate_down[0]) {
    T *g_x = inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_, !accum[0]);

    bool zero_input_present = false;
    for (int i = 0; i < inputs[0]->size(); ++i) {
      if (x[i] == (T)0) {
        zero_input_present = true;
        break;
      }
    }

    for (int i0 = 0; i0 < size0_; ++i0) {
      for (int i2 = 0; i2 < size2_; ++i2) {
        const int j = i0 * size1_ * size2_ + i2;

        AccumType cum_sum_ydy = 0.;
        for (int idx = 0; idx < size1_; ++idx) {

          const int i1 = reverse_ ? idx : size1_ - idx - 1;
          const int x_k = i1 * size2_ + j;

          if (zero_input_present) {

            auto coeff = (i1 == 0)
                             ? (T)1
                             : exclusive_ ? y[x_k] : y[(i1 - 1) * size2_ + j];
            if (reverse_)
              coeff = (i1 == size1_ - 1)
                          ? (T)1
                          : exclusive_ ? y[x_k] : y[(i1 + 1) * size2_ + j];

            AccumType cur = exclusive_ ? (T)0 : coeff * g_y[x_k];

            if (reverse_) {
              for (int i4 = i1 - 1; i4 >= 0; --i4) {
                if (!exclusive_ || i4 != i1 - 1)
                  coeff *= (exclusive_ ? x[(i4 + 1) * size2_ + j]
                                       : x[i4 * size2_ + j]);
                cur += coeff * g_y[i4 * size2_ + j];
              }
            } else {
              for (int i4 = i1 + 1; i4 < size1_; ++i4) {
                if (!exclusive_ || i4 != i1 + 1)
                  coeff *= (exclusive_ ? x[(i4 - 1) * size2_ + j]
                                       : x[i4 * size2_ + j]);
                cur += coeff * g_y[i4 * size2_ + j];
              }
            }

            if (accum[0])
              g_x[x_k] += cur;
            else
              g_x[x_k] = cur;
          }

          else {

            cum_sum_ydy += y[x_k] * g_y[x_k];
            auto cur =
                exclusive_ ? cum_sum_ydy - y[x_k] * g_y[x_k] : cum_sum_ydy;
            if (accum[0]) {
              g_x[x_k] += cur / x[x_k];
            } else {
              g_x[x_k] = cur / x[x_k];
            }
          }
        }
      }
    }
  }
}
}
