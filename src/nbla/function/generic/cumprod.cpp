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

  for (Size_t i0 = 0; i0 < size0_; ++i0) {
    for (Size_t i2 = 0; i2 < size2_; ++i2) {
      const Size_t j = i0 * size1_ * size2_ + i2;

      for (Size_t idx = 0; idx < size1_; ++idx) {

        const Size_t i1 = reverse_ ? size1_ - idx - 1 : idx;
        const Size_t y_k = i1 * size2_ + j;
        if (idx == 0) {
          // To prevent accessing out-of-bounds.
          y[y_k] = exclusive_ ? 1 : x[y_k];
          continue;
        }
        const Size_t d = reverse_ ? -1 : 1;
        const Size_t y_k_prev = y_k - d * size2_;
        const Size_t x_k = exclusive_ ? y_k_prev : y_k;

        y[y_k] = y[y_k_prev] * x[x_k];
      }
    }
  }
}

/**
 * @brief Calculate cumulative prod backward.
 *
 * Here, we introduce O(N) algorithm used in this implementation with PyThon
 * like pseudocode.
 * The rough idea is splitting `dx` calculation into three parts.
 *
 * e.g) Definition of gradient formula for input_size == 4.
 * \code{.py}
 * dx0 = dy0  +  1 * x1 * dy1  +  1 * x1 * x2 * dy2  +  1 * x1 * x2 * x3 * dy3
 * dx1 =         x0 * 1 * dy1  +  x0 * 1 * x2 * dy2  +  x0 * 1 * x2 * x3 * dy3
 * dx2 =                          x0 * x1 * 1 * dy2  +  x0 * x1 * 1 * x3 * dy3
 * dx3 =                                                x0 * x1 * x2 * 1 * dy3
 * \endcode
 *
 * When `x2 == 0`, the formula above is reduced to following.
 *
 * \code{.py}
 * dx0 = dy0  +  1 * x1 * dy1
 * dx1 =         x0 * 1 * dy1
 * dx2 =                          x0 * x1 * 1 * dy2  +  x0 * x1 * 1 * x3 * dy3
 * dx3 = 0
 * \endcode
 *
 * In general,
 *
 * \code{.py}
 * dxi = reversed_cumsum(cumprod(x) * dy) / x     # For i < first_zero_idx
 * dxi = reversed_cumsum(masked_cumprod(x) * dy)  # For i = first_zero_idx
 * dxi = 0                                        # For i > first_zero_idx
 * \endcode
 *
 * `first_zero_idx` is the smallest i such that `x[i] == 0` and masked_cumprod
 * is a cumprod of modified `x` as `x[first_zero_idx] == 1`.
 * Each case calculation is performed at the same time in O(N) like following
 * pseudocode.
 *
 * \code{.py}
 * # n: Array size
 * # x: Input
 * # dx: Input grad
 * # dy: Output grad
 * # first_zero_idx: Smallest i such that x[i] == 0 (n for x[i] != 0)
 * # masked_cumprod: Cumulative prod of x where x[first_zero_idx] == 1 if
 * first_zero_idx != n
 *
 * # We need to pre-compute `masked_cumprod`.
 *
 * sum = 0
 * fot i = n-1; i >= 0; i--:
 *   if i >= first_zero_idx:
 *     sum += masked_cumprod[i] * dy[i]
 *
 *   if i > first_zero_idx:
 *     dx[i] = 0
 *   else if i == first_zero_idx:
 *     dx[i] = sum;
 *     sum = 0
 *   else: # i < first_zero_idx
 *     sum += masked_cumprod[i] * dy[i]
 *     dx[i] = sum / x[i]
 * \endcode
 */
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
  const T *g_y = outputs[0]->get_grad_pointer<T>(this->ctx_);
  T *g_x = inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_, !accum[0]);

  Variable v_masked_cumprod({size1_});
  AccumType *masked_cumprod =
      v_masked_cumprod.cast_data_and_get_pointer<AccumType>(this->ctx_, true);

  for (Size_t i0 = 0; i0 < size0_; ++i0) {
    for (Size_t i2 = 0; i2 < size2_; ++i2) {
      const Size_t offset = i0 * size1_ * size2_ + i2;

      // Create masked_cumprod
      Size_t first_zero_pos = size1_;
      T prod = (T)1;
      for (Size_t k = 0; k < size1_; k++) {
        const Size_t i1 = reverse_ ? size1_ - k - 1 : k;
        Size_t idx = i1 * size2_ + offset;
        if (x[idx] == (T)0 && first_zero_pos == size1_) {
          first_zero_pos = k;
          // prod *= (T)1;
        } else {
          prod *= x[idx];
        }
        masked_cumprod[k] = prod;
      }

      // Calculate gradient
      AccumType sum = 0;
      for (Size_t k = size1_ - 1; k >= 0; k--) {
        const Size_t i1 = reverse_ ? size1_ - k - 1 : k;
        Size_t idx = i1 * size2_ + offset;

        if (!exclusive_) {
          sum += masked_cumprod[k] * g_y[idx];
        }

        T grad;
        if (k == first_zero_pos) {
          grad = (T)sum;
          sum = 0;
        } else if (k > first_zero_pos) {
          grad = (T)0;
        } else {
          grad = (T)sum / x[idx];
        }
        g_x[idx] = grad + (accum[0] ? g_x[idx] : (T)0);

        if (exclusive_ && k != 0) {
          sum += masked_cumprod[k - 1] * g_y[idx];
        }
      }
    }
  }
}
}
