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

/** Pow2Quantize
 */
#include <nbla/array.hpp>
#include <nbla/function/pow2_quantize.hpp>
#include <nbla/variable.hpp>

#include <algorithm>
#include <math.h>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(Pow2Quantize, bool, bool, int, int, bool);

template <typename T>
void Pow2Quantize<T>::setup_impl(const Variables &inputs,
                                 const Variables &outputs) {
  NBLA_CHECK(n_ > 0, error_code::value, "bit width should be positive.");
  // Reshape output size equal to input size
  outputs[0]->reshape(inputs[0]->shape(), true);

  int n = (sign_) ? n_ - 1 : n_;
  n = with_zero_ ? n - 1 : n;
  p_max_ = pow(2., m_);
  p_min_ = pow(2., m_ - ((1 << n) - 1));
  pruning_threshold_ = p_min_ * pow(2., -0.5);

  NBLA_CHECK(n > 0, error_code::value, "bit width should be positive when "
                                       "considering zero (1bit) and sign "
                                       "(1bit).");
}

template <typename T>
void Pow2Quantize<T>::forward_impl(const Variables &inputs,
                                   const Variables &outputs) {
  // TODO: consider arithmetic mean
  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  T *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_, true);

  // Power of 2 Quantization
  T q;
  bool sign_x;
  for (int s = 0; s < inputs[0]->size(); s++) {
    // quantize in positive domain
    T x_abs = std::fabs(x[s]);
    q = std::pow((T)2., std::round(std::log2(x_abs)));
    if (q > p_max_) {
      q = p_max_;
    } else if (q < p_min_ && with_zero_) {
      q = x_abs < pruning_threshold_ ? (T)0 : (T)p_min_;
    } else if (q < p_min_) {
      q = p_min_;
    }

    // address sign
    sign_x = (x[s] < 0.0);
    if (sign_) {
      q = sign_x ? -q : q;
    } else {
      if (with_zero_) {
        q = sign_x ? (T)0 : q;
      } else {
        q = sign_x ? (T)p_min_ : q;
      }
    }
    y[s] = q;
  }
}

// backward core
template <typename T, bool accum>
void quantize_naive_backward_cpu(int size, T *dx, const T *dy) {
  for (int s = 0; s < size; ++s) {
    if (accum) {
      dx[s] += dy[s];
    } else {
      dx[s] = dy[s];
    }
  }
}

// backward core
template <typename T, bool accum>
void quantize_backward_cpu(int size, T *dx, const T *dy, const T *x,
                           const bool sign, const bool with_zero, const T p_max,
                           const T p_min, const T pruning_threshold) {
  T q;
  T x_abs;
  T c;
  for (int s = 0; s < size; s++) {
    x_abs = std::fabs(x[s]);
    ;
    q = std::pow((T)2., std::round(std::log2(x_abs)));
    c = 1.; // normally, assume grad is 1
    if (q > p_max) {
      c = 0.;
    }

    // address sign
    if (!sign) {
      bool sign_x;
      sign_x = (x[s] < 0.0);
      c = sign_x ? (T)0 : c;
    }

    if (accum) {
      dx[s] += c * dy[s];
    } else {
      dx[s] = c * dy[s];
    }
  }
}

template <typename T>
void Pow2Quantize<T>::backward_impl(const Variables &inputs,
                                    const Variables &outputs,
                                    const vector<bool> &propagate_down,
                                    const vector<bool> &accum) {
  // TODO: consider fine-grained STE
  if (!propagate_down[0]) {
    return;
  }

  Size_t size = inputs[0]->size();
  const T *x = inputs[0]->cast_data_and_get_pointer<T>(this->ctx_);
  T *dx = inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_, !accum[0]);
  const T *dy = outputs[0]->get_grad_pointer<T>(this->ctx_);

  if (ste_fine_grained_) {
    if (accum[0])
      quantize_backward_cpu<T, true>(size, dx, dy, x, sign_, with_zero_, p_max_,
                                     p_min_, pruning_threshold_);
    else
      quantize_backward_cpu<T, false>(size, dx, dy, x, sign_, with_zero_,
                                      p_max_, p_min_, pruning_threshold_);
  } else {
    if (accum[0])
      quantize_naive_backward_cpu<T, true>(size, dx, dy);
    else
      quantize_naive_backward_cpu<T, false>(size, dx, dy);
  }
}

} // namespace nbla
