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

/** Quantize
 */
#include <nbla/array.hpp>
#include <nbla/function/fixed_point_quantize.hpp>
#include <nbla/variable.hpp>

#include <algorithm>
#include <cmath>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(FixedPointQuantize, bool, int, float, bool);

template <typename T>
void FixedPointQuantize<T>::setup_impl(const Variables &inputs,
                                       const Variables &outputs) {
  NBLA_CHECK(n_ > 0 && delta_ > 0., error_code::value,
             "Both bit width and delta should be positive.");

  // Reshape output size equal to input size
  outputs[0]->reshape(inputs[0]->shape(), true);

  int n = (sign_) ? n_ - 1 : n_;
  max_ = (pow(2, n) - 1) * delta_;
  min_ = (sign_) ? (T)(-max_) : (T)0;

  NBLA_CHECK(n > 0, error_code::value,
             "bit width should be positive when considering sign (1bit).");
}

template <typename T>
void FixedPointQuantize<T>::forward_impl(const Variables &inputs,
                                         const Variables &outputs) {
  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  T *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_, true);

  // Uniform Quantization
  T y_tmp;
  for (int s = 0; s < inputs[0]->size(); s++) {
    if (x[s] > max_) {
      y_tmp = max_;
    } else if (x[s] < min_) {
      y_tmp = min_;
    } else {
      bool sign_x = (x[s] < 0.0);
      T abs_x = std::fabs(x[s]);
      y_tmp = T(int((abs_x / delta_) + 0.5)) * delta_;
      y_tmp = sign_x ? -y_tmp : y_tmp;
    }
    y[s] = y_tmp;
  }
}

// backward core
template <typename T, bool accum>
void quantize_native_backward_cpu(int size, T *dx, const T *dy) {
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
                           const T max, const T min) {
  for (int s = 0; s < size; s++) {
    if (x[s] > max) {
      if (!accum)
        dx[s] = (T)0.;
    } else if (x[s] < min) { // also consider sign or unsign.
      if (!accum)
        dx[s] = (T)0.;
    } else { // non-clipped region
      if (accum) {
        dx[s] += dy[s];
      } else {
        dx[s] = dy[s];
      }
    }
  }
}

template <typename T>
void FixedPointQuantize<T>::backward_impl(const Variables &inputs,
                                          const Variables &outputs,
                                          const vector<bool> &propagate_down,
                                          const vector<bool> &accum) {
  // TODO: consider fine-grained STE
  if (!propagate_down[0]) {
    return;
  }

  Size_t size = inputs[0]->size();
  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  T *dx = inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_, !accum[0]);
  const T *dy = outputs[0]->get_grad_pointer<T>(this->ctx_);

  if (ste_fine_grained_) {
    if (accum[0])
      quantize_backward_cpu<T, true>(size, dx, dy, x, max_, min_);
    else
      quantize_backward_cpu<T, false>(size, dx, dy, x, max_, min_);
  } else {
    if (accum[0])
      quantize_native_backward_cpu<T, true>(size, dx, dy);
    else
      quantize_native_backward_cpu<T, false>(size, dx, dy);
  }
}

} // namespace nbla
