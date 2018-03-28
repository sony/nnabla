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

/** SELU
 */
#include <nbla/array.hpp>
#include <nbla/function/selu.hpp>
#include <nbla/variable.hpp>

#include <algorithm>
#include <cmath>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(SELU, double, double);

template <typename T>
void SELU<T>::setup_impl(const Variables &inputs, const Variables &outputs) {
  outputs[0]->reshape(inputs[0]->shape(), true);
}

template <typename T>
void SELU<T>::forward_impl(const Variables &inputs, const Variables &outputs) {
  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  T *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_, true);
  const T coef = alpha_ * scale_;
  for (int s = 0; s < inputs[0]->size(); s++) {
    y[s] = x[s] > (T)0 ? (T)scale_ * x[s] : (T)coef * (std::exp(x[s]) - (T)1);
  }
}

template <typename T>
void SELU<T>::backward_impl(const Variables &inputs, const Variables &outputs,
                            const vector<bool> &propagate_down,
                            const vector<bool> &accum) {
  if (!propagate_down[0]) {
    return;
  }
  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  T *dx = inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_, !accum[0]);
  const T *dy = outputs[0]->get_grad_pointer<T>(this->ctx_);
  const T coef = alpha_ * scale_;
  if (accum[0]) {
    for (int s = 0; s < inputs[0]->size(); ++s) {
      dx[s] +=
          (x[s] > (T)0 ? (T)scale_ * dy[s] : (T)coef * std::exp(x[s]) * dy[s]);
    }
  } else {
    for (int s = 0; s < inputs[0]->size(); ++s) {
      dx[s] =
          (x[s] > (T)0 ? (T)scale_ * dy[s] : (T)coef * std::exp(x[s]) * dy[s]);
    }
  }
}

} // namespace nbla
