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

/** LeakyReLU
 */

#include <nbla/array.hpp>
#include <nbla/function/leaky_relu.hpp>
#include <nbla/variable.hpp>

#include <algorithm>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(LeakyReLU, float, bool);

template <typename T>
void LeakyReLU<T>::setup_impl(const Variables &inputs,
                              const Variables &outputs) {
  outputs[0]->reshape(inputs[0]->shape(), true);
  if (inplace_) {
    NBLA_CHECK(
        alpha_ > 0, error_code::value,
        "Alpha must be greater than zero with inplace option being true.");
    outputs[0]->data()->set_array(inputs[0]->data()->array());
  }
}

template <class T>
void LeakyReLU<T>::forward_impl(const Variables &inputs,
                                const Variables &outputs) {
  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  T *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_, !inplace_);
  for (int s = 0; s < inputs[0]->size(); s++) {
    T x_s = x[s];
    if (x_s > (T)0.)
      y[s] = x_s;
    else
      y[s] = alpha_ * x_s;
  }
}
template <typename T, bool accum>
void leaky_relu_backward_cpu(int size, float alpha, T *dx, const T *dy,
                             const T *x) {
  for (int s = 0; s < size; ++s) {
    if (accum) {
      if (x[s] > (T)0.)
        dx[s] += dy[s];
      else
        dx[s] += alpha * dy[s];
    } else {
      if (x[s] > (T)0.)
        dx[s] = dy[s];
      else
        dx[s] = alpha * dy[s];
    }
  }
}

template <class T>
void LeakyReLU<T>::backward_impl(const Variables &inputs,
                                 const Variables &outputs,
                                 const vector<bool> &propagate_down,
                                 const vector<bool> &accum) {
  if (!propagate_down[0]) {
    return;
  }
  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  T *dx = inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_, !accum[0]);
  const T *dy = outputs[0]->get_grad_pointer<T>(this->ctx_);
  if (accum[0])
    leaky_relu_backward_cpu<T, true>(inputs[0]->size(), alpha_, dx, dy, x);
  else
    leaky_relu_backward_cpu<T, false>(inputs[0]->size(), alpha_, dx, dy, x);
}
}
