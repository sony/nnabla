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

// binary_cross_entropy.cpp

#include <nbla/array.hpp>
#include <nbla/function/binary_cross_entropy.hpp>
#include <nbla/variable.hpp>

#include <algorithm>
#include <cmath>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(BinaryCrossEntropy);

template <typename T>
void BinaryCrossEntropy<T>::setup_impl(const Variables &inputs,
                                       const Variables &outputs) {
  NBLA_CHECK(inputs[0]->shape() == inputs[1]->shape(), error_code::value,
             "Dimensions of inputs must match. "
             "inputs[0]: %s != inputs[1]: %s.",
             string_join(inputs[0]->shape(), string(", ")).c_str(),
             string_join(inputs[1]->shape(), string(", ")).c_str());
  outputs[0]->reshape(inputs[0]->shape(), true);
}

template <typename T>
void BinaryCrossEntropy<T>::forward_impl(const Variables &inputs,
                                         const Variables &outputs) {
  const T *x0 = inputs[0]->get_data_pointer<T>(this->ctx_);
  const T *x1 = inputs[1]->get_data_pointer<T>(this->ctx_);
  T *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_, true);
  const Size_t size = inputs[0]->size();
  for (int s = 0; s < size; s++) {
    y[s] = -(x1[s] * std::log(std::max(x0[s], std::numeric_limits<T>::min())) +
             (1 - x1[s]) *
                 std::log(std::max(1 - x0[s], std::numeric_limits<T>::min())));
  }
}

template <typename T>
void BinaryCrossEntropy<T>::backward_impl(const Variables &inputs,
                                          const Variables &outputs,
                                          const vector<bool> &propagate_down,
                                          const vector<bool> &accum) {
  if (!(propagate_down[0] || propagate_down[1])) {
    return;
  }
  const T *dy = outputs[0]->get_grad_pointer<T>(this->ctx_);
  const T *x0 = inputs[0]->get_data_pointer<T>(this->ctx_);
  const T *x1 = inputs[1]->get_data_pointer<T>(this->ctx_);
  const Size_t size = inputs[0]->size();
  if (propagate_down[0]) {
    T *dx0 = inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_, !accum[0]);
    for (int s = 0; s < size; ++s) {
      dx0[s] =
          (accum[0] ? dx0[s] : (T)0) +
          dy[s] * (x0[s] - x1[s]) /
              std::max(x0[s] - x0[s] * x0[s], std::numeric_limits<T>::min());
    }
  }
  if (propagate_down[1]) {
    T *dx1 = inputs[1]->cast_grad_and_get_pointer<T>(this->ctx_, !accum[1]);
    for (int s = 0; s < size; ++s) {
      dx1[s] =
          (accum[1] ? dx1[s] : (T)0) +
          dy[s] *
              (std::log(std::max(1 - x0[s], std::numeric_limits<T>::min())) -
               std::log(std::max(x0[s], std::numeric_limits<T>::min())));
    }
  }
}
}
