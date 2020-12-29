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

/** VATNoise
 */
#include <nbla/array.hpp>
#include <nbla/function/vat_noise.hpp>
#include <nbla/variable.hpp>

#include <algorithm>
#include <cmath>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(VATNoise, int, float);

template <typename T>
void VATNoise<T>::setup_impl(const Variables &inputs,
                             const Variables &outputs) {

  NBLA_CHECK(0 < base_axis_, error_code::value,
             "base_axis_ must be grater than 0. base_axis: %d.", base_axis_);

  auto base_axis = static_cast<Shape_t::size_type>(base_axis_);
  NBLA_CHECK(base_axis < inputs[0]->shape().size(), error_code::value,
             "base_axis must be less than ndim of inputs[0]. "
             "base_axis: %d >= ndim of inputs[0]: %d.",
             base_axis_, inputs[0]->shape().size());

  NBLA_CHECK(inputs[0]->shape() == inputs[1]->shape(), error_code::value,
             "Dimensions of inputs must match. "
             "inputs[0]: (%s) != inputs[1]: (%s).",
             string_join(inputs[0]->shape(), string(", ")).c_str(),
             string_join(inputs[1]->shape(), string(", ")).c_str());

  Shape_t shape = inputs[0]->shape();
  outputs[0]->reshape(shape, true);
}

template <typename T>
void VATNoise<T>::forward_impl(const Variables &inputs,
                               const Variables &outputs) {
  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  T *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_, true);

  const int m = inputs[0]->strides()[base_axis_ - 1];
  const int n = inputs[0]->size() / m;

  for (int i = 0; i < n; i++) {
    const T *xi = x + i * m;
    T *yi = y + i * m;
    T sum = 1e-8;
    for (int k = 0; k < m; k++) {
      sum += xi[k] * xi[k];
    }
    const T scale = eps_ / std::sqrt(sum);
    for (int k = 0; k < m; k++) {
      yi[k] = scale * xi[k];
    }
  }
}

template <typename T>
void VATNoise<T>::backward_impl(const Variables &inputs,
                                const Variables &outputs,
                                const vector<bool> &propagate_down,
                                const vector<bool> &accum) {
  const T *dy = outputs[0]->get_grad_pointer<T>(this->ctx_);
  T *w = inputs[1]->cast_data_and_get_pointer<T>(this->ctx_, true);

  for (int i = 0; i < inputs[0]->size(); ++i) {
    w[i] = dy[i] * eps_;
  }
}

} // namespace nbla
