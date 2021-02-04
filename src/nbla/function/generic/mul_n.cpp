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

#include <memory>
#include <nbla/array.hpp>
#include <nbla/common.hpp>
#include <nbla/function/mul_n.hpp>
#include <nbla/variable.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(MulN);

template <typename T>
void MulN<T>::setup_impl(const Variables &inputs, const Variables &outputs) {
  NBLA_CHECK(inputs.size() >= 2, error_code::value,
             "minimum 2 inputs must be given");
  for (Variables::size_type i = 1; i < inputs.size(); i++) {
    NBLA_CHECK(inputs[0]->shape() == inputs[i]->shape(), error_code::value,
               "shape of all inputs must be shame");
  }
  outputs[0]->reshape(inputs[0]->shape(), true);
}

template <typename T>
void MulN<T>::forward_impl(const Variables &inputs, const Variables &outputs) {
  T *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_, true);
  std::unique_ptr<const T *[]> xs(new const T *[inputs.size()]);
  for (Variables::size_type i = 0; i < inputs.size(); i++) {
    xs[i] = inputs[i]->get_data_pointer<T>(this->ctx_);
  }
  for (int j = 0; j < inputs[0]->size(); j++) {
    T val = (T)(1);
    for (Variables::size_type i = 0; i < inputs.size(); ++i) {
      val *= xs[i][j];
    }
    y[j] = val;
  }
}

template <typename T, bool accum>
void muln_backward_cpu(int size, T *dx, const T *dy, const T *x, const T *y) {
  for (int s = 0; s < size; ++s) {
    if (accum)
      dx[s] += dy[s] * y[s] / x[s];
    else
      dx[s] = dy[s] * y[s] / x[s];
  }
}

template <typename T>
void MulN<T>::backward_impl(const Variables &inputs, const Variables &outputs,
                            const vector<bool> &propagate_down,
                            const vector<bool> &accum) {
  if (!(propagate_down[0] || propagate_down[1])) {
    return;
  }
  const T *dy = outputs[0]->get_grad_pointer<T>(this->ctx_);
  const T *y = outputs[0]->get_data_pointer<T>(this->ctx_);
  Size_t size = inputs[0]->size();
  for (Variables::size_type i = 0; i < inputs.size(); ++i) {
    if (propagate_down[i]) {
      T *dx = inputs[i]->cast_grad_and_get_pointer<T>(this->ctx_, !(accum[i]));
      const T *x = inputs[i]->get_data_pointer<T>(this->ctx_);
      if (accum[i])
        muln_backward_cpu<T, true>(size, dx, dy, x, y);
      else
        muln_backward_cpu<T, false>(size, dx, dy, x, y);
    }
  }
}
}
