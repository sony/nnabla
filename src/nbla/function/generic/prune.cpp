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

#include <nbla/array.hpp>
#include <nbla/array.hpp>
#include <nbla/common.hpp>
#include <nbla/function/prune.hpp>
#include <nbla/variable.hpp>

#include <algorithm>
#include <cmath>
#include <string.h>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(Prune, float);

template <typename T>
void Prune<T>::setup_impl(const Variables &inputs, const Variables &outputs) {
  NBLA_CHECK(rate_ >= 0 && rate_ <= 1, error_code::value,
             "Rate %f must be in [0, 1].", rate_);

  // threshold to prune
  int x_size = inputs[0]->size();
  thresh_idx_ = (int)((x_size - 1) * this->rate_);

  outputs[0]->reshape(inputs[0]->shape(), true);
}

template <typename T>
void Prune<T>::forward_impl(const Variables &inputs, const Variables &outputs) {
  // this->rate_;
  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  T *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_, true);

  // make buffer
  Size_t size = inputs[0]->size();
  dtypes dtype = get_dtype<T>();
  ArrayPtr array = make_shared<CpuCachedArray>(size, dtype, ctx_);
  auto buffer = array->pointer<T>();

  // copy buffer
  int x_size = inputs[0]->size();
  memcpy((void *)buffer, x, sizeof(T) * x_size);

  // sort
  auto abs_comp = [](T i, T j) { return std::abs(i) < std::abs(j); };
  std::sort(buffer, buffer + x_size, abs_comp);

  // threshold value
  T thresh_val = std::abs(buffer[thresh_idx_]);
  thresh_val += this->rate_ == 1.0 ? 1.0 : 0.0;

  // prune, or pruning
  for (int s = 0; s < inputs[0]->size(); s++) {
    y[s] = (std::abs(x[s]) < thresh_val) ? (T)0 : x[s];
  }
}

template <typename T>
void Prune<T>::backward_impl(const Variables &inputs, const Variables &outputs,
                             const vector<bool> &propagate_down,
                             const vector<bool> &accum) {

  if (!(propagate_down[0])) {
    return;
  }
  // Gradient of outputs
  T *dx = inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_, !accum[0]);
  const T *dy = outputs[0]->get_grad_pointer<T>(this->ctx_);

  for (int s = 0; s < inputs[0]->size(); ++s) {
    if (accum[0])
      dx[s] += dy[s];
    else
      dx[s] = dy[s];
  }
}
}
