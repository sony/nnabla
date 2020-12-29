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

/** ClipGradByValue
 */
#include <nbla/array.hpp>
#include <nbla/function/clip_grad_by_value.hpp>
#include <nbla/variable.hpp>

#include <algorithm>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(ClipGradByValue);

template <typename T>
void ClipGradByValue<T>::setup_impl(const Variables &inputs,
                                    const Variables &outputs) {

  // Shape size check
  Shape_t shape0 = inputs[0]->shape();
  Shape_t shape1 = inputs[1]->shape();
  Shape_t shape2 = inputs[2]->shape();
  NBLA_CHECK(shape0.size() && shape1.size() && shape2.size(), error_code::value,
             "Dimensions differ %d, %d, and %d", shape0.size(), shape1.size(),
             shape2.size());

  // Shape check
  for (Shape_t::size_type i = 0; i < shape0.size(); i++) {
    NBLA_CHECK(shape0[i] && shape1[i] && shape2[i], error_code::value,
               "Size at shape[%d] differs %d, %d, and %d", i, shape0[i],
               shape1[i], shape2[i])
  }

  outputs[0]->reshape(inputs[0]->shape(), true);
}

template <typename T>
void ClipGradByValue<T>::forward_impl(const Variables &inputs,
                                      const Variables &outputs) {
  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  T *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_, true);
  for (int i = 0; i < inputs[0]->size(); i++) {
    y[i] = x[i];
  }
}

template <typename T, bool accum>
void clip_grad_by_value_backward_cpu(int size, T *dx, const T *dy, const T *min,
                                     const T *max) {
  for (int i = 0; i < size; i++) {
    T min_i = min[i];
    T max_i = max[i];
    T value;
    if (dy[i] > max_i) {
      value = max_i;
    } else if (dy[i] < min_i) {
      value = min_i;
    } else {
      value = dy[i];
    }
    if (accum) {
      dx[i] += value;
    } else {
      dx[i] = value;
    }
  }
}

template <typename T>
void ClipGradByValue<T>::backward_impl(const Variables &inputs,
                                       const Variables &outputs,
                                       const vector<bool> &propagate_down,
                                       const vector<bool> &accum) {
  // No backward to min and max variables.
  if (!propagate_down[0]) {
    return;
  }

  // Zeroing grads of min and max when accum is false.
  for (int i = 1; i < 3; i++) {
    if (propagate_down[i] && !accum[i]) {
      inputs[i]->grad()->zero();
    }
  }

  Size_t size = inputs[0]->size();
  T *dx = inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_, !accum[0]);
  const T *dy = outputs[0]->get_grad_pointer<T>(this->ctx_);
  const T *min = inputs[1]->get_data_pointer<T>(this->ctx_);
  const T *max = inputs[2]->get_data_pointer<T>(this->ctx_);

  if (accum[0])
    clip_grad_by_value_backward_cpu<T, true>(size, dx, dy, min, max);
  else
    clip_grad_by_value_backward_cpu<T, false>(size, dx, dy, min, max);
}

} // namespace nbla
