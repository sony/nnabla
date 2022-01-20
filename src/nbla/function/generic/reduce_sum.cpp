// Copyright 2018,2019,2020,2021 Sony Corporation.
// Copyright 2021 Sony Group Corporation.
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

/** ReduceSum
 */
#include <nbla/array.hpp>
#include <nbla/function/reduce_sum.hpp>
#include <nbla/variable.hpp>

#include <algorithm>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(ReduceSum);

template <typename T>
void ReduceSum<T>::setup_impl(const Variables &inputs,
                              const Variables &outputs) {
  outputs[0]->reshape(Shape_t(), true);
}

template <typename T>
void ReduceSum<T>::forward_impl(const Variables &inputs,
                                const Variables &outputs) {
  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  T *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_, true);
  T sum = 0;
  for (int i = 0; i < inputs[0]->size(); ++i) {
    sum += x[i];
  }
  *y = sum;
}

template <typename T, bool accum>
void sum_backward_cpu(T *dx, const T *dy, int size) {
  for (int i = 0; i < size; ++i) {
    if (accum)
      dx[i] += (*dy);
    else
      dx[i] = (*dy);
  }
}

template <typename T>
void ReduceSum<T>::backward_impl(const Variables &inputs,
                                 const Variables &outputs,
                                 const vector<bool> &propagate_down,
                                 const vector<bool> &accum) {
  if (!propagate_down[0]) {
    return;
  }
  const T *dy = outputs[0]->get_grad_pointer<T>(this->ctx_);
  T *dx = inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_, !accum[0]);
  const Size_t size = inputs[0]->size();
  if (accum[0])
    sum_backward_cpu<T, true>(dx, dy, size);
  else
    sum_backward_cpu<T, false>(dx, dy, size);
}

} // namespace nbla
