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

/** Embed
 */
#include <nbla/array.hpp>
#include <nbla/function/embed.hpp>
#include <nbla/variable.hpp>

#include <algorithm>
#include <cstring>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(Embed);

template <typename T, typename T1>
void Embed<T, T1>::setup_impl(const Variables &inputs,
                              const Variables &outputs) {
  Shape_t shape_x = inputs[0]->shape();
  Shape_t shape_w = inputs[1]->shape();
  Shape_t shape_y = shape_x;
  shape_y.insert(shape_y.end(), shape_w.begin() + 1, shape_w.end());
  outputs[0]->reshape(shape_y, true);
}

template <typename T, typename T1>
void Embed<T, T1>::forward_impl(const Variables &inputs,
                                const Variables &outputs) {
  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  const T1 *w = inputs[1]->get_data_pointer<T1>(this->ctx_);
  T1 *y = outputs[0]->cast_data_and_get_pointer<T1>(this->ctx_, true);

  Size_t stride0 = inputs[1]->size(1);
  for (int i = 0; i < inputs[0]->size(); ++i) {
    memcpy((void *)(y + i * stride0), w + x[i] * stride0, sizeof(T1) * stride0);
  }
}

template <typename T, typename T1>
void embed_backward_cpu(int size, int stride0, T1 *dw, const T1 *dy,
                        const T *x) {
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < stride0; ++j) {
      dw[x[i] * stride0 + j] += dy[i * stride0 + j];
    }
  }
}

template <typename T, typename T1>
void Embed<T, T1>::backward_impl(const Variables &inputs,
                                 const Variables &outputs,
                                 const vector<bool> &propagate_down,
                                 const vector<bool> &accum) {

  NBLA_CHECK(!propagate_down[0], error_code::value,
             "Index array can not be propagated down.");
  if (!propagate_down[1]) {
    return;
  }
  if (!accum[1])
    inputs[1]->grad()->zero();
  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  T1 *dw = inputs[1]->cast_grad_and_get_pointer<T1>(this->ctx_, false);
  const T1 *dy = outputs[0]->get_grad_pointer<T1>(this->ctx_);
  Size_t stride0 = inputs[1]->size(1);
  embed_backward_cpu(inputs[0]->size(), stride0, dw, dy, x);
}

} // namespace nbla
