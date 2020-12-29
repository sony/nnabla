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
#include <nbla/common.hpp>
#include <nbla/function/where.hpp>
#include <nbla/variable.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(Where);

template <typename T>
void Where<T>::setup_impl(const Variables &inputs, const Variables &outputs) {
  auto cshape = inputs[0]->shape();
  auto tshape = inputs[1]->shape();
  auto fshape = inputs[2]->shape();
  NBLA_CHECK(tshape == fshape, error_code::value,
             "x_true and x_false must have same dimensions.");
  NBLA_CHECK(cshape.size() <= tshape.size(), error_code::value,
             "Rank of condition must be less than or equal to that of x_true "
             "or x_false.");
  for (Shape_t::size_type d = 0; d < cshape.size(); d++) {
    NBLA_CHECK(cshape[d] == tshape[d], error_code::value,
               "The first dimensions of x_true and x_false must be the same as "
               "the shape of condition.");
  }
  outputs[0]->reshape(tshape, true);
}

template <typename T>
void Where<T>::forward_impl(const Variables &inputs, const Variables &outputs) {
  const T *condition = inputs[0]->get_data_pointer<T>(this->ctx_);
  const T *x_true = inputs[1]->get_data_pointer<T>(this->ctx_);
  const T *x_false = inputs[2]->get_data_pointer<T>(this->ctx_);
  T *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_, true);

  size_t csize = inputs[0]->size();
  size_t xsize = inputs[1]->size();
  size_t inner_size = xsize / csize;
  for (auto s = decltype(xsize){0}; s < xsize; s++) {
    auto c = s / inner_size;
    y[s] = condition[c] ? x_true[s] : x_false[s];
  }
}

template <typename T>
void Where<T>::backward_impl(const Variables &inputs, const Variables &outputs,
                             const vector<bool> &propagate_down,
                             const vector<bool> &accum) {
  if (!(propagate_down[1] || propagate_down[2])) {
    return;
  }

  const T *g_y = outputs[0]->get_grad_pointer<T>(this->ctx_);
  const T *condition = inputs[0]->get_data_pointer<T>(this->ctx_);
  T *g_x_true{nullptr};
  T *g_x_false{nullptr};

  if (propagate_down[1]) {
    g_x_true = inputs[1]->cast_grad_and_get_pointer<T>(this->ctx_, !accum[1]);
  }
  if (propagate_down[2]) {
    g_x_false = inputs[2]->cast_grad_and_get_pointer<T>(this->ctx_, !accum[2]);
  }
  size_t csize = inputs[0]->size();
  size_t xsize = inputs[1]->size();
  size_t inner_size = xsize / csize;

  for (auto s = decltype(xsize){0}; s < xsize; s++) {
    const bool cond = condition[s / inner_size];
    if (g_x_true) {
      g_x_true[s] = (accum[1] ? g_x_true[s] : (T)0) + (cond ? g_y[s] : (T)0);
    }
    if (g_x_false) {
      g_x_false[s] = (accum[2] ? g_x_false[s] : (T)0) + (cond ? (T)0 : g_y[s]);
    }
  }
}
}
