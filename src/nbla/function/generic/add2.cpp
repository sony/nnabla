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

// relu.cpp

#include <nbla/array.hpp>
#include <nbla/function/add2.hpp>
#include <nbla/function/bc_add2.hpp>
#include <nbla/variable.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(Add2, bool);

template <typename T>
void Add2<T>::setup_impl(const Variables &inputs, const Variables &outputs) {
  if (inputs[0]->shape() == inputs[1]->shape()) {
    outputs[0]->reshape(inputs[0]->shape(), true);
    if (inplace_) {
      outputs[0]->data()->set_array(inputs[0]->data()->array());
    }
    return;
  }
  // Trying to fallback to broadcastable Add2.
  this->fall_back_func_ = create_BcAdd2(this->ctx_, inplace_);
  this->fall_back_func_->setup(inputs, outputs);
}

template <class T>
void Add2<T>::forward_impl(const Variables &inputs, const Variables &outputs) {
  const T *x0 = inputs[0]->get_data_pointer<T>(this->ctx_);
  const T *x1 = inputs[1]->get_data_pointer<T>(this->ctx_);
  T *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_, !inplace_);
  for (int s = 0; s < inputs[0]->size(); s++) {
    y[s] = x0[s] + x1[s];
  }
}
template <typename T, bool accum>
void add2_backward_cpu(int size, T *dx, const T *dy) {
  for (int s = 0; s < size; ++s) {
    if (accum)
      dx[s] += dy[s];
    else
      dx[s] = dy[s];
  }
}

template <class T>
void Add2<T>::backward_impl(const Variables &inputs, const Variables &outputs,
                            const vector<bool> &propagate_down,
                            const vector<bool> &accum) {
  if (!(propagate_down[0] || propagate_down[1])) {
    return;
  }
  const T *dy = outputs[0]->get_grad_pointer<T>(this->ctx_);
  Size_t size = inputs[0]->size();

  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      T *dx = inputs[i]->cast_grad_and_get_pointer<T>(this->ctx_,
                                                      !(i == 0 || accum[i]));
      if (accum[i])
        add2_backward_cpu<T, true>(size, dx, dy);
      else
        add2_backward_cpu<T, false>(size, dx, dy);
    }
  }
}
}
