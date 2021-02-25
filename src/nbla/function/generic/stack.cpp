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

// stack.cpp

#include <nbla/array.hpp>
#include <nbla/function/stack.hpp>
#include <nbla/variable.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(Stack, int);

template <typename T>
void Stack<T>::setup_impl(const Variables &inputs, const Variables &outputs) {
  const Shape_t in_shape = inputs[0]->shape();
  if (axis_ < 0)
    axis_ += in_shape.size() + 1;

  NBLA_CHECK(axis_ >= 0, error_code::value,
             "axis must not be less than zero, got %d", axis_);
  NBLA_CHECK(static_cast<Shape_t::size_type>(axis_) <= in_shape.size(),
             error_code::value,
             "axis must be less than or equal to ndim of input. "
             "axis: %d > ndim of inputs[0]: %d.",
             axis_, in_shape.size());
  num_inputs_ = inputs.size();
  for (int i = 1; i < num_inputs_; i++) {
    NBLA_CHECK(inputs[i]->shape() == in_shape, error_code::value,
               "All inputs must be the same size. "
               "inputs[%d] shape: (%s) != inputs[0] shape: (%s).",
               i, string_join(inputs[i]->shape(), string(", ")).c_str(),
               string_join(in_shape, string(", ")).c_str());
  }
  Shape_t out_shape = inputs[0]->shape();
  out_shape.insert(out_shape.begin() + axis_, num_inputs_);
  outputs[0]->reshape(out_shape, true);
  inner_size_ = inputs[0]->size(axis_);
  outer_size_ = inputs[0]->size() / inner_size_;
}

template <class T>
void Stack<T>::forward_impl(const Variables &inputs, const Variables &outputs) {
  T *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_, true);
  for (int i0 = 0; i0 < num_inputs_; ++i0) {
    const T *x = inputs[i0]->get_data_pointer<T>(this->ctx_);
    for (int i1 = 0; i1 < outer_size_; ++i1) {
      for (int i2 = 0; i2 < inner_size_; ++i2) {
        y[i1 * (inner_size_ * num_inputs_) + i0 * inner_size_ + i2] =
            x[i1 * inner_size_ + i2];
      }
    }
  }
}

template <class T>
void Stack<T>::backward_impl(const Variables &inputs, const Variables &outputs,
                             const vector<bool> &propagate_down,
                             const vector<bool> &accum) {
  const T *dy = outputs[0]->get_grad_pointer<T>(this->ctx_);
  for (int i0 = 0; i0 < num_inputs_; ++i0) {
    if (propagate_down[i0]) {
      T *dx = inputs[i0]->cast_grad_and_get_pointer<T>(this->ctx_, !accum[i0]);
      for (int i1 = 0; i1 < outer_size_; ++i1) {
        for (int i2 = 0; i2 < inner_size_; ++i2) {
          T &rdx = dx[i1 * inner_size_ + i2];
          if (accum[i0])
            rdx += dy[i1 * (inner_size_ * num_inputs_) + i0 * inner_size_ + i2];
          else
            rdx = dy[i1 * (inner_size_ * num_inputs_) + i0 * inner_size_ + i2];
        }
      }
    }
  }
}
}
