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

// concatenate.cpp

#include <nbla/array.hpp>
#include <nbla/function/concatenate.hpp>
#include <nbla/variable.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(Concatenate, int);

template <typename T>
void Concatenate<T>::setup_impl(const Variables &inputs,
                                const Variables &outputs) {
  Shape_t shape = inputs[0]->shape();
  if (axis_ < 0)
    axis_ += shape.size();

  NBLA_CHECK(axis_ >= 0, error_code::value,
             "axis must not be less than zero, got %d", axis_);
  auto axis = static_cast<Shape_t::size_type>(this->axis_);
  NBLA_CHECK(axis <= shape.size(), error_code::value,
             "axis must be less than or equal to ndim of input. "
             "axis: %d > ndim of input: %d.",
             axis_, shape.size());
  inner_total_size_ = 0;
  for (Variables::size_type i = 0; i < inputs.size(); i++) {
    NBLA_CHECK(inputs[i]->shape().size() != 0, error_code::value,
               "input value(inputs[%d]) does not exist. "
               "inputs[%d]->shape().size(): %d.",
               i, i, inputs[i]->shape().size());
    const int inner_size = inputs[i]->size(this->axis_);
    inner_total_size_ += inner_size;
    if (i >= 1) {
      shape[axis_] += inputs[i]->shape()[axis_]; // Accumulate size along axis
      for (Shape_t::size_type j = 0; j < shape.size(); j++) {
        if (j != axis) {
          NBLA_CHECK(inputs[i]->shape()[j] == shape[j], error_code::value,
                     "Dimensions of inputs must match. "
                     "inputs[%d]->shape()[%d]: %d != "
                     "inputs[0]->shape()[%d]: %d.",
                     i, j, inputs[i]->shape()[j], j, shape[j]);
        }
      }
    }
  }
  outputs[0]->reshape(shape, true);
  outer_size_ = inputs[0]->size() / inputs[0]->size(axis_);
}

template <class T>
void Concatenate<T>::forward_impl(const Variables &inputs,
                                  const Variables &outputs) {
  T *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_, true);
  int inner_offset = 0;
  for (Variables::size_type c = 0; c < inputs.size(); ++c) {
    const T *x = inputs[c]->get_data_pointer<T>(this->ctx_);
    const int inner_size = inputs[c]->size(this->axis_);
    for (int o = 0; o < outer_size_; ++o) {
      for (int i = 0; i < inner_size; ++i) {
        y[o * inner_total_size_ + inner_offset + i] = x[o * inner_size + i];
      }
    }
    inner_offset += inner_size;
  }
}

template <class T>
void Concatenate<T>::backward_impl(const Variables &inputs,
                                   const Variables &outputs,
                                   const vector<bool> &propagate_down,
                                   const vector<bool> &accum) {
  const T *dy = outputs[0]->get_grad_pointer<T>(this->ctx_);
  int inner_offset = 0;

  for (Variables::size_type c = 0; c < inputs.size(); ++c) {
    const int inner_size = inputs[c]->size(this->axis_);
    if (propagate_down[c]) {
      T *dx = inputs[c]->cast_grad_and_get_pointer<T>(this->ctx_, !accum[c]);
      for (int o = 0; o < outer_size_; ++o) {
        for (int i = 0; i < inner_size; ++i) {
          T &rdx = dx[o * inner_size + i];
          rdx = (accum[c] ? rdx : (T)0) +
                dy[o * inner_total_size_ + inner_offset + i];
        }
      }
    }
    inner_offset += inner_size;
  }
}
}
