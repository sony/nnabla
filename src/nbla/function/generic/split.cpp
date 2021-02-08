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

// split.cpp

#include <nbla/array.hpp>
#include <nbla/function/split.hpp>
#include <nbla/variable.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(Split, int);

template <typename T>
void Split<T>::setup_impl(const Variables &inputs, const Variables &outputs) {
  const Shape_t in_shape = inputs[0]->shape();
  if (axis_ < 0)
    axis_ += in_shape.size();

  NBLA_CHECK(axis_ >= 0, error_code::value,
             "axis must not be less than zero, got %d", axis_);
  NBLA_CHECK(static_cast<Shape_t::size_type>(axis_) < in_shape.size(),
             error_code::value, "axis must be less than ndim of inputs[0]. "
                                "axis: %d >= ndim of inputs[0]: %d.",
             axis_, in_shape.size());
  num_outputs_ = in_shape[axis_];
  NBLA_CHECK(static_cast<Shape_t::size_type>(num_outputs_) == outputs.size(),
             error_code::value,
             "inputs[0].shape[axis] must be the same number as the outputs. "
             "inputs[0].shape[axis]: %d, outputs: %d.",
             num_outputs_, outputs.size());
  Shape_t out_shape = in_shape;
  out_shape.erase(out_shape.begin() + axis_);
  for (int i = 0; i < num_outputs_; i++) {
    outputs[i]->reshape(out_shape, true);
  }
  inner_size_ = outputs[0]->size(axis_);
  NBLA_CHECK(inner_size_ != 0, error_code::unclassified,
             "Zero is specified as the input value.");
  outer_size_ = outputs[0]->size() / inner_size_;
  NBLA_CHECK(inner_size_ * num_outputs_ * outer_size_ == inputs[0]->size(),
             error_code::unclassified,
             "An error occurred during setup Split function.");
}

template <class T>
void Split<T>::forward_impl(const Variables &inputs, const Variables &outputs) {
  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  for (int i0 = 0; i0 < num_outputs_; ++i0) {
    T *y = outputs[i0]->cast_data_and_get_pointer<T>(this->ctx_, true);
    for (int i1 = 0; i1 < outer_size_; ++i1) {
      for (int i2 = 0; i2 < inner_size_; ++i2) {
        y[i1 * inner_size_ + i2] =
            x[i1 * (inner_size_ * num_outputs_) + i0 * inner_size_ + i2];
      }
    }
  }
}
template <typename T, bool accum>
void split_backward_cpu(int outer_size, int inner_size, int num_outputs, int i0,
                        T *dx, const T *dy) {
  for (int i1 = 0; i1 < outer_size; ++i1) {
    for (int i2 = 0; i2 < inner_size; ++i2) {
      T &rdx = dx[i1 * (inner_size * num_outputs) + i0 * inner_size + i2];
      if (accum)
        rdx += dy[i1 * inner_size + i2];
      else
        rdx = dy[i1 * inner_size + i2];
    }
  }
}
template <class T>
void Split<T>::backward_impl(const Variables &inputs, const Variables &outputs,
                             const vector<bool> &propagate_down,
                             const vector<bool> &accum) {
  if (!propagate_down[0]) {
    return;
  }
  T *dx = inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_, !accum[0]);
  for (int i0 = 0; i0 < num_outputs_; ++i0) {
    const T *dy = outputs[i0]->get_grad_pointer<T>(this->ctx_);
    if (accum[0])
      split_backward_cpu<T, true>(outer_size_, inner_size_, num_outputs_, i0,
                                  dx, dy);
    else
      split_backward_cpu<T, false>(outer_size_, inner_size_, num_outputs_, i0,
                                   dx, dy);
  }
}
}
