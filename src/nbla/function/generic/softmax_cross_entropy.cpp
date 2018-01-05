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

// softmax_cross_entropy.cpp

#include <nbla/array.hpp>
#include <nbla/function/softmax.hpp>
#include <nbla/function/softmax_cross_entropy.hpp>
#include <nbla/variable.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(SoftmaxCrossEntropy, int);

template <typename T, typename Tl>
void SoftmaxCrossEntropy<T, Tl>::setup_impl(const Variables &inputs,
                                            const Variables &outputs) {

  Shape_t in_shape = inputs[0]->shape();
  Shape_t label_shape = inputs[1]->shape();
  NBLA_CHECK(axis_ < in_shape.size(), error_code::value,
             "axis must be less than ndim of inputs[0]. "
             "axis: %d >= ndim of inputs[0]: %d.",
             axis_, in_shape.size());
  NBLA_CHECK(label_shape.size() == in_shape.size(), error_code::value,
             "The length of each input dimension must match. "
             "inputs[1] length: %d != inputs[0] length: %d.",
             label_shape.size(), in_shape.size());
  for (int axis = 0; axis < label_shape.size(); ++axis) {
    if (axis == axis_) {
      NBLA_CHECK(label_shape[axis] == 1, error_code::value,
                 "Dimensions of axis of inputs[1] must be 1. %d != 1.",
                 label_shape[axis]);
      continue;
    }
    NBLA_CHECK(label_shape[axis] == in_shape[axis], error_code::value,
               "Dimensions of inputs must match except axis. "
               "label_shape[axis]: %d != in_shape[axis]: %d.",
               label_shape[axis], in_shape[axis]);
  }
  outputs[0]->reshape(label_shape, true);

  softmax_ = create_Softmax(ctx_, axis_);
  softmax_->setup(Variables{inputs[0]}, Variables{&softmax_output_});

  Size_t size = inputs[0]->size();
  Size_t size_axis = inputs[0]->size(axis_);
  size0_ = size / size_axis;          // Batch size.
  size1_ = inputs[0]->shape()[axis_]; // Size of specified axis.
  size2_ = size / size0_ / size1_;    // Size of rest.
  NBLA_CHECK(size0_ * size1_ * size2_ == size, error_code::unclassified,
             "An error occurred during setup SoftmaxCrossEntropy function.");
}

template <typename T, typename Tl>
void SoftmaxCrossEntropy<T, Tl>::forward_impl(const Variables &inputs,
                                              const Variables &outputs) {
  softmax_->forward(Variables{inputs[0]}, Variables{&softmax_output_});
  // Setting up variables
  const T *p = softmax_output_.get_data_pointer<T>(this->ctx_);
  const Tl *l = inputs[1]->get_data_pointer<Tl>(this->ctx_);
  T *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_);

  for (int i0 = 0; i0 < size0_; ++i0) {
    for (int i2 = 0; i2 < size2_; ++i2) {
      const int j = i0 * size2_ + i2;
      Tl label = l[j];
      const int k = i0 * size1_ * size2_ + label * size2_ + i2;
      y[j] = -std::log(std::max(p[k], std::numeric_limits<T>::min()));
    }
  }
}

template <typename T, typename Tl>
void SoftmaxCrossEntropy<T, Tl>::backward_impl(
    const Variables &inputs, const Variables &outputs,
    const vector<bool> &propagate_down, const vector<bool> &accum) {
  NBLA_CHECK(!propagate_down[1], error_code::value,
             "Label can not be propagated down.");
  if (!propagate_down[0])
    return;

  const T *p = softmax_output_.get_data_pointer<T>(this->ctx_);
  const T *dy = outputs[0]->get_grad_pointer<T>(this->ctx_);
  const Tl *l = inputs[1]->get_data_pointer<Tl>(this->ctx_);
  T *dx = inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_);
  if (!accum[0])
    memset(dx, 0, sizeof(*dx) * inputs[0]->size());
  for (int i0 = 0; i0 < size0_; ++i0) {
    for (int i2 = 0; i2 < size2_; ++i2) {
      const int j = i0 * size2_ + i2;
      Tl label = l[j];
      T grad = dy[j];
      for (int i1 = 0; i1 < size1_; ++i1) {
        const int k = i0 * size1_ * size2_ + i1 * size2_ + i2;
        // dx[k] = beta * dx[k] + grad * (p[k] - static_cast<int>(label == i1));
        dx[k] += grad * (p[k] - static_cast<int>(label == i1));
      }
    }
  }
}

}
