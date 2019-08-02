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

// softmax.cpp

#include <nbla/array.hpp>
#include <nbla/function/softmax.hpp>
#include <nbla/variable.hpp>

#include <algorithm>
#include <cmath>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(Softmax, int);

template <typename T>
void Softmax<T>::setup_impl(const Variables &inputs, const Variables &outputs) {
  Shape_t in_shape = inputs[0]->shape();
  NBLA_CHECK(axis_ < in_shape.size(), error_code::value,
             "axis must be less than ndim of inputs[0]. "
             "axis: %d >= ndim of inputs[0]: %d.",
             axis_, in_shape.size());
  outputs[0]->reshape(in_shape, true);
  Size_t size = inputs[0]->size();
  Size_t size_axis = inputs[0]->size(axis_);
  size0_ = size / size_axis;          // Batch size.
  size1_ = inputs[0]->shape()[axis_]; // Size of specified axis.
  size2_ = size / size0_ / size1_;    // Size of rest.
  NBLA_CHECK(size0_ * size1_ * size2_ == size, error_code::unclassified,
             "An error occurred during setup Softmax function.");
}

template <class T>
void Softmax<T>::forward_impl(const Variables &inputs,
                              const Variables &outputs) {
  typedef typename force_float<T>::type AccumType;
  // Setting up variables
  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  T *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_, true);
  for (int i0 = 0; i0 < size0_; ++i0) {
    for (int i2 = 0; i2 < size2_; ++i2) {
      const int j = i0 * size1_ * size2_ + i2;
      // compute maximum
      T max_x = x[j];
      for (int i1 = 0; i1 < size1_; ++i1) {
        const int k = i1 * size2_ + j;
        max_x = (max_x >= x[k]) ? max_x : x[k];
      }
      // Compute exponential and sum
      AccumType exp_sum = 0;
      for (int i1 = 0; i1 < size1_; ++i1) {
        const int k = i1 * size2_ + j;
        const T tmp = std::exp(x[k] - max_x);
        y[k] = tmp;
        exp_sum += tmp;
      }
      // Compute softmax
      for (int i1 = 0; i1 < size1_; ++i1) {
        const int k = i1 * size2_ + j;
        y[k] = y[k] / exp_sum;
      }
    }
  }
}

template <class T>
void Softmax<T>::backward_impl(const Variables &inputs,
                               const Variables &outputs,
                               const vector<bool> &propagate_down,
                               const vector<bool> &accum) {
  if (!propagate_down[0]) {
    return;
  }
  typedef typename force_float<T>::type AccumType;
  // Setting up variables
  const T *y = outputs[0]->get_data_pointer<T>(this->ctx_);
  const T *dy = outputs[0]->get_grad_pointer<T>(this->ctx_);
  T *dx = inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_, !accum[0]);

  for (int i0 = 0; i0 < size0_; ++i0) {
    for (int i2 = 0; i2 < size2_; ++i2) {
      const int j = i0 * size1_ * size2_ + i2;
      // compute sum of dy * y
      AccumType dyy_sum = 0;
      for (int i1 = 0; i1 < size1_; ++i1) {
        const int k = i1 * size2_ + j;
        dyy_sum += dy[k] * y[k];
      }
      // Compute backward
      for (int i1 = 0; i1 < size1_; ++i1) {
        const int k = i1 * size2_ + j;
        dx[k] = (accum[0] ? dx[k] : (T)0) + y[k] * (dy[k] - dyy_sum);
      }
    }
  }
}
}
