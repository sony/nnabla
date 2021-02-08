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
#include <nbla/function/top_n_error.hpp>
#include <nbla/variable.hpp>

#include <algorithm>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(TopNError, int, int);

template <typename T, typename T1>
void TopNError<T, T1>::setup_impl(const Variables &inputs,
                                  const Variables &outputs) {
  Shape_t in_shape = inputs[0]->shape();
  Shape_t label_shape = inputs[1]->shape();
  if (axis_ < 0)
    axis_ += in_shape.size();
  NBLA_CHECK(axis_ >= 0, error_code::value,
             "axis must not be less than zero, got %d", axis_);
  NBLA_CHECK(static_cast<Shape_t::size_type>(axis_) < in_shape.size(),
             error_code::value, "axis must be less than ndim of inputs[0]. "
                                "axis: %d >= ndim of inputs[0]: %d.",
             axis_, in_shape.size());
  NBLA_CHECK(label_shape.size() == in_shape.size(), error_code::value,
             "The length of each input dimension must match. "
             "inputs[1] length: %d != inputs[0] length: %d.",
             label_shape.size(), in_shape.size());
  for (int axis = 0; static_cast<size_t>(axis) < label_shape.size(); ++axis) {
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

  Size_t size = inputs[0]->size();
  Size_t size_axis = inputs[0]->size(axis_);
  size0_ = size / size_axis;          // Batch size.
  size1_ = inputs[0]->shape()[axis_]; // Size of specified axis.
  size2_ = size / size0_ / size1_;    // Size of rest.
  NBLA_CHECK(size0_ * size1_ * size2_ == size, error_code::unclassified,
             "An error occurred during setup TopNError function.");
}

template <typename T, typename T1>
void TopNError<T, T1>::forward_impl(const Variables &inputs,
                                    const Variables &outputs) {
  // Setting up variables
  const T *p = inputs[0]->get_data_pointer<T>(this->ctx_);
  const T1 *l = inputs[1]->get_data_pointer<T1>(this->ctx_);
  T *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_, true);

  for (int i0 = 0; i0 < size0_; ++i0) {
    for (int i2 = 0; i2 < size2_; ++i2) {
      const int j = i0 * size2_ + i2;
      T1 label = l[j];
      const int k = i0 * size1_ * size2_ + i2;
      const T threshold = p[k + label * size2_];
      int count = 0;
      for (int i1 = 0; i1 < size1_; ++i1) {
        count += p[k + i1 * size2_] >= threshold;
      }
      y[j] = count > n_;
    }
  }
}

template <typename T, typename T1>
void TopNError<T, T1>::backward_impl(const Variables &inputs,
                                     const Variables &outputs,
                                     const vector<bool> &propagate_down,
                                     const vector<bool> &accum) {
  // Not supported
}

} // namespace nbla
