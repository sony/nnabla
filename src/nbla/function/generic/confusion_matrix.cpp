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

/** ConfusionMatrix
 */
#include <nbla/array.hpp>
#include <nbla/function/confusion_matrix.hpp>
#include <nbla/variable.hpp>

#include <algorithm>
#include <cstring>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(ConfusionMatrix, int);

template <typename T, typename T1>
void ConfusionMatrix<T, T1>::setup_impl(const Variables &inputs,
                                        const Variables &outputs) {
  Shape_t in_shape = inputs[0]->shape();
  Shape_t label_shape = inputs[1]->shape();
  if (axis_ < 0)
    axis_ += in_shape.size();
  NBLA_CHECK(axis_ >= 0, error_code::value,
             "axis must not be less than zero, got %d", axis_);
  auto axis = static_cast<Shape_t::size_type>(this->axis_);
  NBLA_CHECK(axis < in_shape.size(), error_code::value,
             "axis must be less than ndim of inputs[0]. "
             "axis: %d >= ndim of inputs[0]: %d.",
             axis_, in_shape.size());
  NBLA_CHECK(label_shape.size() == in_shape.size(), error_code::value,
             "The length of each input dimension must match. "
             "inputs[1] length: %d != inputs[0] length: %d.",
             label_shape.size(), in_shape.size());
  for (axis = 0; axis < label_shape.size(); ++axis) {
    if (axis == static_cast<Shape_t::size_type>(this->axis_)) {
      NBLA_CHECK(label_shape[axis] == 1, error_code::value,
                 "Dimensions of axis of inputs[1] must be 1. "
                 "label_shape[axis]: %d != 1.",
                 label_shape[axis]);
      continue;
    }
    NBLA_CHECK(label_shape[axis] == in_shape[axis], error_code::value,
               "Dimensions of inputs must match except axis. "
               "label_shape[axis]: %d != in_shape[axis]: %d.",
               label_shape[axis], in_shape[axis]);
  }

  Shape_t out_shape = {in_shape[axis_], in_shape[axis_]};
  outputs[0]->reshape(out_shape, true);

  Size_t size = inputs[0]->size();
  Size_t size_axis = inputs[0]->size(axis_);
  size0_ = size / size_axis;          // Batch size.
  size1_ = inputs[0]->shape()[axis_]; // Size of specified axis.
  size2_ = size / size0_ / size1_;    // Size of rest.
  NBLA_CHECK(size0_ * size1_ * size2_ == size, error_code::unclassified,
             "An error occurred during setup ConfusionMatrix function.");
}

template <typename T, typename T1>
void ConfusionMatrix<T, T1>::forward_impl(const Variables &inputs,
                                          const Variables &outputs) {
  // Setting up variables
  const T *p = inputs[0]->get_data_pointer<T>(this->ctx_);
  const T1 *l = inputs[1]->get_data_pointer<T1>(this->ctx_);
  T1 *y = outputs[0]->cast_data_and_get_pointer<T1>(this->ctx_, true);
  memset(y, 0, sizeof(T1) * size1_ * size1_);

  for (int i0 = 0; i0 < size0_; ++i0) {
    for (int i2 = 0; i2 < size2_; ++i2) {
      const int j = i0 * size2_ + i2;
      T1 label = l[j];
      T1 index = 0;
      const int k = i0 * size1_ * size2_ + i2;
      for (int i1 = 1; i1 < size1_; ++i1) {
        if (p[k + i1 * size2_] > p[k + index * size2_]) {
          index = i1;
        }
      }
      y[label * size1_ + index]++;
    }
  }
}

template <typename T, typename T1>
void ConfusionMatrix<T, T1>::backward_impl(const Variables &inputs,
                                           const Variables &outputs,
                                           const vector<bool> &propagate_down,
                                           const vector<bool> &accum) {
  // not supported.
}

} // namespace nbla
