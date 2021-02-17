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

/** MeanSubtraction
 */
#include <nbla/array.hpp>
#include <nbla/function/mean_subtraction.hpp>
#include <nbla/variable.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(MeanSubtraction, int, bool);

template <typename T>
void MeanSubtraction<T>::setup_impl(const Variables &inputs,
                                    const Variables &outputs) {
  // Check num of inputs and outputs.
  NBLA_CHECK(inputs.size() == 3 ||
                 (inputs.size() == 2 && !update_running_mean_),
             error_code::value, "The Number of inputs must be 3 [x, "
                                "running_mean, t] or 2 [x, running_mean].");
  NBLA_CHECK(outputs.size() == 1, error_code::value,
             "The Number of outputs must be 1.");

  // Check shape
  Shape_t shape_i = inputs[0]->shape();
  if (this->base_axis_ < 0)
    this->base_axis_ += shape_i.size();
  NBLA_CHECK(base_axis_ >= 0, error_code::value,
             "base_axis may not be less than zero, got %d", base_axis_);
  auto base_axis = static_cast<Shape_t::size_type>(base_axis_);
  NBLA_CHECK(base_axis < shape_i.size(), error_code::value,
             "base_axis must be less than ndim of inputs[0]. "
             "base_axis: %d >= ndim of inputs[0]: %d.",
             base_axis_, shape_i.size());
  Shape_t shape_m = shape_i;
  shape_m.erase(shape_m.begin(), shape_m.begin() + base_axis_);
  NBLA_CHECK(inputs[1]->shape() == shape_m, error_code::value,
             "Shape of running_mean(inputs[1]) mismatch. "
             "inputs[1] shape: (%s), expected: (%s).",
             string_join(inputs[1]->shape(), string(", ")).c_str(),
             string_join(shape_m, string(", ")).c_str());
  if (inputs.size() == 3) {
    NBLA_CHECK(inputs[2]->size() == 1, error_code::value,
               "Size of t(inputs[2]) must be 1. "
               "inputs[2] size: %d != 1.",
               inputs[2]->size());
  }

  // Check and parse shapes
  size1_ = inputs[0]->size(base_axis_); // Data size
  size0_ = inputs[0]->size() / size1_;  // Data num

  // Reshape outputs and temporary buffer.
  outputs[0]->reshape(shape_i, true);
  mean_.reshape(shape_m, true);
}

template <class T>
void MeanSubtraction<T>::forward_impl(const Variables &inputs,
                                      const Variables &outputs) {
  if (update_running_mean_) { // Training mode.
    forward_impl_batch(inputs, outputs);
  } else { // Testing mode.
    forward_impl_global(inputs, outputs);
  }
}

template <class T>
void MeanSubtraction<T>::forward_impl_batch(const Variables &inputs,
                                            const Variables &outputs) {
  // Inputs
  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  // Output
  T *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_, true);
  T *m = mean_.cast_data_and_get_pointer<T>(this->ctx_, true); // batch mean
  // Inputs/Outputs
  T *rm = inputs[1]->cast_data_and_get_pointer<T>(this->ctx_); // running mean
  int *t =
      inputs[2]->cast_data_and_get_pointer<int>(this->ctx_); // running count
  T coef = 1.0 / ((*t) + 1);

  // Main loop
  for (int i1 = 0; i1 < size1_; ++i1) {
    // Batch and moving mean calculation.
    // Batch mean
    m[i1] = 0;
    for (int i0 = 0; i0 < size0_; ++i0) {
      m[i1] += x[i0 * size1_ + i1];
    }
    m[i1] /= size0_;

    // Moving mean
    rm[i1] = rm[i1] + (m[i1] - rm[i1]) * coef;

    // Output
    for (int i0 = 0; i0 < size0_; ++i0) {
      const int i = i0 * size1_ + i1;
      y[i] = x[i] - rm[i1];
    }
  }
  *t = std::min((*t) + 1,
                std::numeric_limits<int>::max()); // Update running count.
}

template <class T>
void MeanSubtraction<T>::forward_impl_global(const Variables &inputs,
                                             const Variables &outputs) {
  // Inputs
  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  const T *rm = inputs[1]->get_data_pointer<T>(this->ctx_); // running mean

  // Output
  T *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_, true);

  // Output running mean
  for (int i1 = 0; i1 < size1_; ++i1) {
    for (int i0 = 0; i0 < size0_; ++i0) {
      const int i = i0 * size1_ + i1;
      y[i] = x[i] - rm[i1];
    }
  }
}

template <class T>
void MeanSubtraction<T>::backward_impl(const Variables &inputs,
                                       const Variables &outputs,
                                       const vector<bool> &propagate_down,
                                       const vector<bool> &accum) {
  if (update_running_mean_) { // Training mode.
    backward_impl_batch(inputs, outputs, propagate_down, accum);
  } else { // Testing mode.
    backward_impl_global(inputs, outputs, propagate_down, accum);
  }
}

template <typename T, bool accum>
void mean_subtraction_backward_batch(int size, T *dx, const T *dy, T factor) {
  for (int i = 0; i < size; ++i) {
    if (accum)
      dx[i] += dy[i] * (1 - factor);
    else
      dx[i] = dy[i] * (1 - factor);
  }
}

template <class T>
void MeanSubtraction<T>::backward_impl_batch(const Variables &inputs,
                                             const Variables &outputs,
                                             const vector<bool> &propagate_down,
                                             const vector<bool> &accum) {
  if (!propagate_down[0]) {
    return;
  }

  const T *dy = outputs[0]->get_grad_pointer<T>(this->ctx_);
  T *dx = inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_, !accum[0]);

  const int *t = inputs[2]->get_data_pointer<int>(this->ctx_);
  const T factor = (T)1.0 / ((*t) * size0_);
  if (accum[0])
    mean_subtraction_backward_batch<T, true>(inputs[0]->size(), dx, dy, factor);
  else
    mean_subtraction_backward_batch<T, false>(inputs[0]->size(), dx, dy,
                                              factor);
}

template <typename T, bool accum>
void mean_subtraction_backward_global(int size, T *dx, const T *dy) {
  for (int i = 0; i < size; ++i) {
    if (accum)
      dx[i] += dy[i];
    else
      dx[i] = dy[i];
  }
}

template <class T>
void MeanSubtraction<T>::backward_impl_global(
    const Variables &inputs, const Variables &outputs,
    const vector<bool> &propagate_down, const vector<bool> &accum) {
  if (!propagate_down[0]) {
    return;
  }

  const T *dy = outputs[0]->get_grad_pointer<T>(this->ctx_);
  T *dx = inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_, !accum[0]);

  if (accum[0])
    mean_subtraction_backward_global<T, true>(inputs[0]->size(), dx, dy);
  else
    mean_subtraction_backward_global<T, false>(inputs[0]->size(), dx, dy);
}

} // namespace nbla
