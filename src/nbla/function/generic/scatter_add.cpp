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
#include <nbla/function/scatter_add.hpp>
#include <nbla/utils/nd_index.hpp>
#include <nbla/variable.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(ScatterAdd, int);

template <typename T>
void ScatterAdd<T>::setup_impl(const Variables &inputs,
                               const Variables &outputs) {
  NBLA_CHECK(inputs.size() == 3, error_code::value,
             "scatter_add requires 3 inputs");
  auto axis = (axis_ < 0) ? inputs.at(0)->ndim() + axis_ : axis_;
  NBLA_CHECK(axis <= inputs.at(0)->ndim(), error_code::value,
             "Shape error, input dimension must be greater than axis");
  for (Variables::size_type i = 1; i < inputs.size(); ++i) {
    NBLA_CHECK(inputs.at(0)->ndim() == inputs.at(i)->ndim(), error_code::value,
               "Shape error. all inputs must have same dimension");
  }
  // Check x0 and x1 have greater indices size along each indices' dimension
  auto x0_shape = inputs.at(0)->shape();
  auto indices_shape = inputs.at(1)->shape();
  auto x1_shape = inputs.at(2)->shape();
  for (int i = 0; static_cast<size_t>(i) < indices_shape.size(); ++i) {
    if (i != axis) {
      NBLA_CHECK(indices_shape[i] <= x0_shape[i], error_code::value,
                 "Shape error. indices size: %d at axis %d is greater than x0 "
                 "size: %d",
                 indices_shape[i], i, x0_shape[i]);
    }
    NBLA_CHECK(
        indices_shape[i] <= x1_shape[i], error_code::value,
        "Shape error. indices size: %d at axis %d is greater than x1 size: %d",
        indices_shape[i], i, x1_shape[i]);
  }
  outputs[0]->reshape(inputs[0]->shape(), true);
}

template <typename T>
void ScatterAdd<T>::forward_impl(const Variables &inputs,
                                 const Variables &outputs) {
  // Inputs
  const T *x0 = inputs[0]->get_data_pointer<T>(this->ctx_);
  const T *indices = inputs[1]->get_data_pointer<T>(this->ctx_);
  const T *x1 = inputs[2]->get_data_pointer<T>(this->ctx_);
  T *dst = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_, true);

  for (auto i = 0; i < inputs[0]->size(); ++i) {
    dst[i] = x0[i];
  }

  auto x0_shape = inputs[0]->shape();
  auto x0_strides = ndi::strides(x0_shape);

  auto index_shape = inputs[1]->shape();
  auto index_strides = ndi::strides(index_shape);

  auto x1_shape = inputs[2]->shape();
  auto x1_strides = ndi::strides(x1_shape);

  auto axis = (axis_ < 0) ? inputs[0]->ndim() + axis_ : axis_;

  for (int64_t i = 0; i < inputs[1]->size(); ++i) {
    auto dst_axis_index = indices[i];
    NBLA_CHECK((0 <= dst_axis_index && dst_axis_index < x0_shape[axis]),
               error_code::value, "Given index is out of range.");
    auto nd_index = ndi::flat2nd(i, index_strides);
    auto src_flat_index = ndi::nd2flat(nd_index, x1_strides);
    nd_index[axis] = dst_axis_index;
    auto dst_flat_index = ndi::nd2flat(nd_index, x0_strides);
    dst[dst_flat_index] += x1[src_flat_index];
  }
}

template <typename T>
void ScatterAdd<T>::backward_impl(const Variables &inputs,
                                  const Variables &outputs,
                                  const vector<bool> &propagate_down,
                                  const vector<bool> &accum) {
  if (!(propagate_down[0] || propagate_down[2])) {
    return;
  }
  const T *dy = outputs[0]->get_grad_pointer<T>(this->ctx_);
  // Gradient of outputs
  if (propagate_down[0]) {
    T *dx = inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_, !accum[0]);
    for (auto i = 0; i < inputs[0]->size(); ++i) {
      if (accum[0]) {
        dx[i] += dy[i];
      } else {
        dx[i] = dy[i];
      }
    }
  }
  if (propagate_down[2]) {
    auto x0_shape = inputs[0]->shape();
    auto x0_strides = ndi::strides(x0_shape);

    auto index_shape = inputs[1]->shape();
    auto index_strides = ndi::strides(index_shape);

    auto x1_shape = inputs[2]->shape();
    auto x1_strides = ndi::strides(x1_shape);

    auto axis = (axis_ < 0) ? inputs[0]->ndim() + axis_ : axis_;

    T *dx = inputs[2]->cast_grad_and_get_pointer<T>(this->ctx_, !accum[2]);

    // Initialize dx with 0 if accum is false. Because the indices may not
    // correspond to dx 1-by-1.
    if (!accum[2]) {
      for (int64_t i = 0; i < inputs[2]->size(); ++i) {
        dx[i] = 0.0;
      }
    }
    const T *indices = inputs[1]->get_data_pointer<T>(this->ctx_);
    for (int64_t i = 0; i < inputs[1]->size(); ++i) {
      int64_t dst_axis_index = indices[i];
      NBLA_CHECK((0 <= dst_axis_index && dst_axis_index < x0_shape[axis]),
                 error_code::value, "Given index is out of range.");
      auto nd_index = ndi::flat2nd(i, index_strides);
      auto src_flat_index = ndi::nd2flat(nd_index, x1_strides);
      nd_index[axis] = dst_axis_index;
      auto dst_flat_index = ndi::nd2flat(nd_index, x0_strides);
      if (accum[2]) {
        dx[src_flat_index] += dy[dst_flat_index];
      } else {
        dx[src_flat_index] = dy[dst_flat_index];
      }
    }
  }
}
}
