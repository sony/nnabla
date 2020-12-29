// Copyright (c) 2018 Sony Corporation. All Rights Reserved.
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

#include <algorithm>
#include <nbla/array.hpp>
#include <nbla/common.hpp>
#include <nbla/function/sort.hpp>
#include <nbla/function/transpose.hpp>
#include <nbla/variable.hpp>
#include <numeric>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(Sort, int, bool, bool, bool);

template <typename T>
void Sort<T>::setup_impl(const Variables &inputs, const Variables &outputs) {
  if (this->axis > 0) {
    NBLA_CHECK(static_cast<unsigned>(this->axis) < inputs[0]->shape().size(),
               error_code::value,
               "Sort axis must be less than the number of input dimensions "
               "but axis %d >= ndim of x %d.",
               this->axis, inputs[0]->shape().size());
  }
  if (this->axis < 0) {
    NBLA_CHECK(inputs[0]->shape().size() + this->axis >= 0, error_code::value,
               "Negative sort axis must not be less than -%d dimensions of "
               "input variable, but got axis %d.",
               inputs[0]->shape().size(), inputs[0]->shape().size(),
               this->axis);
    this->axis += inputs[0]->shape().size();
  }

  const auto &shape = inputs[0]->shape();

  this->inner_size = 1;
  for (int i = shape.size() - 1; i > this->axis; i--)
    this->inner_size *= shape[i];

  this->outer_size = this->inner_size * shape[this->axis];

  this->total_size = this->outer_size;
  for (int i = this->axis - 1; i >= 0; i--)
    this->total_size *= shape[i];

  this->sort_index.reshape(shape, true);
  this->temp_index.reshape(Shape_t{shape[this->axis]}, true);

  outputs[0]->reshape(shape, true);
  if (this->with_index && !this->only_index)
    outputs[1]->reshape(shape, true);
}

template <typename T>
void Sort<T>::forward_impl(const Variables &inputs, const Variables &outputs) {
  const auto &shape = inputs[0]->shape();
  Variable &sort_index_var = this->sort_index;
  Variable &temp_index_var = this->temp_index;

  auto sort_index_ptr = sort_index_var.cast_data_and_get_pointer<size_t>(ctx_);
  auto temp_index_ptr = temp_index_var.cast_data_and_get_pointer<size_t>(ctx_);
  auto x_data = inputs[0]->get_data_pointer<T>(ctx_);

  auto outer_x_ptr = x_data;
  auto outer_i_ptr = sort_index_ptr;
  auto stride = this->inner_size;

  while (outer_x_ptr < x_data + this->total_size) {
    auto inner_x_ptr = outer_x_ptr;
    auto inner_i_ptr = outer_i_ptr;

    while (inner_x_ptr < outer_x_ptr + this->inner_size) {
      std::iota(temp_index_ptr, temp_index_ptr + temp_index_var.size(), 0);
      auto x = inner_x_ptr; // shorter name for compare function
      auto s = stride;      // shorter name for compare function

      if (this->reverse) {
        std::sort(temp_index_ptr, temp_index_ptr + temp_index_var.size(),
                  [&](size_t i1, size_t i2) { return x[i1 * s] > x[i2 * s]; });
      } else {
        std::sort(temp_index_ptr, temp_index_ptr + temp_index_var.size(),
                  [&](size_t i1, size_t i2) { return x[i1 * s] < x[i2 * s]; });
      }

      for (size_t i = 0; i < static_cast<size_t>(shape[this->axis]); i++) {
        inner_i_ptr[i * stride] = temp_index_ptr[i];
      }
      inner_x_ptr++;
      inner_i_ptr++;
    }
    outer_x_ptr += this->outer_size;
    outer_i_ptr += this->outer_size;
  }

  if (!this->only_index) {
    auto y_data = outputs[0]->cast_data_and_get_pointer<T>(ctx_, true);
    auto outer_x_ptr = x_data;
    auto outer_y_ptr = y_data;
    auto outer_i_ptr = sort_index_ptr;
    auto stride = this->inner_size;

    while (outer_x_ptr < x_data + this->total_size) {
      auto inner_x_ptr = outer_x_ptr;
      auto inner_y_ptr = outer_y_ptr;
      auto inner_i_ptr = outer_i_ptr;

      while (inner_x_ptr < outer_x_ptr + this->inner_size) {
        for (size_t i = 0; i < static_cast<size_t>(shape[this->axis]); i++) {
          const auto sort_index = inner_i_ptr[i * stride];
          inner_y_ptr[i * stride] = inner_x_ptr[sort_index * stride];
        }
        inner_x_ptr++;
        inner_y_ptr++;
        inner_i_ptr++;
      }
      outer_x_ptr += this->outer_size;
      outer_y_ptr += this->outer_size;
      outer_i_ptr += this->outer_size;
    }
  }

  if (this->with_index || this->only_index) {
    Variable *out_var = this->only_index ? outputs[0] : outputs[1];
    auto out_ptr = out_var->cast_data_and_get_pointer<size_t>(ctx_, true);
    std::copy_n(sort_index_ptr, out_var->size(), out_ptr);
  }
}

template <typename T>
void Sort<T>::backward_impl(const Variables &inputs, const Variables &outputs,
                            const vector<bool> &propagate_down,
                            const vector<bool> &accum) {
  if (!propagate_down[0]) {
    return;
  }

  Variable &sort_index_var = this->sort_index;
  auto sort_index_ptr = sort_index_var.cast_data_and_get_pointer<size_t>(ctx_);
  auto x_grad = inputs[0]->cast_grad_and_get_pointer<T>(ctx_, !accum[0]);
  auto y_grad = outputs[0]->get_grad_pointer<T>(ctx_);
  const auto &shape = inputs[0]->shape();

  auto outer_x_ptr = x_grad;
  auto outer_y_ptr = y_grad;
  auto outer_i_ptr = sort_index_ptr;
  auto stride = this->inner_size;

  while (outer_x_ptr < x_grad + this->total_size) {
    auto inner_x_ptr = outer_x_ptr;
    auto inner_y_ptr = outer_y_ptr;
    auto inner_i_ptr = outer_i_ptr;

    while (inner_y_ptr < outer_y_ptr + this->inner_size) {
      if (accum[0]) {
        for (size_t i = 0; i < static_cast<size_t>(shape[this->axis]); i++) {
          const auto sort_index = inner_i_ptr[i * stride];
          inner_x_ptr[i * stride] += inner_y_ptr[sort_index * stride];
        }
      } else {
        for (size_t i = 0; i < static_cast<size_t>(shape[this->axis]); i++) {
          const auto sort_index = inner_i_ptr[i * stride];
          inner_x_ptr[i * stride] = inner_y_ptr[sort_index * stride];
        }
      }
      inner_x_ptr++;
      inner_y_ptr++;
      inner_i_ptr++;
    }
    outer_x_ptr += this->outer_size;
    outer_y_ptr += this->outer_size;
    outer_i_ptr += this->outer_size;
  }
}

} // namespace nbla
