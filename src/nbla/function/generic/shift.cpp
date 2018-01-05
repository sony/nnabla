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

/** Shift
 */
#include <nbla/array.hpp>
#include <nbla/function/shift.hpp>
#include <nbla/variable.hpp>

#include <algorithm>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(Shift, const vector<int> &, const string &);

template <typename T>
void Shift<T>::setup_impl(const Variables &inputs, const Variables &outputs) {
  outputs[0]->reshape(inputs[0]->shape(), true);
  prepare_addr_table(inputs);
}

template <typename T>
void Shift<T>::prepare_addr_table(const Variables &inputs) {
  const Shape_t &in_shape = inputs[0]->shape();
  const int in_dim = in_shape.size();
  addr_table_.resize(in_dim);
  for (int id = 0; id < in_dim; ++id) {
    const int stride = inputs[0]->strides()[id];
    std::vector<int> &table = addr_table_[id];
    const int size = inputs[0]->shape()[id];
    table.resize(size);
    const int shift_index = shifts_.size() - in_dim + id;
    const int shift = shift_index >= 0 ? -shifts_[shift_index] : 0;

    if (border_mode_ == "reflect") {
      for (int i = 0; i < size; i++) {
        const int a =
            size > 1 ? (std::abs(i + size * 2 + shift) % (size * 2)) : 0;
        table[i] = (a >= size ? (size * 2) - 1 - a : a) * stride;
      }
    } else { // if (border_mode_ == "nearest") {
      for (int i = 0; i < size; i++) {
        table[i] = std::min(std::max(i + shift, 0), size - 1) * stride;
      }
    }
  }
}

template <typename T>
void Shift<T>::shift_recursive(Variable *inp, const T *x, T *y, int x_offset,
                               int y_offset, int dim) {
  int current_y_offset = y_offset;
  const int stride = inp->strides()[dim];
  const int size = inp->shape()[dim];
  const std::vector<int> &table = addr_table_[dim];
  if (dim == inp->shape().size() - 1) {
    for (int i = 0; i < size; ++i) {
      y[current_y_offset] = x[x_offset + table[i]];
      current_y_offset += stride;
    }
  } else {
    for (int i = 0; i < size; ++i) {
      shift_recursive(inp, x, y, x_offset + table[i], current_y_offset,
                      dim + 1);
      current_y_offset += stride;
    }
  }
}

template <typename T>
void Shift<T>::shift_backward_recursive(Variable *outp, const T *dy, T *dx,
                                        int x_offset, int y_offset, int dim) {
  int current_y_offset = y_offset;
  const int stride = outp->strides()[dim];
  const int size = outp->shape()[dim];
  const std::vector<int> &table = addr_table_[dim];
  if (dim == outp->shape().size() - 1) {
    for (int i = 0; i < size; ++i) {
      dx[x_offset + table[i]] += dy[current_y_offset];
      current_y_offset += stride;
    }
  } else {
    for (int i = 0; i < size; ++i) {
      shift_backward_recursive(outp, dy, dx, x_offset + table[i],
                               current_y_offset, dim + 1);
      current_y_offset += stride;
    }
  }
}

template <typename T>
void Shift<T>::forward_impl(const Variables &inputs, const Variables &outputs) {
  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  T *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_);
  shift_recursive(inputs[0], x, y, 0, 0, 0);
}

template <typename T>
void Shift<T>::backward_impl(const Variables &inputs, const Variables &outputs,
                             const vector<bool> &propagate_down,
                             const vector<bool> &accum) {
  if (!propagate_down[0]) {
    return;
  }
  if (!accum[0])
    inputs[0]->grad()->zero();
  const T *dy = outputs[0]->get_grad_pointer<T>(this->ctx_);
  T *dx = inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_);
  shift_backward_recursive(outputs[0], dy, dx, 0, 0, 0);
}

} // namespace nbla
