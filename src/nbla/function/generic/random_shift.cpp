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

/** RandomShift
 */
#include <nbla/array.hpp>
#include <nbla/function/random_shift.hpp>
#include <nbla/random_manager.hpp>
#include <nbla/variable.hpp>

#include <algorithm>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(RandomShift, const vector<int> &, const string &,
                              float, int, int);

template <typename T>
void RandomShift<T>::setup_impl(const Variables &inputs,
                                const Variables &outputs) {
  std::random_device rdev_;
  rgen_ = std::mt19937((seed_ == -1 ? rdev_() : seed_));
  size_ = inputs[0]->size() / inputs[0]->size(base_axis_);
  outputs[0]->reshape(inputs[0]->shape(), true);
}

template <typename T>
vector<vector<int>>
RandomShift<T>::prepare_addr_table(const Variables &inputs,
                                   const vector<int> &shifts) {
  vector<vector<int>> addr_table;
  const Shape_t &in_shape = inputs[0]->shape();
  const int in_dim = in_shape.size();
  addr_table.resize(in_dim);
  for (int id = 0; id < in_dim; ++id) {
    const int stride = inputs[0]->strides()[id];
    std::vector<int> &table = addr_table[id];
    const int size = inputs[0]->shape()[id];
    table.resize(size);
    const int shift_index = shifts.size() - in_dim + id;
    const int shift = shift_index >= 0 ? -shifts[shift_index] : 0;

    if (border_mode_ == "reflect") {
      for (int i = 0; i < size; i++) {
        const int a =
            size > 1 ? (std::abs(i + size * 2 + shift) % (size * 2)) : 0;
        table[i] = (a >= size ? (size * 2) - 1 - a : a) * stride;
      }
    } else if (border_mode_ == "nearest") {
      for (int i = 0; i < size; i++) {
        table[i] = std::min(std::max(i + shift, 0), size - 1) * stride;
      }
    } else { // if (border_mode_ == "constant") {
      for (int i = 0; i < size; i++) {
        if ((i + shift < 0) || (i + shift >= size))
          table[i] = CVAL_INDEX;
        else
          table[i] = (i + shift) * stride;
      }
    }
  }
  return addr_table;
}

template <typename T>
void RandomShift<T>::shift_recursive(const Variable *inp, const T *x, T *y,
                                     int x_offset, int y_offset, int dim,
                                     int &shift_index) {
  int current_y_offset = y_offset;
  const int stride = inp->strides()[dim];
  const int size = inp->shape()[dim];
  const std::vector<int> &table = addr_table_[shift_index][dim];
  if (static_cast<Shape_t::size_type>(dim) == inp->shape().size() - 1) {
    for (int i = 0; i < size; ++i) {
      if (x_offset == CVAL_INDEX || table[i] == CVAL_INDEX)
        y[current_y_offset] = constant_value_;
      else
        y[current_y_offset] = x[x_offset + table[i]];
      current_y_offset += stride;
    }
  } else {
    for (int i = 0; i < size; ++i) {
      if (x_offset == CVAL_INDEX || table[i] == CVAL_INDEX) {
        shift_recursive(inp, x, y, CVAL_INDEX, current_y_offset, dim + 1,
                        shift_index);
      } else {
        shift_recursive(inp, x, y, x_offset + table[i], current_y_offset,
                        dim + 1, shift_index);
      }
      current_y_offset += stride;
      if (dim < base_axis_) {
        shift_index = (shift_index + 1) % addr_table_.size();
      }
    }
  }
}

template <typename T>
void RandomShift<T>::shift_backward_recursive(const Variable *inp, const T *dy,
                                              T *dx, int x_offset, int y_offset,
                                              int dim, int &shift_index) {
  int current_y_offset = y_offset;
  const int stride = inp->strides()[dim];
  const int size = inp->shape()[dim];
  const std::vector<int> &table = addr_table_[shift_index][dim];
  if (static_cast<Shape_t::size_type>(dim) == inp->shape().size() - 1) {
    for (int i = 0; i < size; ++i) {
      if ((x_offset != CVAL_INDEX) && (table[i] != CVAL_INDEX)) {
        dx[x_offset + table[i]] += dy[current_y_offset];
      }
      current_y_offset += stride;
    }
  } else {
    for (int i = 0; i < size; ++i) {
      if (x_offset == CVAL_INDEX || table[i] == CVAL_INDEX) {
        shift_backward_recursive(inp, dy, dx, CVAL_INDEX, current_y_offset,
                                 dim + 1, shift_index);
      } else {
        shift_backward_recursive(inp, dy, dx, x_offset + table[i],
                                 current_y_offset, dim + 1, shift_index);
      }
      current_y_offset += stride;
      if (dim < base_axis_) {
        shift_index = (shift_index + 1) % addr_table_.size();
      }
    }
  }
}

template <typename T>
void RandomShift<T>::forward_impl(const Variables &inputs,
                                  const Variables &outputs) {
  std::mt19937 &rgen =
      seed_ == -1 ? SingletonManager::get<RandomManager>()->get_rand_generator()
                  : rgen_;
  addr_table_.resize(size_);
  for (int i = 0; i < size_; i++) {
    vector<int> shifts;
    for (Shape_t::size_type id = 0; id < shifts_.size(); id++) {
      shifts.push_back(rgen() % (shifts_[id] * 2 + 1) - shifts_[id]);
    }
    addr_table_[i] = prepare_addr_table(inputs, shifts);
  }

  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  T *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_, true);
  int shift_index = 0;
  shift_recursive(inputs[0], x, y, 0, 0, 0, shift_index);
}

template <typename T>
void RandomShift<T>::backward_impl(const Variables &inputs,
                                   const Variables &outputs,
                                   const vector<bool> &propagate_down,
                                   const vector<bool> &accum) {
  if (!propagate_down[0]) {
    return;
  }
  if (!accum[0]) {
    inputs[0]->grad()->zero();
  }
  const T *dy = outputs[0]->get_grad_pointer<T>(this->ctx_);
  T *dx = inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_, false);
  int shift_index = 0;
  shift_backward_recursive(outputs[0], dy, dx, 0, 0, 0, shift_index);
}

} // namespace nbla
