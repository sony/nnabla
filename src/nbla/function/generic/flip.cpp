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

/** Flip
 */
#include <nbla/array.hpp>
#include <nbla/function/flip.hpp>
#include <nbla/variable.hpp>

#include <algorithm>
#include <cstring>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(Flip, const vector<int> &);

template <typename T>
void Flip<T>::setup_impl(const Variables &inputs, const Variables &outputs) {

  for (std::size_t i = 0; i < axes_.size(); ++i) {
    if (axes_[i] < 0)
      axes_[i] += inputs[0]->shape().size();
  }

  outputs[0]->reshape(inputs[0]->shape(), true);
  flip_.resize(inputs[0]->ndim());
}

template <typename T>
void Flip<T>::flip_recursive(Variable *inp, const T *x, T *y,
                             const std::vector<bool> &flip, bool add,
                             int x_offset, int y_offset, int dim) {
  int current_x_offset = x_offset, current_y_offset = y_offset;
  const int y_stride = inp->strides()[dim];
  int x_stride = y_stride;
  const int size = inp->shape()[dim];
  if (flip[dim]) {
    current_x_offset += x_stride * (size - 1);
    x_stride = -x_stride;
  }
  if (dim == inp->ndim() - 1) {
    const T *current_x = x + current_x_offset;
    const T *end_x = current_x + size * x_stride;
    T *current_y = y + current_y_offset;
    if (add) {
      while (current_x != end_x) {
        *current_y += *current_x;
        current_x += x_stride;
        current_y += y_stride;
      }
    } else {
      if (x_stride == 1) {
        memcpy((void *)current_y, current_x, sizeof(T) * size);
      } else
        while (current_x != end_x) {
          *current_y = *current_x;
          current_x += x_stride;
          current_y += y_stride;
        }
    }
    /*
    if (add) {
      for (int i = 0; i < size; i++) {
        y[current_y_offset] += x[current_x_offset];
        current_x_offset += x_stride;
        current_y_offset += y_stride;
      }
    }else {
      for (int i = 0; i < size; i++) {
        y[current_y_offset] = x[current_x_offset];
        current_x_offset += x_stride;
        current_y_offset += y_stride;
      }
    }
    */
  } else {
    for (int i = 0; i < size; i++) {
      flip_recursive(inp, x, y, flip, add, current_x_offset, current_y_offset,
                     dim + 1);
      current_x_offset += x_stride;
      current_y_offset += y_stride;
    }
  }
}

template <typename T>
void Flip<T>::forward_impl(const Variables &inputs, const Variables &outputs) {
  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  T *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_, true);
  for (int id = 0; id < inputs[0]->ndim(); id++) {
    auto itr = std::find(axes_.begin(), axes_.end(), id);
    flip_[id] = itr != axes_.end();
  }
  flip_recursive(inputs[0], x, y, flip_, false, 0, 0, 0);
}

template <typename T>
void Flip<T>::backward_impl(const Variables &inputs, const Variables &outputs,
                            const vector<bool> &propagate_down,
                            const vector<bool> &accum) {
  if (!propagate_down[0]) {
    return;
  }
  const T *dy = outputs[0]->get_grad_pointer<T>(this->ctx_);
  T *dx = inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_, !accum[0]);
  flip_recursive(outputs[0], dy, dx, flip_, accum[0], 0, 0, 0);
}

} // namespace nbla
