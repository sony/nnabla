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
}

template <typename T>
template <bool is_backward>
void Shift<T>::shift_recursive(Variable *inp, const T *src, T *dst,
                               int x_offset, int y_offset, int dim) {
  // Note src is x and dst is y in forward,
  // while src is dy and dst is dx in backward.

  int current_y_offset = y_offset;
  const int stride = inp->strides()[dim];
  const int size = inp->shape()[dim];
  const int shift_index = shifts_.size() - inp->shape().size() + dim;
  const int shift = shift_index >= 0 ? -shifts_[shift_index] : 0;

  for (int i = 0; i < size; ++i) {
    // Determine shifted index j
    int j;
    if (border_mode_ == "reflect") {
      const int a =
          size > 1 ? (std::abs(i + size * 2 + shift) % (size * 2)) : 0;
      j = (a >= size ? (size * 2) - 1 - a : a) * stride;

    } else { // if (border_mode_ == "nearest") {
      j = std::min(std::max(i + shift, 0), size - 1) * stride;
    }

    // Copy src values to dst with shifted index recursively.
    if (static_cast<Shape_t::size_type>(dim) == inp->shape().size() - 1) {
      if (is_backward) {
        // In backward, dy is accumulated to dx.
        dst[x_offset + j] += src[current_y_offset];
      } else {
        dst[current_y_offset] = src[x_offset + j];
      }
      current_y_offset += stride;
    } else {
      shift_recursive<is_backward>(inp, src, dst, x_offset + j,
                                   current_y_offset, dim + 1);
      current_y_offset += stride;
    }
  }
}

template <typename T>
void Shift<T>::forward_impl(const Variables &inputs, const Variables &outputs) {
  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  T *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_, true);
  shift_recursive<false>(inputs[0], x, y, 0, 0, 0);
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
  T *dx = inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_, false);
  shift_recursive<true>(outputs[0], dy, dx, 0, 0, 0);
}

} // namespace nbla
