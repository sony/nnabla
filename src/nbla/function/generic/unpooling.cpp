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

// Unpooling.cpp

#include <nbla/array.hpp>
#include <nbla/function/unpooling.hpp>
#include <nbla/variable.hpp>

#include <algorithm>
#include <cstring>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(Unpooling, const vector<int> &, bool);

template <typename T>
void Unpooling<T>::setup_impl(const Variables &inputs,
                              const Variables &outputs) {

  // compute out shape
  Shape_t inshape = inputs[0]->shape();
  Shape_t outshape = inshape;

  NBLA_CHECK(this->kernel_.size() <= inshape.size(), error_code::value,
             "Length of kernel must be less than length of inshape. "
             "Length of kernel: %d > Length of inshape: %d.",
             kernel_.size(), inshape.size());
  auto ndim = inputs[0]->ndim();
  auto kdim = kernel_.size();
  auto offset = channel_last_ ? (ndim - kdim - 1) : (ndim - kdim);
  for (auto i = decltype(kdim){0}; i < kdim; i++) {
    outshape[offset + i] = inshape[offset + i] * kernel_[i];
  }
  outputs[0]->reshape(outshape, true);
}

template <typename T>
void Unpooling<T>::unpooling_forward_recursive(const Variable *inp,
                                               Variable *outp, const T *x, T *y,
                                               int x_offset, int y_offset,
                                               int dim) {
  int current_x_offset = x_offset, current_y_offset = y_offset;
  auto ndim = inp->shape().size();
  auto kdim = this->kernel_.size();
  const int x_stride = inp->strides()[dim];
  const int y_stride = outp->strides()[dim];
  const int kernel = (static_cast<unsigned>(dim) < (ndim - kdim))
                         ? 1
                         : this->kernel_[dim - (ndim - kdim)];
  const int size = outp->shape()[dim];

  if (static_cast<unsigned>(dim) == inp->shape().size() - 1) {
    const T *current_x = x + current_x_offset;
    T *current_y = y + current_y_offset;
    if (x_stride == 1 && kernel == 1) {
      memcpy((void *)current_y, current_x, sizeof(T) * size);
    } else {
      const T *end_y = current_y + size * y_stride;
      int count = 0;
      while (current_y != end_y) {
        *current_y = *current_x;
        if (++count >= kernel) {
          count = 0;
          current_x += x_stride;
        }
        current_y += y_stride;
      }
    }
  } else {
    int count = 0;
    for (int i = 0; i < size; i++) {
      unpooling_forward_recursive(inp, outp, x, y, current_x_offset,
                                  current_y_offset, dim + 1);
      if (++count >= kernel) {
        count = 0;
        current_x_offset += x_stride;
      }
      current_y_offset += y_stride;
    }
  }
}

template <class T>
void Unpooling<T>::forward_impl(const Variables &inputs,
                                const Variables &outputs) {
  NBLA_CHECK(!channel_last_, error_code::not_implemented,
             "Unpooling with channel_last is not supported in CPU.");

  const T *px = inputs[0]->get_data_pointer<T>(this->ctx_);
  T *py = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_, true);

  unpooling_forward_recursive(inputs[0], outputs[0], px, py, 0, 0, 0);
}

template <typename T>
void Unpooling<T>::unpooling_backward_recursive(Variable *outp,
                                                const Variable *inp, T *dx,
                                                const T *dy, int x_offset,
                                                int y_offset, int dim) {

  int current_x_offset = x_offset, current_y_offset = y_offset;
  auto ndim = inp->shape().size();
  auto kdim = this->kernel_.size();
  const int x_stride = outp->strides()[dim];
  const int y_stride = inp->strides()[dim];
  const int kernel = (static_cast<unsigned>(dim) < (ndim - kdim))
                         ? 1
                         : this->kernel_[dim - (ndim - kdim)];
  const int size = inp->shape()[dim];

  if (static_cast<unsigned>(dim) == inp->shape().size() - 1) {
    T *current_dx = dx + current_x_offset;
    const T *current_dy = dy + current_y_offset;
    const T *end_dy = current_dy + size * y_stride;
    int count = 0;
    while (current_dy != end_dy) {
      *current_dx += *current_dy;
      if (++count >= kernel) {
        count = 0;
        current_dx += x_stride;
      }
      current_dy += y_stride;
    }
  } else {
    int count = 0;
    for (int i = 0; i < size; i++) {
      unpooling_backward_recursive(outp, inp, dx, dy, current_x_offset,
                                   current_y_offset, dim + 1);
      if (++count >= kernel) {
        count = 0;
        current_x_offset += x_stride;
      }
      current_y_offset += y_stride;
    }
  }
}

template <class T>
void Unpooling<T>::backward_impl(const Variables &inputs,
                                 const Variables &outputs,
                                 const vector<bool> &propagate_down,
                                 const vector<bool> &accum) {
  if (!propagate_down[0])
    return;
  if (!accum[0])
    inputs[0]->grad()->zero();
  T *pdx = inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_, false);
  const T *pdy = outputs[0]->get_grad_pointer<T>(this->ctx_);
  unpooling_backward_recursive(inputs[0], outputs[0], pdx, pdy, 0, 0, 0);
}
}
