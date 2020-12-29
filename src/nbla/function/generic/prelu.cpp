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

/** PReLU
 */
#include <nbla/array.hpp>
#include <nbla/function/prelu.hpp>
#include <nbla/variable.hpp>

#include <algorithm>
#include <cstring>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(PReLU, int);

template <typename T>
void PReLU<T>::setup_impl(const Variables &inputs, const Variables &outputs) {
  Shape_t shape_x = inputs[0]->shape();
  Shape_t shape_w = inputs[1]->shape();

  NBLA_CHECK(base_axis_ >= 0, error_code::value,
             "base_axis may not be less than zero, got %d", base_axis_);
  auto base_axis = static_cast<Shape_t::size_type>(base_axis_);
  NBLA_CHECK(base_axis < shape_x.size(), error_code::value,
             "base_axis must be less than ndim of inputs[0]. "
             "base_axis: %d >= ndim of inputs[0]: %d.",
             base_axis_, shape_x.size());
  NBLA_CHECK(inputs[1]->size() == 1 ||
                 (shape_w.size() == 1 && shape_w[0] == shape_x[base_axis]),
             error_code::value, "The negative slope must be a 1d "
                                "tensor or a scalar.");
  Shape_t stride_x = get_c_contiguous_strides(shape_x);
  base_shape_ = shape_x[base_axis];
  base_stride_ = stride_x[base_axis];
  outputs[0]->reshape(shape_x, true);
}

template <typename T>
void PReLU<T>::forward_impl(const Variables &inputs, const Variables &outputs) {
  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  const T *w = inputs[1]->get_data_pointer<T>(this->ctx_);
  T *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_, true);
  const Size_t size = inputs[0]->size();
  if (inputs[1]->size() == 1) {
    // Single slope mode.
    for (int s = 0; s < size; ++s) {
      y[s] = (x[s] >= 0) ? x[s] : x[s] * (*w);
    }
  } else {
    // Channel-varying slope mode.
    for (int s = 0; s < size; ++s) {
      const int iw = int(s / base_stride_) % base_shape_;
      y[s] = (x[s] >= 0) ? x[s] : x[s] * w[iw];
    }
  }
}

template <typename T>
void PReLU<T>::backward_impl(const Variables &inputs, const Variables &outputs,
                             const vector<bool> &propagate_down,
                             const vector<bool> &accum) {
  if (!(propagate_down[0] || propagate_down[1])) {
    return;
  }

  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  const T *w = inputs[1]->get_data_pointer<T>(this->ctx_);
  const T *dy = outputs[0]->get_grad_pointer<T>(this->ctx_);
  T *dx = nullptr;
  T *dw = nullptr;
  if (propagate_down[0])
    dx = inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_, !accum[0]);
  if (propagate_down[1])
    dw = inputs[1]->cast_grad_and_get_pointer<T>(this->ctx_, !accum[1]);

  const Size_t size = inputs[0]->size();
  if (inputs[1]->size() == 1) {
    // Single slope mode.
    if (dx) {
      for (int s = 0; s < size; ++s) {
        dx[s] =
            (accum[0] ? dx[s] : (T)0) + ((x[s] >= 0) ? dy[s] : dy[s] * (*w));
      }
    }
    if (dw) {
      if (!accum[1])
        *dw = 0;
      for (int s = 0; s < size; ++s) {
        if (x[s] < 0) {
          *dw += dy[s] * x[s];
        }
      }
    }
  } else {
    // Channel-varying slope mode.
    if (dx) {
      for (int s = 0; s < size; ++s) {
        const int iw = int(s / base_stride_) % base_shape_;
        dx[s] =
            (accum[0] ? dx[s] : (T)0) + ((x[s] >= 0) ? dy[s] : dy[s] * w[iw]);
      }
    }
    if (dw) {
      if (!accum[1])
        memset((void *)dw, 0, sizeof(*dw) * inputs[1]->size());
      for (int s = 0; s < size; ++s) {
        if (x[s] < 0) {
          const int iw = int(s / base_stride_) % base_shape_;
          dw[iw] += dy[s] * x[s];
        }
      }
    }
  }
}

} // namespace nbla
