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

// transpose.cpp

#include <nbla/array.hpp>
#include <nbla/function/transpose.hpp>
#include <nbla/variable.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(Transpose, const vector<int> &);

template <typename T>
void Transpose<T>::setup_impl(const Variables &inputs,
                              const Variables &outputs) {
  const int ndim = inputs[0]->ndim();
  NBLA_CHECK(ndim == axes_.size(), error_code::value,
             "Length of axes must be same as inputs. Given %d != %d.", ndim,
             axes_.size());

  Shape_t shape(ndim);
  for (int i = 0; i < ndim; i++) {
    NBLA_CHECK(axes_[i] < inputs[0]->shape().size(), error_code::value,
               "axes must be less than ndim of inputs[0]. "
               "axes[%d]: %d >= ndim of inputs[0]: %d.",
               i, axes_[i], inputs[0]->shape().size());
    for (int i2 = 0; i2 < i; i2++) {
      NBLA_CHECK(axes_[i] != axes_[i2], error_code::value,
                 "Axes duplicated. axes[%d]: %d == axes[%d]: %d.", i, axes_[i],
                 i2, axes_[i2]);
    }
    shape[i] = inputs[0]->shape()[axes_[i]];
  }
  outputs[0]->reshape(shape, true);

  v_axes_.reshape(Shape_t{ndim}, true);
  v_x_strides_.reshape(Shape_t{ndim}, true);
  v_y_strides_.reshape(Shape_t{ndim}, true);
  v_y_shape_.reshape(Shape_t{ndim}, true);
  Context cpu; // CPU Context
  int64_t *p_axes = v_axes_.cast_data_and_get_pointer<int64_t>(cpu);
  int64_t *p_x_strides = v_x_strides_.cast_data_and_get_pointer<int64_t>(cpu);
  int64_t *p_y_strides = v_y_strides_.cast_data_and_get_pointer<int64_t>(cpu);
  int64_t *p_y_shape = v_y_shape_.cast_data_and_get_pointer<int64_t>(cpu);
  for (int i = 0; i < ndim; ++i) {
    p_axes[i] = axes_[i];
    p_x_strides[i] = inputs[0]->strides()[i];
    p_y_strides[i] = outputs[0]->strides()[i];
    p_y_shape[i] = outputs[0]->shape()[i];
  }
}

template <class T>
void Transpose<T>::forward_impl(const Variables &inputs,
                                const Variables &outputs) {

  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  T *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_);
  const int64_t *axes = v_axes_.get_data_pointer<int64_t>(this->ctx_);
  const int64_t *x_strides = v_x_strides_.get_data_pointer<int64_t>(this->ctx_);
  const int64_t *y_strides = v_y_strides_.get_data_pointer<int64_t>(this->ctx_);
  const int64_t *y_shape = v_y_shape_.get_data_pointer<int64_t>(this->ctx_);
  const int ndim = inputs[0]->ndim();
  const int size = outputs[0]->size();

  for (int o = 0; o < size; ++o) {
    int i = 0;
    for (int d = 0; d < ndim; ++d) {
      const int k = int(o / y_strides[d]) % y_shape[d];
      i += k * x_strides[axes[d]];
    }
    y[o] = x[i];
  }
}

template <typename T, bool accum>
void transpose_backward_cpu(int size, int ndim, const int64_t *axes,
                            const int64_t *x_strides, const int64_t *y_strides,
                            const int64_t *y_shape, T *dx, const T *dy) {
  for (int o = 0; o < size; ++o) {
    int i = 0;
    for (int d = 0; d < ndim; ++d) {
      const int k = int(o / y_strides[d]) % y_shape[d];
      i += k * x_strides[axes[d]];
    }
    if (accum)
      dx[i] += dy[o];
    else
      dx[i] = dy[o];
  }
}

template <class T>
void Transpose<T>::backward_impl(const Variables &inputs,
                                 const Variables &outputs,
                                 const vector<bool> &propagate_down,
                                 const vector<bool> &accum) {
  if (!propagate_down[0])
    return;

  T *dx = inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_);
  const T *dy = outputs[0]->get_grad_pointer<T>(this->ctx_);

  const int64_t *axes = v_axes_.get_data_pointer<int64_t>(this->ctx_);
  const int64_t *x_strides = v_x_strides_.get_data_pointer<int64_t>(this->ctx_);
  const int64_t *y_strides = v_y_strides_.get_data_pointer<int64_t>(this->ctx_);
  const int64_t *y_shape = v_y_shape_.get_data_pointer<int64_t>(this->ctx_);
  const int ndim = inputs[0]->ndim();
  const int size = outputs[0]->size();
  if (accum[0])
    transpose_backward_cpu<T, true>(size, ndim, axes, x_strides, y_strides,
                                    y_shape, dx, dy);
  else
    transpose_backward_cpu<T, false>(size, ndim, axes, x_strides, y_strides,
                                     y_shape, dx, dy);
}
}
