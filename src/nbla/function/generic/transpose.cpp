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
#include <nbla/function/transpose.hpp>
#include <nbla/utils/nd_index.hpp>
#include <nbla/variable.hpp>

#include <cmath>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(Transpose, const vector<int> &);

template <typename T>
void Transpose<T>::setup_impl(const Variables &inputs,
                              const Variables &outputs) {
  vector<int> axes = this->axes_;
  const int ndim = inputs[0]->ndim();

  NBLA_CHECK(static_cast<size_t>(ndim) == axes.size(), error_code::value,
             "Length of axes must be same as ndim of input. Given %d != %d.",
             ndim, axes.size());

  Shape_t ishape = inputs[0]->shape();
  Shape_t oshape(ndim);

  for (int i = 0; i < ndim; i++) {
    if (axes[i] < 0) {
      axes[i] += ishape.size();
      NBLA_CHECK(axes[i] >= 0, error_code::value,
                 "Absolute value of each element of axes must be less than "
                 "that of input ndim. "
                 "axes[%d]: %d >= ndim of input: %d.",
                 i, abs(axes[i] - static_cast<int>(ishape.size())), ndim);
    }
    NBLA_CHECK(axes[i] < ndim, error_code::value,
               "Each element of axes must be less than ndim of input. "
               "axes[%d]: %d >= ndim of input: %d.",
               i, axes[i], ndim);
    for (int j = 0; j < i; j++) {
      NBLA_CHECK(axes[i] != axes[j], error_code::value,
                 "Axes duplicated. axes[%d]: %d == axes[%d]: %d.", i, axes[i],
                 j, axes[j]);
    }
    oshape[i] = ishape[axes[i]];
  }
  outputs[0]->reshape(oshape, true);

  // Combine all successively increasing by one transpose axes. For example,
  // transposing shape (16, 3, 100, 100) by axes (0, 2, 3, 1) will be reduced
  // to transposing (16, 3, 10000) by axes (0, 2, 1).
  volatile auto axes_size = axes.size();
  for (size_t i = 1; i < axes_size;) {
    auto this_axis = axes[i];
    auto prev_axis = axes[i - 1];
    if (this_axis == prev_axis + 1) {
      ishape[prev_axis] *= ishape[this_axis];
      ishape.erase(ishape.begin() + this_axis);
      axes.erase(axes.begin() + i);
      axes_size = axes.size();
      for (size_t j = 0; j < axes_size; j++) {
        if (axes[j] > this_axis)
          axes[j] -= 1;
      }
    } else
      i++;
  }

  // Now ishape and axes are compacted and we compute the appropriate shapes,
  // strides and transposed strides to be used in forward and backward.

  this->x_shape_ = ishape;
  this->y_shape_.resize(axes.size());
  for (size_t i = 0; i < axes.size(); i++) {
    this->y_shape_[i] = this->x_shape_[axes[i]];
  }
  this->x_strides_ = ndi::strides(this->x_shape_);
  this->y_strides_ = ndi::strides(this->y_shape_);
  this->x_strides_transposed_.resize(axes.size());
  this->y_strides_transposed_.resize(axes.size());
  for (size_t i = 0; i < axes.size(); i++) {
    this->x_strides_transposed_[i] = this->x_strides_[axes[i]];
    this->y_strides_transposed_[axes[i]] = this->y_strides_[i];
  }
}

template <typename T>
void transpose(const T *src, T *dst, const Shape_t dst_shape,
               const Shape_t transposed_dst_strides, bool accum) {
  auto dst_ndindex = ndi::make_index(dst_shape.size(), Size_t(0));
  Size_t i = 0;
  do {
    T val = src[ndi::nd2flat(dst_ndindex, transposed_dst_strides)];
    dst[i] = accum ? dst[i] + val : val;
  } while (++i && ndi::increment(dst_ndindex, dst_shape));
}

template <typename T>
void Transpose<T>::forward_impl(const Variables &inputs,
                                const Variables &outputs) {
  auto x = inputs[0]->get_data_pointer<T>(this->ctx_);
  auto y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_, true);
  transpose(x, y, this->y_shape_, this->x_strides_transposed_, false);
}

template <typename T>
void Transpose<T>::backward_impl(const Variables &inputs,
                                 const Variables &outputs,
                                 const vector<bool> &propagate_down,
                                 const vector<bool> &accum) {
  if (!propagate_down[0])
    return;

  auto dy = outputs[0]->get_grad_pointer<T>(this->ctx_);
  auto dx = inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_, !accum[0]);
  transpose(dy, dx, this->x_shape_, this->y_strides_transposed_, accum[0]);
}
}
