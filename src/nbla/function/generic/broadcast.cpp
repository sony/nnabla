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

/** Broadcast
 */

#include <nbla/array.hpp>
#include <nbla/function/broadcast.hpp>
#include <nbla/variable.hpp>

#include <cstring>
#include <iostream>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(Broadcast, const vector<int> &);

template <typename T>
void Broadcast<T>::setup_impl(const Variables &inputs,
                              const Variables &outputs) {
  auto inshape = inputs[0]->shape();
  auto ndim = inputs[0]->ndim();
  if (ndim > 0) {
    NBLA_CHECK(shape_.size() == static_cast<unsigned>(ndim), error_code::value,
               "Number of dimension must match. Shape: %d != input: %d.",
               shape_.size(), ndim);
  }
  // X Stride and Y shape.
  stride_x_.reshape({ndim}, true);
  shape_y_.reshape({ndim}, true);
  Context cpu = Context().set_array_class("CpuCachedArray");
  int *stride_x = stride_x_.cast_data_and_get_pointer<int>(cpu, true);
  int *shape_y = shape_y_.cast_data_and_get_pointer<int>(cpu, true);
  auto stride_x_in = inputs[0]->strides();
  // Check shape, and store variables.
  for (int d = 0; d < ndim; ++d) {
    shape_y[d] = shape_[d];
    if (inshape[d] == shape_[d]) {
      stride_x[d] = stride_x_in[d];
      continue;
    }
    NBLA_CHECK(
        inshape[d] == 1, error_code::value,
        "Size of a dimension broadcasted must be 1 (%d at the dimension %d).",
        inshape[d], d);
    stride_x[d] = 0;
  }
  Shape_t outshape(shape_.begin(), shape_.end());
  outputs[0]->reshape(outshape, true);
}

// ----------------------------------------------------------------------------
// Strided index getter
// ----------------------------------------------------------------------------
template <int ND> struct strided_index {
  static int get(int y_index, const int *stride_x, const int *shape_y) {
    int stride = 1;
    int x_index = 0;
    strided_index<ND - 1>::_get(y_index, stride_x, shape_y, stride, x_index);
    return x_index;
  }
  static void _get(int y_index, const int *stride_x, const int *shape_y,
                   int &stride, int &x_index) {
    const int dim_index = int(y_index / stride) % shape_y[ND];
    stride *= shape_y[ND];
    x_index += dim_index * stride_x[ND];
    strided_index<ND - 1>::_get(y_index, stride_x, shape_y, stride, x_index);
  }
};
template <> struct strided_index<0> {
  static int get(int y_index, const int *stride_x, const int *shape_y) {
    return 0;
  }
  static void _get(int y_index, const int *stride_x, const int *shape_y,
                   int &stride, int &x_index) {
    const int dim_index = int(y_index / stride) % shape_y[0];
    stride *= shape_y[0];
    x_index += dim_index * stride_x[0];
  }
};

// ----------------------------------------------------------------------------
// Broadcast all elements
// ----------------------------------------------------------------------------
template <int Ndim, typename T>
void broadcast(size_t size, const T *x, const int *stride_x, const int *shape_y,
               T *y) {
  for (size_t i = 0; i < size; ++i) {
    int j = strided_index<Ndim>::get(i, stride_x, shape_y);
    y[i] = x[j];
  }
}

// ----------------------------------------------------------------------------
// Unrolled broadcast caller for templated dimension
// ----------------------------------------------------------------------------
template <int ND, typename T> struct switch_broadcast {
  static void call(int num, size_t size, const T *x, const int *stride_x,
                   const int *shape_y, T *y) {
    if (ND == num) {
      return broadcast<ND, T>(size, x, stride_x, shape_y, y);
    }
    switch_broadcast<ND - 1, T>::call(num, size, x, stride_x, shape_y, y);
  }
};

template <typename T> struct switch_broadcast<-1, T> {
  static void call(int num, size_t size, const T *x, const int *stride_x,
                   const int *shape_y, T *y) {
    NBLA_ERROR(error_code::not_implemented,
               "Broadcast is not implemented for %d dimensional array.", num);
  }
};

// ----------------------------------------------------------------------------
// Broadcast backward (naive implementation)
// ----------------------------------------------------------------------------
template <int Ndim, typename T>
void broadcast_backward(size_t size, const T *dy, const int *stride_x,
                        const int *shape_y, T *g) {
  for (size_t i = 0; i < size; ++i) {
    int j = strided_index<Ndim>::get(i, stride_x, shape_y);
    g[j] += dy[i];
  }
}

// ----------------------------------------------------------------------------
// Unrolled caller
// ----------------------------------------------------------------------------
template <int ND, typename T> struct switch_broadcast_backward {
  static void call(int num, size_t size, const T *dy, const int *stride_x,
                   const int *shape_y, T *g) {
    if (ND == num) {
      return broadcast_backward<ND, T>(size, dy, stride_x, shape_y, g);
    }
    switch_broadcast_backward<ND - 1, T>::call(num, size, dy, stride_x, shape_y,
                                               g);
  }
};

template <typename T> struct switch_broadcast_backward<-1, T> {
  static void call(int num, size_t size, const T *dy, const int *stride_x,
                   const int *shape_y, T *g) {
    NBLA_ERROR(error_code::not_implemented,
               "Broadcast is not implemented for %d dimensional array.", num);
  }
};

// ----------------------------------------------------------------------------
// Forward
// ----------------------------------------------------------------------------
template <typename T>
void Broadcast<T>::forward_impl(const Variables &inputs,
                                const Variables &outputs) {
  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  T *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_, true);
  const int *stride_x = stride_x_.get_data_pointer<int>(this->ctx_);
  const int *shape_y = shape_y_.get_data_pointer<int>(this->ctx_);
  int ndim = inputs[0]->ndim();
  int size = outputs[0]->size();
  switch_broadcast<NBLA_BROADCAST_MAX_DIM, T>::call(ndim, size, x, stride_x,
                                                    shape_y, y);
}

// ----------------------------------------------------------------------------
// Backward
// ----------------------------------------------------------------------------
template <typename T>
void Broadcast<T>::backward_impl(const Variables &inputs,
                                 const Variables &outputs,
                                 const vector<bool> &propagate_down,
                                 const vector<bool> &accum) {
  if (!propagate_down[0])
    return;
  const T *dy = outputs[0]->get_grad_pointer<T>(this->ctx_);
  T *g = inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_, !accum[0]);
  const int *stride_x = stride_x_.get_data_pointer<int>(this->ctx_);
  const int *shape_y = shape_y_.get_data_pointer<int>(this->ctx_);
  int ndim = inputs[0]->ndim();
  int size = outputs[0]->size();
  if (!accum[0])
    memset((void *)g, 0, sizeof(*g) * inputs[0]->size());
  switch_broadcast_backward<NBLA_BROADCAST_MAX_DIM, T>::call(
      ndim, size, dy, stride_x, shape_y, g);
}

} // namespace nbla
