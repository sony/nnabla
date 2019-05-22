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

// AveragePooling.cpp

#include <nbla/array.hpp>
#include <nbla/function/average_pooling.hpp>
#include <nbla/variable.hpp>

#include <algorithm>
#include <array>
#include <cstring>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(AveragePooling, const vector<int> &,
                              const vector<int> &, bool, const vector<int> &,
                              bool, bool);

using std::min;
using std::max;

namespace avg_pooling_impl {

template <typename TI, typename TO, int NDIM>
inline const std::array<TO, NDIM> v2a(const std::vector<TI> &v,
                                      const int skip = 0) {
  std::array<TO, NDIM> a;
  for (int i = 0; i < NDIM; i++)
    a[i] = v.at(skip + i);
  return a;
}

typedef std::array<int, 2> Array2D;
typedef std::array<int, 3> Array3D;

template <typename T>
inline void forward_map(const T *x, T *y, bool including_pad,
                        const Array2D &x_stride, const Array2D &x_shape,
                        const Array2D &y_shape, const Array2D &kernel,
                        const Array2D &stride, const Array2D &pad) {
  Array2D y_idx, pool_start, pool_end;

  for (y_idx[0] = 0; y_idx[0] < y_shape[0]; y_idx[0]++) {
    for (y_idx[1] = 0; y_idx[1] < y_shape[1]; y_idx[1]++) {
      for (int a = 0; a < 2; a++) {
        pool_start[a] = y_idx[a] * stride[a] - pad[a];
        pool_end[a] = min(pool_start[a] + kernel[a], x_shape[a] + pad[a]);
      }
      int pool_size = 1;
      for (int a = 0; a < 2; a++) {
        pool_size *= pool_end[a] - pool_start[a];
      }
      for (int a = 0; a < 2; a++) {
        pool_start[a] = max(pool_start[a], 0);
        pool_end[a] = min(pool_end[a], x_shape[a]);
      }
      if (including_pad == false) {
        pool_size = 1;
        for (int a = 0; a < 2; a++) {
          pool_size *= pool_end[a] - pool_start[a];
        }
      }
      T pool_sum = 0;
      for (int i0 = pool_start[0]; i0 < pool_end[0]; i0++) {
        for (int i1 = pool_start[1]; i1 < pool_end[1]; i1++) {
          pool_sum += x[i0 * x_stride[0] + i1];
        }
      }
      *y++ = pool_sum / pool_size;
    }
  }
}

template <typename T>
inline void forward_map(const T *x, T *y, bool including_pad,
                        const Array3D &x_stride, const Array3D &x_shape,
                        const Array3D &y_shape, const Array3D &kernel,
                        const Array3D &stride, const Array3D &pad) {
  Array3D y_idx, pool_start, pool_end;

  for (y_idx[0] = 0; y_idx[0] < y_shape[0]; y_idx[0]++) {
    for (y_idx[1] = 0; y_idx[1] < y_shape[1]; y_idx[1]++) {
      for (y_idx[2] = 0; y_idx[2] < y_shape[2]; y_idx[2]++) {
        for (int a = 0; a < 3; a++) {
          pool_start[a] = y_idx[a] * stride[a] - pad[a];
          pool_end[a] = min(pool_start[a] + kernel[a], x_shape[a] + pad[a]);
        }
        int pool_size = 1;
        for (int a = 0; a < 3; a++) {
          pool_size *= pool_end[a] - pool_start[a];
        }
        for (int a = 0; a < 3; a++) {
          pool_start[a] = max(pool_start[a], 0);
          pool_end[a] = min(pool_end[a], x_shape[a]);
        }
        if (including_pad == false) {
          pool_size = 1;
          for (int a = 0; a < 3; a++) {
            pool_size *= pool_end[a] - pool_start[a];
          }
        }
        T pool_sum = 0;
        for (int i0 = pool_start[0]; i0 < pool_end[0]; i0++) {
          for (int i1 = pool_start[1]; i1 < pool_end[1]; i1++) {
            for (int i2 = pool_start[2]; i2 < pool_end[2]; i2++) {
              pool_sum += x[i0 * x_stride[0] + i1 * x_stride[1] + i2];
            }
          }
        }
        *y++ = pool_sum / pool_size;
      }
    }
  }
}

template <typename T>
inline void backward_map(T *dx, const T *dy, bool including_pad,
                         const Array2D &x_stride, const Array2D &x_shape,
                         const Array2D &y_shape, const Array2D &kernel,
                         const Array2D &stride, const Array2D &pad) {
  Array2D y_idx, pool_start, pool_end;

  for (y_idx[0] = 0; y_idx[0] < y_shape[0]; y_idx[0]++) {
    for (y_idx[1] = 0; y_idx[1] < y_shape[1]; y_idx[1]++) {
      for (int a = 0; a < 2; a++) {
        pool_start[a] = y_idx[a] * stride[a] - pad[a];
        pool_end[a] = min(pool_start[a] + kernel[a], x_shape[a] + pad[a]);
      }
      int pool_size = 1;
      for (int a = 0; a < 2; a++) {
        pool_size *= pool_end[a] - pool_start[a];
      }
      for (int a = 0; a < 2; a++) {
        pool_start[a] = max(pool_start[a], 0);
        pool_end[a] = min(pool_end[a], x_shape[a]);
      }
      if (including_pad == false) {
        pool_size = 1;
        for (int a = 0; a < 2; a++) {
          pool_size *= pool_end[a] - pool_start[a];
        }
      }
      T pool_grad = *dy++ / pool_size;
      for (int i0 = pool_start[0]; i0 < pool_end[0]; i0++) {
        for (int i1 = pool_start[1]; i1 < pool_end[1]; i1++) {
          dx[i0 * x_stride[0] + i1] += pool_grad;
        }
      }
    }
  }
}

template <typename T>
inline void backward_map(T *dx, const T *dy, bool including_pad,
                         const Array3D &x_stride, const Array3D &x_shape,
                         const Array3D &y_shape, const Array3D &kernel,
                         const Array3D &stride, const Array3D &pad) {
  Array3D y_idx, pool_start, pool_end;

  for (y_idx[0] = 0; y_idx[0] < y_shape[0]; y_idx[0]++) {
    for (y_idx[1] = 0; y_idx[1] < y_shape[1]; y_idx[1]++) {
      for (y_idx[2] = 0; y_idx[2] < y_shape[2]; y_idx[2]++) {
        for (int a = 0; a < 3; a++) {
          pool_start[a] = y_idx[a] * stride[a] - pad[a];
          pool_end[a] = min(pool_start[a] + kernel[a], x_shape[a] + pad[a]);
        }
        int pool_size = 1;
        for (int a = 0; a < 3; a++) {
          pool_size *= pool_end[a] - pool_start[a];
        }
        for (int a = 0; a < 3; a++) {
          pool_start[a] = max(pool_start[a], 0);
          pool_end[a] = min(pool_end[a], x_shape[a]);
        }
        if (including_pad == false) {
          pool_size = 1;
          for (int a = 0; a < 3; a++) {
            pool_size *= pool_end[a] - pool_start[a];
          }
        }
        T pool_grad = *dy++ / pool_size;
        for (int i0 = pool_start[0]; i0 < pool_end[0]; i0++) {
          for (int i1 = pool_start[1]; i1 < pool_end[1]; i1++) {
            for (int i2 = pool_start[2]; i2 < pool_end[2]; i2++) {
              dx[i0 * x_stride[0] + i1 * x_stride[1] + i2] += pool_grad;
            }
          }
        }
      }
    }
  }
}
} // namespace avg_pooling_impl

using avg_pooling_impl::v2a;
using avg_pooling_impl::Array2D;
using avg_pooling_impl::Array3D;
using avg_pooling_impl::forward_map;
using avg_pooling_impl::backward_map;

template <typename T>
void AveragePooling<T>::forward_impl(const Variables &inputs,
                                     const Variables &outputs) {
  NBLA_CHECK(!this->channel_last_, error_code::not_implemented,
             "The passed argument channel_last=true is not supported in CPU "
             "pooling.");

  auto x = inputs[0]->get_data_pointer<T>(this->ctx_);
  auto y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_, true);

  const Shape_t &inshape = inputs[0]->shape();
  const Shape_t &outshape = outputs[0]->shape();
  const Shape_t &instrides = inputs[0]->strides();
  const Shape_t &outstrides = outputs[0]->strides();
  const int s = inshape.size() - this->kernel_.size();
  const int x_map_size = (s == 0) ? inputs[0]->size() : instrides[s - 1];
  const int y_map_size = (s == 0) ? outputs[0]->size() : outstrides[s - 1];
  const int n_map = inputs[0]->size() / x_map_size;

  if (this->kernel_.size() == 2) {
    const auto x_stride = v2a<Size_t, int, 2>(instrides, s);
    const auto x_shape = v2a<Size_t, int, 2>(inshape, s);
    const auto y_shape = v2a<Size_t, int, 2>(outshape, s);
    const auto kernel = v2a<int, int, 2>(this->kernel_);
    const auto stride = v2a<int, int, 2>(this->stride_);
    const auto pad = v2a<int, int, 2>(this->pad_);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int n = 0; n < n_map; n++) {
      forward_map(x + n * x_map_size, y + n * y_map_size, this->including_pad_,
                  x_stride, x_shape, y_shape, kernel, stride, pad);
    }
  }

  else if (this->kernel_.size() == 3) {
    const auto x_stride = v2a<Size_t, int, 3>(instrides, s);
    const auto x_shape = v2a<Size_t, int, 3>(inshape, s);
    const auto y_shape = v2a<Size_t, int, 3>(outshape, s);
    const auto kernel = v2a<int, int, 3>(this->kernel_);
    const auto stride = v2a<int, int, 3>(this->stride_);
    const auto pad = v2a<int, int, 3>(this->pad_);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int n = 0; n < n_map; n++) {
      forward_map(x + n * x_map_size, y + n * y_map_size, this->including_pad_,
                  x_stride, x_shape, y_shape, kernel, stride, pad);
    }
  }
}

template <typename T>
void AveragePooling<T>::backward_impl(const Variables &inputs,
                                      const Variables &outputs,
                                      const vector<bool> &propagate_down,
                                      const vector<bool> &accum) {
  if (!propagate_down[0])
    return;

  NBLA_CHECK(!this->channel_last_, error_code::not_implemented,
             "The passed argument channel_last=true is not supported in CPU "
             "pooling.");

  if (!accum[0])
    inputs[0]->grad()->zero();

  auto dx = inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_, false);
  auto dy = outputs[0]->get_grad_pointer<T>(this->ctx_);

  const Shape_t &inshape = inputs[0]->shape();
  const Shape_t &outshape = outputs[0]->shape();
  const Shape_t &instrides = inputs[0]->strides();
  const Shape_t &outstrides = outputs[0]->strides();
  const int s = inshape.size() - this->kernel_.size();
  const int x_map_size = (s == 0) ? inputs[0]->size() : instrides[s - 1];
  const int y_map_size = (s == 0) ? outputs[0]->size() : outstrides[s - 1];
  const int n_map = outputs[0]->size() / y_map_size;

  if (this->kernel_.size() == 2) {
    const auto x_stride = v2a<Size_t, int, 2>(instrides, s);
    const auto x_shape = v2a<Size_t, int, 2>(inshape, s);
    const auto y_shape = v2a<Size_t, int, 2>(outshape, s);
    const auto kernel = v2a<int, int, 2>(this->kernel_);
    const auto stride = v2a<int, int, 2>(this->stride_);
    const auto pad = v2a<int, int, 2>(this->pad_);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int n = 0; n < n_map; n++) {
      backward_map(dx + n * x_map_size, dy + n * y_map_size, including_pad_,
                   x_stride, x_shape, y_shape, kernel, stride, pad);
    }
  }

  else if (this->kernel_.size() == 3) {
    const auto x_stride = v2a<Size_t, int, 3>(instrides, s);
    const auto x_shape = v2a<Size_t, int, 3>(inshape, s);
    const auto y_shape = v2a<Size_t, int, 3>(outshape, s);
    const auto kernel = v2a<int, int, 3>(this->kernel_);
    const auto stride = v2a<int, int, 3>(this->stride_);
    const auto pad = v2a<int, int, 3>(this->pad_);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int n = 0; n < n_map; n++) {
      backward_map(dx + n * x_map_size, dy + n * y_map_size, including_pad_,
                   x_stride, x_shape, y_shape, kernel, stride, pad);
    }
  }
}
}
