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
#include <nbla/function/max_pooling.hpp>
#include <nbla/function/max_pooling_backward.hpp>
#include <nbla/utils/nd_index.hpp>
#include <nbla/variable.hpp>

#include <algorithm>
#include <array>
#include <cstring>

namespace nbla {

using std::min;
using std::max;
using std::ceil;

NBLA_REGISTER_FUNCTION_SOURCE(MaxPoolingBackward, const vector<int> &,
                              const vector<int> &, bool, const vector<int> &,
                              bool);

namespace max_pooling_backward {

template <typename T> void zeroing(T *x, int size) {
  for (int i = 0; i < size; ++i) {
    *x++ = T(0);
  }
}

template <typename T>
void max_pooling_2d_forward(T *dx, const T *dy, const T *x, int Cx, int Hx,
                            int Wx, Shape_t xstride, int By, int Cy, int Hy,
                            int Wy, Shape_t ystride, int wkernel, int hkernel,
                            int wstride, int hstride, int wpad, int hpad) {
  auto oidx = 0;
  for (auto b = 0; b < By; ++b) {
    for (auto c = 0; c < Cy; ++c) {
      for (auto h = 0; h < Hy; ++h) {
        for (auto w = 0; w < Wy; ++w) {
          // region
          auto hi_pool_start = h * hstride - hpad;
          auto wi_pool_start = w * wstride - wpad;
          auto hi_pool_end = min(hi_pool_start + hkernel, Hx);
          auto wi_pool_end = min(wi_pool_start + wkernel, Wx);
          hi_pool_start = max(hi_pool_start, 0);
          wi_pool_start = max(wi_pool_start, 0);
          // pool
          auto ind_idx = Shape_t{b, c, hi_pool_start, wi_pool_start};
          auto max_idx = ndi::nd2flat(ind_idx, xstride);
          auto max_val = x[max_idx];
          for (auto rh = hi_pool_start; rh < hi_pool_end; ++rh) {
            for (auto rw = wi_pool_start; rw < wi_pool_end; ++rw) {
              ind_idx = Shape_t{b, c, rh, rw};
              auto iidx = ndi::nd2flat(ind_idx, xstride);
              if (max_val < x[iidx]) {
                max_val = x[iidx];
                max_idx = iidx;
              }
            }
          }
          dx[max_idx] += dy[oidx];
          oidx++;
        }
      }
    }
  }
}

template <typename T>
void max_pooling_3d_forward(T *dx, const T *dy, const T *x, int Cx, int Dx,
                            int Hx, int Wx, Shape_t xstride, int By, int Cy,
                            int Dy, int Hy, int Wy, Shape_t ystride,
                            int wkernel, int hkernel, int dkernel, int wstride,
                            int hstride, int dstride, int wpad, int hpad,
                            int dpad) {
  auto oidx = 0;
  for (auto b = 0; b < By; ++b) {
    for (auto c = 0; c < Cy; ++c) {
      for (auto d = 0; d < Dy; ++d) {
        for (auto h = 0; h < Hy; ++h) {
          for (auto w = 0; w < Wy; ++w) {
            // region
            auto di_pool_start = d * dstride - dpad;
            auto hi_pool_start = h * hstride - hpad;
            auto wi_pool_start = w * wstride - wpad;
            auto di_pool_end = min(di_pool_start + dkernel, Dx);
            auto hi_pool_end = min(hi_pool_start + hkernel, Hx);
            auto wi_pool_end = min(wi_pool_start + wkernel, Wx);
            di_pool_start = max(di_pool_start, 0);
            hi_pool_start = max(hi_pool_start, 0);
            wi_pool_start = max(wi_pool_start, 0);
            // pool
            auto ind_idx =
                Shape_t{b, c, di_pool_start, hi_pool_start, wi_pool_start};
            auto max_idx = ndi::nd2flat(ind_idx, xstride);
            auto max_val = x[max_idx];
            for (auto rd = di_pool_start; rd < di_pool_end; ++rd) {
              for (auto rh = hi_pool_start; rh < hi_pool_end; ++rh) {
                for (auto rw = wi_pool_start; rw < wi_pool_end; ++rw) {
                  ind_idx = Shape_t{b, c, rd, rh, rw};
                  auto iidx = ndi::nd2flat(ind_idx, xstride);
                  if (max_val < x[iidx]) {
                    max_val = x[iidx];
                    max_idx = iidx;
                  }
                }
              }
            }
            dx[max_idx] += dy[oidx];
            oidx++;
          }
        }
      }
    }
  }
}

template <typename T, bool accum>
void max_pooling_2d_backward(T *gdy, const T *gdx, const T *x, int Cx, int Hx,
                             int Wx, Shape_t xstride, int By, int Cy, int Hy,
                             int Wy, Shape_t ystride, int wkernel, int hkernel,
                             int wstride, int hstride, int wpad, int hpad) {
  auto oidx = 0;
  for (auto b = 0; b < By; ++b) {
    for (auto c = 0; c < Cy; ++c) {
      for (auto h = 0; h < Hy; ++h) {
        for (auto w = 0; w < Wy; ++w) {
          // region
          auto hi_pool_start = h * hstride - hpad;
          auto wi_pool_start = w * wstride - wpad;
          auto hi_pool_end = min(hi_pool_start + hkernel, Hx);
          auto wi_pool_end = min(wi_pool_start + wkernel, Wx);
          hi_pool_start = max(hi_pool_start, 0);
          wi_pool_start = max(wi_pool_start, 0);
          // pool
          auto ind_idx = Shape_t{b, c, hi_pool_start, wi_pool_start};
          auto max_idx = ndi::nd2flat(ind_idx, xstride);
          auto max_val = x[max_idx];
          for (auto rh = hi_pool_start; rh < hi_pool_end; ++rh) {
            for (auto rw = wi_pool_start; rw < wi_pool_end; ++rw) {
              ind_idx = Shape_t{b, c, rh, rw};
              auto iidx = ndi::nd2flat(ind_idx, xstride);
              if (max_val < x[iidx]) {
                max_val = x[iidx];
                max_idx = iidx;
              }
            }
          }
          gdy[oidx] = accum ? gdy[oidx] + gdx[max_idx] : gdx[max_idx];
          oidx++;
        }
      }
    }
  }
}

template <typename T, bool accum>
void max_pooling_3d_backward(T *gdy, const T *gdx, const T *x, int Cx, int Dx,
                             int Hx, int Wx, Shape_t xstride, int By, int Cy,
                             int Dy, int Hy, int Wy, Shape_t ystride,
                             int wkernel, int hkernel, int dkernel, int wstride,
                             int hstride, int dstride, int wpad, int hpad,
                             int dpad) {
  auto oidx = 0;
  for (auto b = 0; b < By; ++b) {
    for (auto c = 0; c < Cy; ++c) {
      for (auto d = 0; d < Dy; ++d) {
        for (auto h = 0; h < Hy; ++h) {
          for (auto w = 0; w < Wy; ++w) {
            // region
            auto di_pool_start = d * dstride - dpad;
            auto hi_pool_start = h * hstride - hpad;
            auto wi_pool_start = w * wstride - wpad;
            auto di_pool_end = min(di_pool_start + dkernel, Dx);
            auto hi_pool_end = min(hi_pool_start + hkernel, Hx);
            auto wi_pool_end = min(wi_pool_start + wkernel, Wx);
            di_pool_start = max(di_pool_start, 0);
            hi_pool_start = max(hi_pool_start, 0);
            wi_pool_start = max(wi_pool_start, 0);
            // pool
            auto ind_idx =
                Shape_t{b, c, di_pool_start, hi_pool_start, wi_pool_start};
            auto max_idx = ndi::nd2flat(ind_idx, xstride);
            auto max_val = x[max_idx];
            for (auto rd = di_pool_start; rd < di_pool_end; ++rd) {
              for (auto rh = hi_pool_start; rh < hi_pool_end; ++rh) {
                for (auto rw = wi_pool_start; rw < wi_pool_end; ++rw) {
                  ind_idx = Shape_t{b, c, rd, rh, rw};
                  auto iidx = ndi::nd2flat(ind_idx, xstride);
                  if (max_val < x[iidx]) {
                    max_val = x[iidx];
                    max_idx = iidx;
                  }
                }
              }
            }
            gdy[oidx] = accum ? gdy[oidx] + gdx[max_idx] : gdx[max_idx];
            oidx++;
          }
        }
      }
    }
  }
}
}

template <typename T>
void MaxPoolingBackward<T>::setup_impl(const Variables &inputs,
                                       const Variables &outputs) {
  outputs[0]->reshape(inputs[1]->shape(), true);
}

template <typename T>
void MaxPoolingBackward<T>::forward_impl(const Variables &inputs,
                                         const Variables &outputs) {
  NBLA_CHECK(!this->channel_last_, error_code::not_implemented,
             "The passed argument channel_last=true is not supported in CPU "
             "pooling.");
  // inputs[0]  : dy
  // inputs[1]  : x
  // outputs[0] : dx
  // dx = df(dy, x)

  auto sdim = this->kernel_.size();
  auto yshape = inputs[0]->shape();
  auto xshape = inputs[1]->shape();
  int ndim = xshape.size();
  // data
  auto dy = inputs[0]->get_data_pointer<T>(this->ctx_);
  auto x = inputs[1]->get_data_pointer<T>(this->ctx_);
  auto dx = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_, false);
  // zeroing
  max_pooling_backward::zeroing(dx, outputs[0]->size());
  if (sdim == 2) {
    // pool params
    int hstride = this->stride_[0];
    int wstride = this->stride_[1];
    int hpad = this->pad_[0];
    int wpad = this->pad_[1];
    int hkernel = this->kernel_[0];
    int wkernel = this->kernel_[1];
    int Cx = xshape[ndim - 3];
    int Hx = xshape[ndim - 2];
    int Wx = xshape[ndim - 1];
    int Cy = yshape[ndim - 3];
    int Hy = yshape[ndim - 2];
    int Wy = yshape[ndim - 1];
    int By = inputs[0]->size() / (Cy * Hy * Wy);
    auto ystride = ndi::strides(Shape_t{By, Cy, Hy, Wy});
    auto xstride = ndi::strides(Shape_t{By, Cx, Hx, Wx});
    // pool
    max_pooling_backward::max_pooling_2d_forward(
        dx, dy, x, Cx, Hx, Wx, xstride, By, Cy, Hy, Wy, ystride, wkernel,
        hkernel, wstride, hstride, wpad, hpad);
  } else if (sdim == 3) {
    // pool params
    int dstride = this->stride_[0];
    int hstride = this->stride_[1];
    int wstride = this->stride_[2];
    int dpad = this->pad_[0];
    int hpad = this->pad_[1];
    int wpad = this->pad_[2];
    int dkernel = this->kernel_[0];
    int hkernel = this->kernel_[1];
    int wkernel = this->kernel_[2];
    int Cx = xshape[ndim - 4];
    int Dx = xshape[ndim - 3];
    int Hx = xshape[ndim - 2];
    int Wx = xshape[ndim - 1];
    int Cy = yshape[ndim - 4];
    int Dy = yshape[ndim - 3];
    int Hy = yshape[ndim - 2];
    int Wy = yshape[ndim - 1];
    int By = inputs[0]->size() / (Cy * Dy * Hy * Wy);
    auto ystride = ndi::strides(Shape_t{By, Cy, Dy, Hy, Wy});
    auto xstride = ndi::strides(Shape_t{By, Cx, Dx, Hx, Wx});
    // pool
    max_pooling_backward::max_pooling_3d_forward(
        dx, dy, x, Cx, Dx, Hx, Wx, xstride, By, Cy, Dy, Hy, Wy, ystride,
        wkernel, hkernel, dkernel, wstride, hstride, dstride, wpad, hpad, dpad);
  }
}

template <typename T>
void MaxPoolingBackward<T>::backward_impl(const Variables &inputs,
                                          const Variables &outputs,
                                          const vector<bool> &propagate_down,
                                          const vector<bool> &accum) {
  // inputs[0]  : dy
  // inputs[1]  : x
  // outputs[0] : dx
  // dx = df(dy, x)
  // gdy = gdf(gdx, x)

  NBLA_CHECK(!this->channel_last_, error_code::not_implemented,
             "The passed argument channel_last=true is not supported in CPU "
             "pooling.");

  if (!propagate_down[0]) {
    return;
  }

  auto sdim = this->kernel_.size();
  auto yshape = inputs[0]->shape();
  auto xshape = inputs[1]->shape();
  int ndim = xshape.size();
  // data
  auto gdy = inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_, !accum[0]);
  auto x = inputs[1]->get_data_pointer<T>(this->ctx_);
  auto gdx = outputs[0]->get_grad_pointer<T>(this->ctx_);

  if (sdim == 2) {
    // pool params
    int hstride = this->stride_[0];
    int wstride = this->stride_[1];
    int hpad = this->pad_[0];
    int wpad = this->pad_[1];
    int hkernel = this->kernel_[0];
    int wkernel = this->kernel_[1];
    int Cx = xshape[ndim - 3];
    int Hx = xshape[ndim - 2];
    int Wx = xshape[ndim - 1];
    int Cy = yshape[ndim - 3];
    int Hy = yshape[ndim - 2];
    int Wy = yshape[ndim - 1];
    int By = inputs[0]->size() / (Cy * Hy * Wy);
    auto ystride = ndi::strides(Shape_t{By, Cy, Hy, Wy});
    auto xstride = ndi::strides(Shape_t{By, Cx, Hx, Wx});
    // pool
    auto backward =
        accum[0] ? max_pooling_backward::max_pooling_2d_backward<T, true>
                 : max_pooling_backward::max_pooling_2d_backward<T, false>;
    backward(gdy, gdx, x, Cx, Hx, Wx, xstride, By, Cy, Hy, Wy, ystride, wkernel,
             hkernel, wstride, hstride, wpad, hpad);
  } else if (sdim == 3) {
    // pool params
    int dstride = this->stride_[0];
    int hstride = this->stride_[1];
    int wstride = this->stride_[2];
    int dpad = this->pad_[0];
    int hpad = this->pad_[1];
    int wpad = this->pad_[2];
    int dkernel = this->kernel_[0];
    int hkernel = this->kernel_[1];
    int wkernel = this->kernel_[2];
    int Cx = xshape[ndim - 4];
    int Dx = xshape[ndim - 3];
    int Hx = xshape[ndim - 2];
    int Wx = xshape[ndim - 1];
    int Cy = yshape[ndim - 4];
    int Dy = yshape[ndim - 3];
    int Hy = yshape[ndim - 2];
    int Wy = yshape[ndim - 1];
    int By = inputs[0]->size() / (Cy * Dy * Hy * Wy);
    auto ystride = ndi::strides(Shape_t{By, Cy, Dy, Hy, Wy});
    auto xstride = ndi::strides(Shape_t{By, Cx, Dx, Hx, Wx});
    // pool
    auto backward =
        accum[0] ? max_pooling_backward::max_pooling_3d_backward<T, true>
                 : max_pooling_backward::max_pooling_3d_backward<T, false>;
    backward(gdy, gdx, x, Cx, Dx, Hx, Wx, xstride, By, Cy, Dy, Hy, Wy, ystride,
             wkernel, hkernel, dkernel, wstride, hstride, dstride, wpad, hpad,
             dpad);
  }
}
}
