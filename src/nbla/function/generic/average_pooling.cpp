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
#include <cstring>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(AveragePooling, const vector<int> &,
                              const vector<int> &, bool, const vector<int> &,
                              bool);

using std::min;
using std::max;

template <class T>
void AveragePooling<T>::forward_impl(const Variables &inputs,
                                     const Variables &outputs) {

  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  T *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_);

  const Shape_t inshape = inputs[0]->shape();
  const Shape_t outshape = outputs[0]->shape();
  const int s = inshape.size() - this->kernel_.size();
  const int x_stride =
      (s == 0) ? inputs[0]->size() : inputs[0]->strides()[s - 1];
  const int y_stride =
      (s == 0) ? outputs[0]->size() : outputs[0]->strides()[s - 1];
  const int hx = inshape[s + 0];
  const int wx = inshape[s + 1];
  const int hy = outshape[s + 0];
  const int wy = outshape[s + 1];
  const int hkernel = this->kernel_[0];
  const int wkernel = this->kernel_[1];
  const int hstride = this->stride_[0];
  const int wstride = this->stride_[1];
  const int hpad = this->pad_[0];
  const int wpad = this->pad_[1];
  const int n_map = inputs[0]->size() / x_stride;
  for (int n = 0; n < n_map; ++n) {
    for (int iy = 0; iy < hy; ++iy) {
      for (int jy = 0; jy < wy; ++jy) {
        int hstart = iy * hstride - hpad;
        int wstart = jy * wstride - wpad;
        int hend = min(hstart + hkernel, hx + hpad);
        int wend = min(wstart + wkernel, wx + wpad);
        int pool_size = (hend - hstart) * (wend - wstart);
        hstart = max(hstart, 0);
        wstart = max(wstart, 0);
        hend = min(hend, hx);
        wend = min(wend, wx);
        if (including_pad_ == false)
          pool_size = (hend - hstart) * (wend - wstart);
        const int k = iy * wy + jy;
        T yk = 0;
        for (int ix = hstart; ix < hend; ++ix) {
          for (int jx = ix * wx + wstart; jx < ix * wx + wend; ++jx) {
            yk += x[jx];
          }
        }
        y[k] = yk / pool_size;
      }
    }
    x += x_stride;
    y += y_stride;
  }
}

template <class T>
void AveragePooling<T>::backward_impl(const Variables &inputs,
                                      const Variables &outputs,
                                      const vector<bool> &propagate_down,
                                      const vector<bool> &accum) {
  if (!propagate_down[0])
    return;

  T *dx = inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_);
  const T *dy = outputs[0]->get_grad_pointer<T>(this->ctx_);

  const Shape_t inshape = inputs[0]->shape();
  const Shape_t outshape = outputs[0]->shape();
  const int s = inshape.size() - this->kernel_.size();
  const int x_stride =
      (s == 0) ? inputs[0]->size() : inputs[0]->strides()[s - 1];
  const int y_stride =
      (s == 0) ? outputs[0]->size() : outputs[0]->strides()[s - 1];
  const int hx = inshape[s + 0];
  const int wx = inshape[s + 1];
  const int hy = outshape[s + 0];
  const int wy = outshape[s + 1];
  const int hkernel = this->kernel_[0];
  const int wkernel = this->kernel_[1];
  const int hstride = this->stride_[0];
  const int wstride = this->stride_[1];
  const int hpad = this->pad_[0];
  const int wpad = this->pad_[1];
  const int n_map = outputs[0]->size() / y_stride;
  for (int n = 0; n < n_map; ++n) {
    for (int iy = 0; iy < hy; ++iy) {
      for (int jy = 0; jy < wy; ++jy) {
        int hstart = iy * hstride - hpad;
        int wstart = jy * wstride - wpad;
        int hend = min(hstart + hkernel, hx + hpad);
        int wend = min(wstart + wkernel, wx + wpad);
        int pool_size = (hend - hstart) * (wend - wstart);
        hstart = max(hstart, 0);
        wstart = max(wstart, 0);
        hend = min(hend, hx);
        wend = min(wend, wx);
        if (including_pad_ == false)
          pool_size = (hend - hstart) * (wend - wstart);
        const int k = iy * wy + jy;
        const T dyk = dy[k] / pool_size;
        for (int ix = hstart; ix < hend; ++ix) {
          for (int jx = ix * wx + wstart; jx < ix * wx + wend; ++jx) {
            dx[jx] = (accum[0] ? dx[jx] : 0) + dyk;
          }
        }
      }
    }
    dx += x_stride;
    dy += y_stride;
  }
}

}
