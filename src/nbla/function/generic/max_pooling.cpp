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

// MaxPooling.cpp

#include <nbla/array.hpp>
#include <nbla/function/max_pooling.hpp>
#include <nbla/variable.hpp>

#include <algorithm>
#include <cstring>

namespace nbla {

using std::min;
using std::max;
using std::ceil;

NBLA_REGISTER_FUNCTION_SOURCE(MaxPooling, const vector<int> &,
                              const vector<int> &, bool, const vector<int> &);

template <typename T>
void MaxPooling<T>::setup_impl(const Variables &inputs,
                               const Variables &outputs) {
  BasePooling<T, const vector<int> &, const vector<int> &, bool,
              const vector<int> &>::setup_impl(inputs, outputs);
  max_idx_.reshape(outputs[0]->shape(), true);
  forward_done_ = false;
}

template <class T>
void MaxPooling<T>::forward_impl(const Variables &inputs,
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
  int *m = max_idx_.cast_data_and_get_pointer<int>(this->ctx_);
  for (int n = 0; n < n_map; ++n) {
    for (int iy = 0; iy < hy; ++iy) {
      for (int jy = 0; jy < wy; ++jy) {
        int hstart = iy * hstride - hpad;
        int wstart = jy * wstride - wpad;
        int hend = min(hstart + hkernel, hx + hpad);
        int wend = min(wstart + wkernel, wx + wpad);
        hstart = max(hstart, 0);
        wstart = max(wstart, 0);
        hend = min(hend, hx);
        wend = min(wend, wx);
        const int k = iy * wy + jy;
        const int l = hstart * wx + wstart;
        int max_idx = l;
        T max_val = x[l];
        for (int ix = hstart; ix < hend; ++ix) {
          for (int jx = ix * wx + wstart; jx < ix * wx + wend; ++jx) {
            T val = x[jx];
            if (max_val < val) {
              max_idx = jx;
              max_val = val;
            }
          }
        }
        m[k] = max_idx;
        y[k] = max_val;
      }
    }
    x += x_stride;
    y += y_stride;
    m += y_stride;
  }
  forward_done_ = true;
}

template <class T>
void MaxPooling<T>::backward_impl(const Variables &inputs,
                                  const Variables &outputs,
                                  const vector<bool> &propagate_down,
                                  const vector<bool> &accum) {
  if (!propagate_down[0])
    return;

  NBLA_CHECK(forward_done_, error_code::value,
             "Forward must be called before calling backward.");
  if (!accum[0])
    inputs[0]->grad()->zero();
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

  const int *m = max_idx_.get_data_pointer<int>(this->ctx_);
  for (int n = 0; n < n_map; ++n) {
    for (int iy = 0; iy < hy; ++iy) {
      for (int jy = 0; jy < wy; ++jy) {
        int hstart = iy * hstride - hpad;
        int wstart = jy * wstride - wpad;
        int hend = min(hstart + hkernel, hx + hpad);
        int wend = min(wstart + wkernel, wx + wpad);
        hstart = max(hstart, 0);
        wstart = max(wstart, 0);
        hend = min(hend, hx);
        wend = min(wend, wx);
        const int k = iy * wy + jy;

        dx[m[k]] += dy[k];
      }
    }
    dx += x_stride;
    dy += y_stride;
    m += y_stride;
  }
}
}
