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
#include <nbla/common.hpp>
#include <nbla/function/adaptive_separable_convolution.hpp>
#include <nbla/variable.hpp>

// TODO: remove the following headers if not used.
#include <iostream>
#include <typeinfo>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(AdaptiveSeparableConvolution);

template <typename T>
void AdaptiveSeparableConvolution<T>::setup_impl(const Variables &inputs,
                                                 const Variables &outputs) {
  auto x = inputs[0];
  auto kv = inputs[1];
  auto kh = inputs[2];
  // Input shape
  auto B = x->shape()[0];
  auto C = x->shape()[1];
  auto H = x->shape()[2];
  auto W = x->shape()[3];
  NBLA_CHECK(B == kv->shape()[0] && B == kh->shape()[0], error_code::value,
             "Batch size for each input must same. Batch sizes = (%d, %d, %d).",
             B, kv->shape()[0], kv->shape()[0]);
  // Output shape
  auto oH = x->shape()[2] - kv->shape()[1] + 1;
  auto oW = x->shape()[3] - kh->shape()[1] + 1;
  NBLA_CHECK(oH > 0 && oW > 0, error_code::value,
             "Both the input height (%d) - the virtial filter size (%d) "
             "and the input width (%d) - the horizontal filter size (%d) "
             "must be greater than 0.",
             x->shape()[2], oH, x->shape()[3], oW);
  NBLA_CHECK(
      kv->shape()[2] >= oH && kv->shape()[2] <= H, error_code::value,
      "Height of the vertical filter must be in "
      "[the input height - the virtial filter size + 1, the input height], "
      "[%d, %d]",
      oH, H);
  NBLA_CHECK(
      kh->shape()[3] >= oW && kh->shape()[3] <= W, error_code::value,
      "Width of the horizontal filter must be in "
      "[the input width - the horizontal filter size + 1, the input width], "
      "[%d, %d]",
      oW, W);
  outputs[0]->reshape({B, C, oH, oW}, true);
}

template <typename T>
void AdaptiveSeparableConvolution<T>::forward_impl(const Variables &inputs,
                                                   const Variables &outputs) {
  // Inputs and outputs
  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  const T *kv = inputs[1]->get_data_pointer<T>(this->ctx_);
  const T *kh = inputs[2]->get_data_pointer<T>(this->ctx_);
  T *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_, true);
  // Output shape
  auto B = outputs[0]->shape()[0];
  auto C = outputs[0]->shape()[1];
  auto H = outputs[0]->shape()[2];
  auto W = outputs[0]->shape()[3];
  // Output and input strides
  auto Y_Bs = outputs[0]->strides()[0];
  auto Y_Cs = outputs[0]->strides()[1];
  auto Y_Hs = outputs[0]->strides()[2];
  auto X_Bs = inputs[0]->strides()[0];
  auto X_Cs = inputs[0]->strides()[1];
  auto X_Hs = inputs[0]->strides()[2];
  // Filter size
  auto Kv = inputs[1]->shape()[1];
  auto Kh = inputs[2]->shape()[1];
  // Filter strides
  auto KV_Bs = inputs[1]->strides()[0];
  auto KV_Cs = inputs[1]->strides()[1];
  auto KV_Hs = inputs[1]->strides()[2];
  auto KH_Bs = inputs[2]->strides()[0];
  auto KH_Cs = inputs[2]->strides()[1];
  auto KH_Hs = inputs[2]->strides()[2];

  // Convolve approximately
  for (int b = 0; b < B; ++b) {
    // filter at b
    auto kv_b = kv + (KV_Bs * b);
    auto kh_b = kh + (KH_Bs * b);
    for (int c = 0; c < C; ++c) {
      // input and output at b and c
      auto x_bc = x + (X_Bs * b + X_Cs * c);
      auto y_bc = y + (Y_Bs * b + Y_Cs * c);
      for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
          // sum_{i, j} K_h(i, h, w) * K_v(j, h, w) * I(c, h+j, w+i)
          T val = T(0.0);
          for (int j = 0; j < Kv; ++j) {
            for (int i = 0; i < Kh; ++i) {
              auto kval = kv_b[KV_Cs * j + KV_Hs * h + w] *
                          kh_b[KH_Cs * i + KH_Hs * h + w];
              auto pval = x_bc[X_Hs * (h + j) + (w + i)];
              val += kval * pval;
            }
          }
          y_bc[Y_Hs * h + w] = val;
        }
      }
    }
  }
}

template <typename T>
void AdaptiveSeparableConvolution<T>::backward_impl(
    const Variables &inputs, const Variables &outputs,
    const vector<bool> &propagate_down, const vector<bool> &accum) {
  if (!(propagate_down[0] || propagate_down[1] || propagate_down[2])) {
    return;
  }
  // Inputs
  const T *x = inputs[0]->get_data_pointer<T>(this->ctx_);
  const T *kv = inputs[1]->get_data_pointer<T>(this->ctx_);
  const T *kh = inputs[2]->get_data_pointer<T>(this->ctx_);
  // Output gradient
  const T *g_y = outputs[0]->get_grad_pointer<T>(this->ctx_);
  // Input gradient
  T *g_x{nullptr};
  T *g_kv{nullptr};
  T *g_kh{nullptr};
  // Output shape
  auto B = outputs[0]->shape()[0];
  auto C = outputs[0]->shape()[1];
  auto H = outputs[0]->shape()[2];
  auto W = outputs[0]->shape()[3];
  // Output and input strides
  auto Y_Bs = outputs[0]->strides()[0];
  auto Y_Cs = outputs[0]->strides()[1];
  auto Y_Hs = outputs[0]->strides()[2];
  auto X_Bs = inputs[0]->strides()[0];
  auto X_Cs = inputs[0]->strides()[1];
  auto X_Hs = inputs[0]->strides()[2];
  // Filter size
  auto Kv = inputs[1]->shape()[1];
  auto Kh = inputs[2]->shape()[1];
  // Filter strides
  auto KV_Bs = inputs[1]->strides()[0];
  auto KV_Cs = inputs[1]->strides()[1];
  auto KV_Hs = inputs[1]->strides()[2];
  auto KH_Bs = inputs[2]->strides()[0];
  auto KH_Cs = inputs[2]->strides()[1];
  auto KH_Hs = inputs[2]->strides()[2];

  // w.r.t x
  if (propagate_down[0]) {
    // Always accumulated since one pixel value in the input are used for many
    // outputs,
    // meaning a broadcast in the forward pass, then a summation in the backawrd
    // pass
    g_x = inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_, false);
    for (int b = 0; b < B; ++b) {
      auto kv_b = kv + (KV_Bs * b);
      auto kh_b = kh + (KH_Bs * b);
      for (int c = 0; c < C; ++c) {
        auto g_x_bc = g_x + (X_Bs * b + X_Cs * c);
        for (int h = 0; h < H; ++h) {
          for (int w = 0; w < W; ++w) {
            auto g_y_bchw = *(g_y + (Y_Bs * b + Y_Cs * c + Y_Hs * h + w));
            // g_y(c, h, w) * Kv(j, h, w) * Kh(i, h, w)
            for (int j = 0; j < Kv; ++j) {
              for (int i = 0; i < Kh; ++i) {
                auto val = g_y_bchw * kv_b[KV_Cs * j + KV_Hs * h + w] *
                           kh_b[KH_Cs * i + KH_Hs * h + w];
                g_x_bc[X_Hs * (h + j) + (w + i)] += val;
              }
            }
          }
        }
      }
    }
  }
  // w.r.t. virtical weight
  if (propagate_down[1]) {
    g_kv = inputs[1]->cast_grad_and_get_pointer<T>(this->ctx_, !accum[1]);
    for (int b = 0; b < B; ++b) {
      auto x_b = x + (X_Bs * b);
      auto kh_b = kh + (KH_Bs * b);
      auto g_y_b = g_y + (Y_Bs * b);
      auto g_kv_b = g_kv + (KV_Bs * b);
      for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
          for (int j = 0; j < Kv; ++j) {
            // sum_{c} (sum_{i} K_h(i, h, w) * I(c, h+j, w+i)) * g_y(c, h, w))
            auto osum = T(0.0);
            for (int c = 0; c < C; ++c) {
              auto isum = T(0.0);
              for (int i = 0; i < Kh; ++i) {
                isum += kh_b[KH_Cs * i + KH_Hs * h + w] *
                        x_b[X_Cs * c + X_Hs * (h + j) + (w + i)];
              }
              osum += g_y_b[Y_Cs * c + Y_Hs * h + w] * isum;
            }
            if (accum[1])
              g_kv_b[KV_Cs * j + KV_Hs * h + w] += osum;
            else
              g_kv_b[KV_Cs * j + KV_Hs * h + w] = osum;
          }
        }
      }
    }
  }
  // w.r.t. horizontal weight
  if (propagate_down[2]) {
    g_kh = inputs[2]->cast_grad_and_get_pointer<T>(this->ctx_, !accum[2]);
    for (int b = 0; b < B; ++b) {
      auto x_b = x + (X_Bs * b);
      auto kv_b = kv + (KV_Bs * b);
      auto g_y_b = g_y + (Y_Bs * b);
      auto g_kh_b = g_kh + (KH_Bs * b);
      for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
          for (int i = 0; i < Kh; ++i) {
            // sum_{c} (sum_{j} K_v(j, h, w) * I(c, h+j, w+i)) * g_y(c, h, w))
            auto osum = T(0.0);
            for (int c = 0; c < C; ++c) {
              auto isum = T(0.0);
              for (int j = 0; j < Kv; ++j) {
                isum += kv_b[KV_Cs * j + KV_Hs * h + w] *
                        x_b[X_Cs * c + X_Hs * (h + j) + (w + i)];
              }
              osum += g_y_b[Y_Cs * c + Y_Hs * h + w] * isum;
            }
            if (accum[2])
              g_kh_b[KH_Cs * i + KH_Hs * h + w] += osum;
            else
              g_kh_b[KH_Cs * i + KH_Hs * h + w] = osum;
          }
        }
      }
    }
  }
}
}
