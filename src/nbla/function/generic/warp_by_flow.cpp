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
#include <nbla/function/warp_by_flow.hpp>
#include <nbla/variable.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(WarpByFlow);

template <typename T>
void WarpByFlow<T>::setup_impl(const Variables &inputs,
                               const Variables &outputs) {
  auto data_shape = inputs[0]->shape();
  auto flow_shape = inputs[1]->shape();

  NBLA_CHECK(data_shape.size() == 4, error_code::value,
             "The input data must have the four dimensions NCHW.");

  NBLA_CHECK(data_shape.size() == flow_shape.size(), error_code::value,
             "The data and flow input shapes must be same length.");

  NBLA_CHECK(flow_shape[0] == data_shape[0], error_code::value,
             "The data and flow input batch size must be identical.");

  NBLA_CHECK(flow_shape[1] == 2, error_code::value,
             "The flow variable must have two channels for a 2D warp.");

  NBLA_CHECK(flow_shape[2] == data_shape[2], error_code::value,
             "The data and flow height dimension must be identical.");

  NBLA_CHECK(flow_shape[3] == data_shape[3], error_code::value,
             "The data and flow width dimension must be identical.");

  outputs[0]->reshape(inputs[0]->shape(), true);
}

template <typename T1, typename T2>
inline T2 clamp_to_index(const T1 value, const T2 maximum) {
  return std::max(T2(0), std::min(maximum, T2(value)));
}

template <typename T>
void WarpByFlow<T>::forward_impl(const Variables &inputs,
                                 const Variables &outputs) {
  auto data = inputs[0]->get_data_pointer<T>(this->ctx_);
  auto flow = inputs[1]->get_data_pointer<T>(this->ctx_);
  auto out = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_, true);

  auto N = outputs[0]->shape().at(0);
  auto C = outputs[0]->shape().at(1);
  auto H = outputs[0]->shape().at(2);
  auto W = outputs[0]->shape().at(3);

  // For data shape (N,C,H,W) the flow shape is (N,2,C,W) where flow[:,0,:,:]
  // corresponds to the horizontal (x) and flow[:,1,:,:] to the vertical (y)
  // component. Note that the outer loop iteration over channels is more cache
  // efficient although it repeats the inner loop index computations.

  for (decltype(N) n = 0; n < N; n++) {
    for (decltype(C) c = 0; c < C; c++) {
      auto flow_x = flow + (H * W) * (n * 2);
      auto flow_y = flow + (H * W) * (n * 2 + 1);
      auto data_c = data + (H * W) * (n * C + c);
      for (decltype(H) y = 0; y < H; y++) {
        for (decltype(W) x = 0; x < W; x++) {
          auto xf = x + *flow_x++;
          auto yf = y + *flow_y++;
          auto xl = clamp_to_index(std::floor(xf), W - 1);
          auto yt = clamp_to_index(std::floor(yf), H - 1);
          auto xr = clamp_to_index(std::floor(xf) + 1, W - 1);
          auto yb = clamp_to_index(std::floor(yf) + 1, H - 1);
          auto tl = data_c[yt * W + xl];
          auto tr = data_c[yt * W + xr];
          auto bl = data_c[yb * W + xl];
          auto br = data_c[yb * W + xr];
          auto alpha = xf - xl, beta = yf - yt;
          auto interp_upper = alpha * (tr - tl) + tl;
          auto interp_lower = alpha * (br - bl) + bl;
          *out++ = beta * (interp_lower - interp_upper) + interp_upper;
        }
      }
    }
  }
}

template <typename T>
void WarpByFlow<T>::backward_impl(const Variables &inputs,
                                  const Variables &outputs,
                                  const vector<bool> &propagate_down,
                                  const vector<bool> &accum) {
  if (!(propagate_down[0] || propagate_down[1])) {
    return;
  }

  auto g_outp = outputs[0]->get_grad_pointer<T>(this->ctx_);

  auto N = outputs[0]->shape().at(0);
  auto C = outputs[0]->shape().at(1);
  auto H = outputs[0]->shape().at(2);
  auto W = outputs[0]->shape().at(3);

  if (propagate_down[0]) {
    if (!accum[0])
      inputs[0]->grad()->zero();

    auto g_data = inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_, false);
    auto flow = inputs[1]->get_data_pointer<T>(this->ctx_);
    auto grad = g_outp;

    for (decltype(N) n = 0; n < N; n++) {
      for (decltype(C) c = 0; c < C; c++) {
        auto flow_x = flow + (H * W) * (n * 2);
        auto flow_y = flow + (H * W) * (n * 2 + 1);
        auto g_data_c = g_data + (H * W) * (n * C + c);
        for (decltype(H) y = 0; y < H; y++) {
          for (decltype(W) x = 0; x < W; x++) {
            auto xf = x + *flow_x++;
            auto yf = y + *flow_y++;
            auto xl = clamp_to_index(std::floor(xf), W - 1);
            auto yt = clamp_to_index(std::floor(yf), H - 1);
            auto xr = clamp_to_index(std::floor(xf) + 1, W - 1);
            auto yb = clamp_to_index(std::floor(yf) + 1, H - 1);
            g_data_c[yt * W + xl] += (1 - xf + xl) * (1 - yf + yt) * (*grad);
            g_data_c[yb * W + xl] += (1 - xf + xl) * (yf - yt) * (*grad);
            g_data_c[yt * W + xr] += (xf - xl) * (1 - yf + yt) * (*grad);
            g_data_c[yb * W + xr] += (xf - xl) * (yf - yt) * (*grad);
            grad++;
          }
        }
      }
    }
  }

  if (propagate_down[1]) {
    if (!accum[1])
      inputs[1]->grad()->zero();

    auto g_flow = inputs[1]->cast_grad_and_get_pointer<T>(this->ctx_, false);
    auto data = inputs[0]->get_data_pointer<T>(this->ctx_);
    auto flow = inputs[1]->get_data_pointer<T>(this->ctx_);
    auto grad = g_outp;

    // Note that the gradient computation w.r.t. flow could also be merged into
    // the gradient computation w.r.t. data (above). However, this may cause
    // worse performance due to memory writes with image height * width stride.

    for (decltype(N) n = 0; n < N; n++) {
      for (decltype(C) c = 0; c < C; c++) {
        auto flow_x = flow + (H * W) * (n * 2);
        auto flow_y = flow + (H * W) * (n * 2 + 1);
        auto data_c = data + (H * W) * (n * C + c);
        auto g_flow_x = g_flow + (H * W) * (n * 2);
        auto g_flow_y = g_flow + (H * W) * (n * 2 + 1);
        for (decltype(H) y = 0; y < H; y++) {
          for (decltype(W) x = 0; x < W; x++) {
            auto xf = x + *flow_x++;
            auto yf = y + *flow_y++;
            auto xl = clamp_to_index(std::floor(xf), W - 1);
            auto yt = clamp_to_index(std::floor(yf), H - 1);
            auto xr = clamp_to_index(std::floor(xf) + 1, W - 1);
            auto yb = clamp_to_index(std::floor(yf) + 1, H - 1);
            auto tl = data_c[yt * W + xl];
            auto tr = data_c[yt * W + xr];
            auto bl = data_c[yb * W + xl];
            auto br = data_c[yb * W + xr];
            auto g1 = (yb - yf) * (tr - tl) + (1 - yb + yf) * (br - bl);
            auto g2 = (xr - xf) * (bl - tl) + (1 - xr + xf) * (br - tr);
            *g_flow_x++ += (*grad) * g1;
            *g_flow_y++ += (*grad) * g2;
            grad++;
          }
        }
      }
    }
  }
}
}
