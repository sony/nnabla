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
#include <nbla/function/patch_correlation.hpp>
#include <nbla/variable.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(PatchCorrelation, const vector<int> &,
                              const vector<int> &, const vector<int> &,
                              const vector<int> &, const vector<int> &);

struct Shape2D {
  int h; // height
  int w; // width
  Shape2D(const vector<int> v) {
    h = v.at(0);
    w = v.at(1);
  };
};

struct Pad2D {
  int t; // top
  int b; // bottom
  int l; // left
  int r; // right
  int h;
  int w;
  Pad2D(const vector<int> v) {
    t = v.at(0);
    b = v.at(1);
    l = v.at(2);
    r = v.at(3);
    h = t + b;
    w = l + r;
  };
};

template <typename T>
void PatchCorrelation<T>::setup_impl(const Variables &inputs,
                                     const Variables &outputs) {
#define CHECK_VALUE(condition, msg, ...)                                       \
  NBLA_CHECK(condition, error_code::value, msg, ##__VA_ARGS__);

  auto const x1 = inputs.at(0);
  auto const x2 = inputs.at(1);

  CHECK_VALUE(x1->shape().size() == 4, "The input x1 must have 4 dimensions.");
  CHECK_VALUE(x2->shape().size() == 4, "The input x2 must have 4 dimensions.");
  CHECK_VALUE(x1->shape() == x2->shape(), "x1 and x2 must have same shape.");

  auto const patch = Shape2D(this->patch_);
  auto const shift = Shape2D(this->shift_);
  auto const patch_step = Shape2D(this->patch_step_);
  auto const shift_step = Shape2D(this->shift_step_);
  auto const padding = Pad2D(this->padding_);

  CHECK_VALUE(patch.h > 0, "patch height must be greater than zero.");
  CHECK_VALUE(patch.w > 0, "patch width must be greater than zero.");
  CHECK_VALUE(shift.h >= 0, "shift height must not be negative.");
  CHECK_VALUE(shift.w >= 0, "shift width must not be negative.");
  CHECK_VALUE(patch_step.h > 0, "patch_step height must be greater than zero.");
  CHECK_VALUE(patch_step.w > 0, "patch_step width must be greater than zero.");
  CHECK_VALUE(shift_step.h > 0, "shift_step height must be greater than zero.");
  CHECK_VALUE(shift_step.w > 0, "shift_step width must be greater than zero.");
  CHECK_VALUE(padding.t >= 0, "top padding must not be negative.");
  CHECK_VALUE(padding.b >= 0, "bottom padding must not be negative.");
  CHECK_VALUE(padding.l >= 0, "left padding must not be negative.");
  CHECK_VALUE(padding.r >= 0, "right padding must not be negative.");

  auto const N = x1->shape()[0];
  auto const H = x1->shape()[1];
  auto const W = x1->shape()[2];

  auto const och = (2 * shift.h + shift_step.h) / shift_step.h;
  auto const ocw = (2 * shift.w + shift_step.w) / shift_step.w;
  auto const oh = (H + padding.h - patch.h + patch_step.h) / patch_step.h;
  auto const ow = (W + padding.w - patch.w + patch_step.w) / patch_step.w;

  outputs.at(0)->reshape({N, oh, ow, och, ocw}, true);

#undef CHECK_VALUE
}

template <typename T>
void PatchCorrelation<T>::forward_impl(const Variables &inputs,
                                       const Variables &outputs) {
  auto in1_data = inputs[0]->get_data_pointer<T>(this->ctx_);
  auto in2_data = inputs[1]->get_data_pointer<T>(this->ctx_);
  auto out_data = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_, true);

  auto const patch = Shape2D(this->patch_);
  auto const shift = Shape2D(this->shift_);
  auto const patch_step = Shape2D(this->patch_step_);
  auto const shift_step = Shape2D(this->shift_step_);
  auto const pad = Pad2D(this->padding_);

  auto const N = inputs[0]->shape()[0];
  auto const H = inputs[0]->shape()[1];
  auto const W = inputs[0]->shape()[2];
  auto const C = inputs[0]->shape()[3];

  auto flat = [N, H, W, C](decltype(N) n, decltype(H) y, decltype(W) x) {
    return n * H * W * C + y * W * C + x * C;
  };

  auto out_iter = Size_t{0};

  for (auto n = decltype(N){0}; n < N; n++) {
    for (auto y = -pad.t; y <= H - patch.h + pad.b; y += patch_step.h) {
      for (auto x = -pad.l; x <= W - patch.w + pad.r; x += patch_step.w) {
        for (auto yy = -shift.h; yy <= shift.h; yy += shift_step.h) {
          for (auto xx = -shift.w; xx <= shift.w; xx += shift_step.w) {
            auto value = T{0};
            for (auto ky = decltype(patch.h){0}; ky < patch.h; ky++) {
              if ((0 <= y + ky) && (y + ky < H) && (0 <= y + yy + ky) &&
                  (y + yy + ky < H)) {
                for (auto kx = decltype(patch.w){0}; kx < patch.w; kx++) {
                  if ((0 <= x + kx) && (x + kx < W) && (0 <= x + xx + kx) &&
                      (x + xx + kx < W)) {
                    auto i1 = flat(n, y + ky, x + kx);
                    auto i2 = flat(n, y + yy + ky, x + xx + kx);
                    for (auto c = decltype(C){0}; c < C; c++) {
                      value += in1_data[i1++] * in2_data[i2++];
                    }
                  }
                }
              }
            }
            out_data[out_iter++] = value;
          }
        }
      }
    }
  }
}

template <typename T>
void PatchCorrelation<T>::backward_impl(const Variables &inputs,
                                        const Variables &outputs,
                                        const vector<bool> &propagate_down,
                                        const vector<bool> &accum) {
  if (!(propagate_down[0] || propagate_down[1])) {
    return;
  }

  if ((propagate_down[0]) && (!accum[0]))
    inputs[0]->grad()->zero();

  if ((propagate_down[1]) && (!accum[1]))
    inputs[1]->grad()->zero();

  auto in1_grad = inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_, false);
  auto in2_grad = inputs[1]->cast_grad_and_get_pointer<T>(this->ctx_, false);
  auto out_grad = outputs[0]->get_grad_pointer<T>(this->ctx_);

  auto in1_data = inputs[0]->get_data_pointer<T>(this->ctx_);
  auto in2_data = inputs[1]->get_data_pointer<T>(this->ctx_);

  auto const patch = Shape2D(this->patch_);
  auto const shift = Shape2D(this->shift_);
  auto const patch_step = Shape2D(this->patch_step_);
  auto const shift_step = Shape2D(this->shift_step_);
  auto const pad = Pad2D(this->padding_);

  auto const N = inputs[0]->shape()[0];
  auto const H = inputs[0]->shape()[1];
  auto const W = inputs[0]->shape()[2];
  auto const C = inputs[0]->shape()[3];

  auto flat = [N, H, W, C](decltype(N) n, decltype(H) y, decltype(W) x) {
    return n * H * W * C + y * W * C + x * C;
  };

  auto out_iter = Size_t{0};

  for (auto n = decltype(N){0}; n < N; n++) {
    for (auto y = -pad.t; y <= H - patch.h + pad.b; y += patch_step.h) {
      for (auto x = -pad.l; x <= W - patch.w + pad.r; x += patch_step.w) {
        for (auto yy = -shift.h; yy <= shift.h; yy += shift_step.h) {
          for (auto xx = -shift.w; xx <= shift.w; xx += shift_step.w) {
            auto grad = out_grad[out_iter++];
            for (auto ky = decltype(patch.h){0}; ky < patch.h; ky++) {
              if ((0 <= y + ky) && (y + ky < H) && (0 <= y + yy + ky) &&
                  (y + yy + ky < H)) {
                for (auto kx = decltype(patch.w){0}; kx < patch.w; kx++) {
                  if ((0 <= x + kx) && (x + kx < W) && (0 <= x + xx + kx) &&
                      (x + xx + kx < W)) {
                    auto i1 = flat(n, y + ky, x + kx);
                    auto i2 = flat(n, y + yy + ky, x + xx + kx);
                    for (auto c = decltype(C){0}; c < C; c++) {
                      if (propagate_down[0])
                        in1_grad[i1 + c] += in2_data[i2 + c] * grad;
                      if (propagate_down[1])
                        in2_grad[i2 + c] += in1_data[i1 + c] * grad;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}
}
