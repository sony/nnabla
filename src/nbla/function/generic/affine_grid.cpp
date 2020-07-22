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
#include <nbla/function/affine_grid.hpp>
#include <nbla/utils/nd_index.hpp>
#include <nbla/variable.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(AffineGrid, const vector<int> &, bool);

template <typename T, bool align_corners>
void generate_target_grid_2d(T *grid, Shape_t shape, Shape_t stride) {
  auto B = shape[0];
  auto H = shape[1];
  auto W = shape[2];
  auto idx = 0;
  for (int b = 0; b < B; ++b) {
    for (int h = 0; h < H; ++h) {
      for (int w = 0; w < W; ++w) {
        // [-1, 1] <--> [0, S - 1] if align_corner
        // [-1, 1] <--> [-0.5, S - 0.5] = [0 - 0.5, S - 1 + 0.5] if not
        // align_corner
        auto y = T(2.0) * h / (H - 1) - T(1.0);
        auto x = T(2.0) * w / (W - 1) - T(1.0);
        y = align_corners ? y : y * (T(H - 1) / T(H));
        x = align_corners ? x : x * (T(W - 1) / T(W));
        grid[idx++] = x;
        grid[idx++] = y;
        grid[idx++] = T(1.0);
      }
    }
  }
}

template <typename T, bool align_corners>
void generate_target_grid_3d(T *grid, Shape_t shape, Shape_t stride) {
  auto B = shape[0];
  auto D = shape[1];
  auto H = shape[2];
  auto W = shape[3];
  for (int b = 0; b < B; ++b) {
    for (int d = 0; d < D; ++d) {
      for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
          auto idx = ndi::nd2flat(Shape_t{b, d, h, w, 0}, stride);
          auto z = T(2.0) * d / (D - 1) - T(1.0);
          auto y = T(2.0) * h / (H - 1) - T(1.0);
          auto x = T(2.0) * w / (W - 1) - T(1.0);
          z = align_corners ? z : z * (T(D - 1) / T(D));
          y = align_corners ? y : y * (T(H - 1) / T(H));
          x = align_corners ? x : x * (T(W - 1) / T(W));
          grid[idx] = x;
          grid[idx + 1] = y;
          grid[idx + 2] = z;
          grid[idx + 3] = T(1.0);
        }
      }
    }
  }
}

template <typename T>
void AffineGrid<T>::setup_impl(const Variables &inputs,
                               const Variables &outputs) {
  auto theta = inputs[0];
  auto grid = outputs[0];
  auto B = theta->shape()[0];
  auto tshape = theta->shape();
  if (size_.size() == 2) { // 2D
    NBLA_CHECK(tshape[1] == 2 && tshape[2] == 3, error_code::not_implemented,
               "Shape of theta must be (B, 2, 3) for 2D.");
    auto H = size_[0];
    auto W = size_[1];
    grid->reshape(Shape_t{B, H, W, 2}, true);
  } else if (size_.size() == 3) { // 3D
    NBLA_CHECK(tshape[1] == 3 && tshape[2] == 4, error_code::not_implemented,
               "Shape of theta must be (B, 3, 4) for 3D.");
    auto D = size_[0];
    auto H = size_[1];
    auto W = size_[2];
    grid->reshape(Shape_t{B, D, H, W, 3}, true);
  } else {
    NBLA_ERROR(error_code::not_implemented, "2D or 3D is only supported.");
  }
  batch_matmul_ = create_BatchMatmul(ctx_, false, true);
}

template <typename T>
void AffineGrid<T>::forward_impl(const Variables &inputs,
                                 const Variables &outputs) {
  auto theta = inputs[0];
  auto grid_s = outputs[0];

  if (size_.size() == 2) {
    // Target grid (with 1 for the translation)
    auto B = theta->shape()[0];
    auto H = size_[0];
    auto W = size_[1];
    Variable grid_t(Shape_t{B, H, W, 3});
    auto shape = grid_t.shape();
    auto stride = grid_t.strides();
    auto grid_t_ptr = grid_t.cast_data_and_get_pointer<T>(this->ctx_, true);
    auto generate_target_grid = align_corners_
                                    ? generate_target_grid_2d<T, true>
                                    : generate_target_grid_2d<T, false>;
    generate_target_grid(grid_t_ptr, shape, stride);
    // Transform: (B, H, W, 3) @ (B, 2, 3) --> (B, H, W, 2)
    grid_t.reshape(Shape_t{B, H * W, 3}, false);
    grid_s->reshape(Shape_t{B, H * W, 2}, false);
    execute(batch_matmul_, Variables{&grid_t, theta}, Variables{grid_s});
    grid_s->reshape(Shape_t{B, H, W, 2}, false);
  } else if (size_.size() == 3) {
    // Target grid (with 1 for the translation)
    auto B = theta->shape()[0];
    auto D = size_[0];
    auto H = size_[1];
    auto W = size_[2];
    Variable grid_t(Shape_t{B, D, H, W, 4});
    auto shape = grid_t.shape();
    auto stride = grid_t.strides();
    auto grid_t_ptr = grid_t.cast_data_and_get_pointer<T>(this->ctx_, true);
    auto generate_target_grid = align_corners_
                                    ? generate_target_grid_3d<T, true>
                                    : generate_target_grid_3d<T, false>;
    generate_target_grid(grid_t_ptr, shape, stride);
    // Transform: (B, D, H, W, 4) @ (B, 3, 4) --> (B, D, H, W, 3)
    grid_t.reshape(Shape_t{B, D * H * W, 4}, false);
    grid_s->reshape(Shape_t{B, D * H * W, 3}, false);
    execute(batch_matmul_, Variables{&grid_t, theta}, Variables{grid_s});
    grid_s->reshape(Shape_t{B, D, H, W, 3}, false);
  }
}

template <typename T>
void AffineGrid<T>::backward_impl(const Variables &inputs,
                                  const Variables &outputs,
                                  const vector<bool> &propagate_down,
                                  const vector<bool> &accum) {
  if (!(propagate_down[0])) {
    return;
  }
  // Gradient of outputs
  auto theta = inputs[0];
  auto B = theta->shape()[0];
  auto grid_s = outputs[0];
  if (size_.size() == 2) {
    // Target grid with 1 for the translation
    auto B = theta->shape()[0];
    auto H = size_[0];
    auto W = size_[1];
    Variable grid_t(Shape_t{B, H, W, 3});
    auto shape = grid_t.shape();
    auto stride = grid_t.strides();
    auto grid_t_ptr = grid_t.cast_data_and_get_pointer<T>(this->ctx_, true);
    auto generate_target_grid = align_corners_
                                    ? generate_target_grid_2d<T, true>
                                    : generate_target_grid_2d<T, false>;
    generate_target_grid(grid_t_ptr, shape, stride);

    // Backward of the transformation: (B, H, W, 2) @ (B, 2, 3) --> (B, H, W, 2)
    grid_t.reshape(Shape_t{B, H * W, 3}, false);
    grid_s->reshape(Shape_t{B, H * W, 2}, false);
    nbla::backward(batch_matmul_, Variables{&grid_t, theta}, Variables{grid_s},
                   vector<bool>{false, propagate_down[0]},
                   vector<bool>{false, accum[0]});
    grid_s->reshape(Shape_t{B, H, W, 2}, false);
  } else if (size_.size() == 3) {
    // Target grid with 1 for the translation
    auto D = size_[0];
    auto H = size_[1];
    auto W = size_[2];
    Variable grid_t(Shape_t{B, D, H, W, 4});
    auto shape = grid_t.shape();
    auto stride = grid_t.strides();
    auto grid_t_ptr = grid_t.cast_data_and_get_pointer<T>(this->ctx_, true);
    auto generate_target_grid = align_corners_
                                    ? generate_target_grid_3d<T, true>
                                    : generate_target_grid_3d<T, false>;
    generate_target_grid(grid_t_ptr, shape, stride);

    // Backward of the transformation: (B, D, H, W, 4) @ (B, 3, 4) --> (B, D, H,
    // W, 3)
    grid_t.reshape(Shape_t{B, D * H * W, 4}, false);
    grid_s->reshape(Shape_t{B, D * H * W, 3}, false);
    nbla::backward(batch_matmul_, Variables{&grid_t, theta}, Variables{grid_s},
                   vector<bool>{false, propagate_down[0]},
                   vector<bool>{false, accum[0]});
    grid_s->reshape(Shape_t{B, D, H, W, 3}, false);
  }
}
}
