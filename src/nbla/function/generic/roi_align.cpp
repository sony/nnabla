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
#include <nbla/function/roi_align.hpp>
#include <nbla/variable.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(RoiAlign, const vector<int> &,
                              const vector<float> &, int, bool, bool);

template <typename T>
void RoiAlign<T>::setup_impl(const Variables &inputs,
                             const Variables &outputs) {
#define CHECK(condition, msg, ...)                                             \
  NBLA_CHECK(condition, error_code::value, msg, ##__VA_ARGS__);

  auto const input = inputs.at(0);
  auto const boxes = inputs.at(1);

  CHECK(input->shape().size() == 4, "input variable must be 4-D.");
  CHECK(boxes->shape().size() == 2, "boxes variable must be 2-D.");
  CHECK(boxes->shape().at(1) == 5, "boxes shape must be [K, 5]");
  CHECK(output_size_.size() == 2, "output_size must be (height, width) tuple");
  CHECK(output_size_.at(0) > 0, "output height must be greater zero");
  CHECK(output_size_.at(1) > 0, "output width must be greater zero");
  CHECK(spatial_scale_.size() == 2, "spatial_scale must be an (y, x) tuple");

  if (not this->channel_last_) {
    auto const K = boxes->shape().at(0);
    auto const C = input->shape().at(1);
    auto const H = output_size_.at(0);
    auto const W = output_size_.at(1);
    outputs.at(0)->reshape({K, C, H, W}, true);
  } else {
    auto const K = boxes->shape().at(0);
    auto const C = input->shape().at(3);
    auto const H = output_size_.at(0);
    auto const W = output_size_.at(1);
    outputs.at(0)->reshape({K, H, W, C}, true);
  }

#undef CHECK
}

template <typename T> struct Box {
  T batch_index, x1, y1, x2, y2;

  inline void scale_and_shift(const vector<float> &scale, const float shift) {
    x1 = x1 * scale.at(1) - shift;
    y1 = y1 * scale.at(0) - shift;
    x2 = x2 * scale.at(1) - shift;
    y2 = y2 * scale.at(0) - shift;
  }

  inline T width(const bool aligned) {
    return aligned ? x2 - x1 : std::max(x2 - x1, T(1));
  };
  inline T height(const bool aligned) {
    return aligned ? y2 - y1 : std::max(y2 - y1, T(1));
  };
  inline int index() { return std::max(static_cast<int>(batch_index), 0); }
};

template <typename T>
inline int sampling_grid(const int sampling_ratio, const T step_size) {
  return sampling_ratio > 0 ? sampling_ratio
                            : static_cast<int>(std::ceil(step_size));
}

template <typename T>
void RoiAlign<T>::forward_impl(const Variables &inputs,
                               const Variables &outputs) {
  NBLA_CHECK(this->channel_last_ == false, error_code::not_implemented,
             "Argument channel_last=true is not supported in CPU context.");

  auto input = inputs.at(0);
  auto boxes = inputs.at(1);
  auto output = outputs.at(0);

  auto input_data = input->get_data_pointer<T>(this->ctx_);
  auto boxes_data = boxes->get_data_pointer<T>(this->ctx_);
  auto output_data = output->cast_data_and_get_pointer<T>(this->ctx_, true);

  auto roiboxes = boxes->shape().at(0);
  auto channels = input->shape().at(1);

  auto image_rows = input->shape().at(2);
  auto image_cols = input->shape().at(3);
  auto image_size = image_rows * image_cols;

  auto output_rows = output->shape().at(2);
  auto output_cols = output->shape().at(3);

  for (auto n = 0; n < roiboxes; n++) {
    auto roi = *reinterpret_cast<Box<T> const *>(boxes_data + n * 5);
    roi.scale_and_shift(this->spatial_scale_, this->aligned_ ? T(0.5) : T(0));

    auto const step_size_x = roi.width(this->aligned_) / output_cols;
    auto const step_size_y = roi.height(this->aligned_) / output_rows;

    auto const grid_size_x = sampling_grid(this->sampling_ratio_, step_size_x);
    auto const grid_size_y = sampling_grid(this->sampling_ratio_, step_size_y);

    auto const step_size_xx = step_size_x / grid_size_x;
    auto const step_size_yy = step_size_y / grid_size_y;
    auto const half_step_xx = T(0.5) * step_size_xx;
    auto const half_step_yy = T(0.5) * step_size_yy;

    auto const grid_size_xy = grid_size_x * grid_size_y;
    auto const inverse_grid_count = T(1) / grid_size_xy;

    auto channel_data = input_data + roi.index() * channels * image_size;

    for (auto c = 0; c < channels; c++) {
      auto output_index = (n * channels + c) * output_rows * output_cols;

      for (auto y = 0; y < output_rows; y++) {
        auto yf = roi.y1 + static_cast<T>(y) * step_size_y;

        for (auto x = 0; x < output_cols; x++) {
          auto xf = roi.x1 + static_cast<T>(x) * step_size_x;
          auto output_value = T(0);

          for (auto yy = 0; yy < grid_size_y; yy++) {
            auto yyf = yf + static_cast<T>(yy) * step_size_yy + half_step_yy;

            if (yyf < T(-1) || yyf > static_cast<T>(image_rows))
              continue;

            if (yyf < T(0))
              yyf = T(0);

            for (auto xx = 0; xx < grid_size_x; xx++) {
              auto xxf = xf + static_cast<T>(xx) * step_size_xx + half_step_xx;

              if (xxf < T(-1) || xxf > static_cast<T>(image_cols))
                continue;

              if (xxf < T(0))
                xxf = T(0);

              auto y_low = static_cast<int>(yyf);
              auto x_low = static_cast<int>(xxf);
              auto y_high = y_low + 1;
              auto x_high = x_low + 1;

              if (y_low >= image_rows - 1) {
                y_high = y_low = image_rows - 1;
                yyf = static_cast<T>(y_low);
              }

              if (x_low >= image_cols - 1) {
                x_high = x_low = image_cols - 1;
                xxf = static_cast<T>(x_low);
              }

              auto const ly = yyf - std::floor(yyf);
              auto const lx = xxf - std::floor(xxf);
              auto const hy = T(1) - ly;
              auto const hx = T(1) - lx;
              auto const p1 = y_low * image_cols + x_low;
              auto const p2 = y_low * image_cols + x_high;
              auto const p3 = y_high * image_cols + x_low;
              auto const p4 = y_high * image_cols + x_high;
              output_value += hy * hx * channel_data[p1];
              output_value += hy * lx * channel_data[p2];
              output_value += ly * hx * channel_data[p3];
              output_value += ly * lx * channel_data[p4];
            }
          }
          output_data[output_index++] = output_value * inverse_grid_count;
        }
      }
      channel_data += image_size;
    }
  }
}

template <typename T>
void RoiAlign<T>::backward_impl(const Variables &inputs,
                                const Variables &outputs,
                                const vector<bool> &propagate_down,
                                const vector<bool> &accum) {
  if (!propagate_down.at(0))
    return;

  NBLA_CHECK(this->channel_last_ == false, error_code::not_implemented,
             "Argument channel_last=true is not supported in CPU context.");

  auto input = inputs.at(0);
  auto boxes = inputs.at(1);
  auto output = outputs.at(0);

  if (!accum.at(0)) {
    input->grad()->zero();
  }

  auto input_grad = input->cast_grad_and_get_pointer<T>(this->ctx_, false);
  auto boxes_data = boxes->get_data_pointer<T>(this->ctx_);
  auto output_grad = output->get_grad_pointer<T>(this->ctx_);

  auto roiboxes = boxes->shape().at(0);
  auto channels = input->shape().at(1);

  auto image_rows = input->shape().at(2);
  auto image_cols = input->shape().at(3);
  auto image_size = image_rows * image_cols;

  auto output_rows = output->shape().at(2);
  auto output_cols = output->shape().at(3);

  for (auto n = 0; n < roiboxes; n++) {
    auto roi = *reinterpret_cast<Box<T> const *>(boxes_data + n * 5);
    roi.scale_and_shift(this->spatial_scale_, this->aligned_ ? T(0.5) : T(0));

    auto const step_size_x = roi.width(this->aligned_) / output_cols;
    auto const step_size_y = roi.height(this->aligned_) / output_rows;

    auto const grid_size_x = sampling_grid(this->sampling_ratio_, step_size_x);
    auto const grid_size_y = sampling_grid(this->sampling_ratio_, step_size_y);

    auto const step_size_xx = step_size_x / grid_size_x;
    auto const step_size_yy = step_size_y / grid_size_y;
    auto const half_step_xx = T(0.5) * step_size_xx;
    auto const half_step_yy = T(0.5) * step_size_yy;

    auto const grid_size_xy = grid_size_x * grid_size_y;
    auto const inverse_grid_count = T(1) / grid_size_xy;

    auto channel_grad = input_grad + roi.index() * channels * image_size;

    for (auto c = 0; c < channels; c++) {
      auto output_index = (n * channels + c) * output_rows * output_cols;

      for (auto y = 0; y < output_rows; y++) {
        auto yf = roi.y1 + static_cast<T>(y) * step_size_y;

        for (auto x = 0; x < output_cols; x++) {
          auto xf = roi.x1 + static_cast<T>(x) * step_size_x;
          auto grad_value = output_grad[output_index++] * inverse_grid_count;

          for (auto yy = 0; yy < grid_size_y; yy++) {
            auto yyf = yf + static_cast<T>(yy) * step_size_yy + half_step_yy;

            if (yyf < T(-1) || yyf > static_cast<T>(image_rows))
              continue;

            if (yyf < T(0))
              yyf = T(0);

            for (auto xx = 0; xx < grid_size_x; xx++) {
              auto xxf = xf + static_cast<T>(xx) * step_size_xx + half_step_xx;

              if (xxf < T(-1) || xxf > static_cast<T>(image_cols))
                continue;

              if (xxf < T(0))
                xxf = T(0);

              auto y_low = static_cast<int>(yyf);
              auto x_low = static_cast<int>(xxf);
              auto y_high = y_low + 1;
              auto x_high = x_low + 1;

              if (y_low >= image_rows - 1) {
                y_high = y_low = image_rows - 1;
                yyf = static_cast<T>(y_low);
              }

              if (x_low >= image_cols - 1) {
                x_high = x_low = image_cols - 1;
                xxf = static_cast<T>(x_low);
              }

              T ly = yyf - std::floor(yyf);
              T lx = xxf - std::floor(xxf);
              T hy = T(1) - ly;
              T hx = T(1) - lx;
              auto const p1 = y_low * image_cols + x_low;
              auto const p2 = y_low * image_cols + x_high;
              auto const p3 = y_high * image_cols + x_low;
              auto const p4 = y_high * image_cols + x_high;
              channel_grad[p1] += hy * hx * grad_value;
              channel_grad[p2] += hy * lx * grad_value;
              channel_grad[p3] += ly * hx * grad_value;
              channel_grad[p4] += ly * lx * grad_value;
            }
          }
        }
      }
      channel_grad += image_size;
    }
  }
}
}
