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
#include <nbla/function/istft.hpp>
#include <nbla/variable.hpp>

#include <nbla/function/deconvolution.hpp>
#include <nbla/function/div2.hpp>
#include <nbla/function/mul2.hpp>
#include <nbla/function/slice.hpp>
#include <nbla/function/sub2.hpp>

#include <nbla/imperative.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(ISTFT, int, int, int, const string &, bool);

template <typename T>
void ISTFT<T>::setup_impl(const Variables &inputs, const Variables &outputs) {
  const bool is_valid_window_type = window_type_ == "hanning" ||
                                    window_type_ == "hamming" ||
                                    window_type_ == "rectangular";
  NBLA_CHECK(is_valid_window_type, error_code::value, "Unknown window type %s.",
             window_type_.c_str());
  NBLA_CHECK(fft_size_ >= window_size_, error_code::value,
             "FFT size has to be at least as large as window size.");
  NBLA_CHECK(fft_size_ % stride_ == 0, error_code::value,
             "FFT size needs to be a multiple of stride.");

  // setup functions
  // mul2
  mul2_ = create_Mul2(ctx_, true);
  Variable mat({fft_size_ / 2 + 1, 1, fft_size_});
  Variable window_func({1, 1, fft_size_});
  Variable mul2_out;
  mul2_->setup({&mat, &window_func}, {&mul2_out});
  // div2
  div2_ = create_Div2(ctx_, true);
  Variable inv_window_func({1, 1, fft_size_});
  Variable deconv_weight;
  div2_->setup({&mul2_out, &inv_window_func}, {&deconv_weight});
  deconv_w_shape_ = deconv_weight.shape();
  // deconv
  const int base_axis = 1;
  const vector<int> pad = {0};
  const vector<int> stride = {stride_};
  const vector<int> dilation = {1};
  const int group = 1;
  const bool channel_last = false;
  const vector<int> output_padding = {0};
  deconv_ = create_Deconvolution(ctx_, base_axis, pad, stride, dilation, group,
                                 channel_last, output_padding);
  Variable deconv_out;
  deconv_->setup({inputs[0], &deconv_weight}, {&deconv_out});
  deconv_y_shape_ = deconv_out.shape();
  // sub2
  sub2_ = create_Sub2(ctx_, false);
  Variable sub2_out;
  sub2_->setup({&deconv_out, &deconv_out}, {&sub2_out});
  sub2_out.reshape({deconv_y_shape_[0], deconv_y_shape_[2]}, false);
  if (center_) {
    // slice
    int batch_size = inputs[0]->shape()[0];
    slice_ = create_Slice(ctx_, {0, fft_size_ / 2},
                          {batch_size, -fft_size_ / 2}, {1, 1});
    Variable slice_out;
    slice_->setup({&sub2_out}, {&slice_out});
    outputs[0]->reshape(slice_out.shape(), true);
  } else {
    outputs[0]->reshape(sub2_out.shape(), true);
  }
}

template <typename T>
void ISTFT<T>::calculate_conv_weight(Variable &conv_cos, Variable &conv_sin) {
  // create window_func
  Variable window_func(Shape_t{1, 1, fft_size_});
  window_func.data()->zero();

  auto window_func_data = window_func.cast_data_and_get_pointer<T>(ctx_);
  const double pi = std::acos(-1);

  const int left_pad = (fft_size_ - window_size_) / 2;
  if (window_type_ == "hanning") {
    for (int i = 0; i < window_size_; i++) {
      window_func_data[left_pad + i] =
          0.5 - 0.5 * std::cos(2.0 * pi * i / (window_size_));
    }
  } else if (window_type_ == "hamming") {
    for (int i = 0; i < window_size_; i++) {
      window_func_data[left_pad + i] =
          0.54 - 0.46 * std::cos(2.0 * pi * i / (window_size_));
    }
  } else { // window_type == "rectangular"
    // fill 1
    for (int i = 0; i < window_size_; i++) {
      window_func_data[left_pad + i] = 1.;
    }
  }

  // compute inverse STFT filter coefficients
  Variable inv_window_func(window_func.shape());
  inv_window_func.data()->zero();
  auto inv_window_func_data =
      inv_window_func.cast_data_and_get_pointer<T>(ctx_);
  for (int i = 0; i < fft_size_; i += stride_) {
    for (int j = 0; j < fft_size_; j++) {
      // roll window_func to the right by i
      const auto w_data = window_func_data[(j - i + fft_size_) % fft_size_];
      inv_window_func_data[j] += w_data * w_data;
    }
  }

  const auto mat_shape = Shape_t{fft_size_ / 2 + 1, 1, fft_size_};
  Variable mat_cos(mat_shape), mat_sin(mat_shape);

  auto mat_cos_data = mat_cos.cast_data_and_get_pointer<T>(ctx_);
  auto mat_sin_data = mat_sin.cast_data_and_get_pointer<T>(ctx_);

  for (int w = 0; w < fft_size_ / 2 + 1; w++) {
    const auto alpha = (w == 0 || w == fft_size_ / 2 ? 1.0 : 2.0) / fft_size_;
    for (int t = 0; t < fft_size_; t++) {
      mat_cos_data[w * fft_size_ + t] =
          alpha * std::cos(2.0 * pi * w * t / fft_size_);
      mat_sin_data[w * fft_size_ + t] =
          alpha * std::sin(2.0 * pi * w * t / fft_size_);
    }
  }

  // conv_cos
  mul2_->forward({&mat_cos, &window_func}, {&mat_cos});
  div2_->forward({&mat_cos, &inv_window_func}, {&conv_cos});
  // conv_sin
  mul2_->forward({&mat_sin, &window_func}, {&mat_sin});
  div2_->forward({&mat_sin, &inv_window_func}, {&conv_sin});
}

template <typename T>
void ISTFT<T>::forward_impl(const Variables &inputs, const Variables &outputs) {
  auto y_r = inputs[0];
  auto y_i = inputs[1];
  auto x = outputs[0];

  // calculate weight matrix
  Variable conv_cos(deconv_w_shape_);
  Variable conv_sin(deconv_w_shape_);
  calculate_conv_weight(conv_cos, conv_sin);

  // compute inverse STFT
  Variable x_cos(deconv_y_shape_);
  Variable x_sin(deconv_y_shape_);
  deconv_->forward({y_r, &conv_cos}, {&x_cos});
  deconv_->forward({y_i, &conv_sin}, {&x_sin});

  if (center_) {
    Variable sub2_out(deconv_y_shape_);
    sub2_->forward({&x_cos, &x_sin}, {&sub2_out});

    // remove channel axis
    sub2_out.reshape({deconv_y_shape_[0], deconv_y_shape_[2]}, false);

    slice_->forward({&sub2_out}, {x});
  } else {
    // add channel axis
    x->reshape({deconv_y_shape_[0], 1, deconv_y_shape_[2]}, false);

    sub2_->forward({&x_cos, &x_sin}, {x});

    // remove channel axis
    x->reshape({deconv_y_shape_[0], deconv_y_shape_[2]}, false);
  }
}

template <typename T>
void ISTFT<T>::backward_impl(const Variables &inputs, const Variables &outputs,
                             const vector<bool> &propagate_down,
                             const vector<bool> &accum) {
  if (!(propagate_down[0] || propagate_down[1])) {
    return;
  }

  auto y_r = inputs[0];
  auto y_i = inputs[1];
  auto x = outputs[0];

  // calculate weight matrix
  Variable conv_cos(deconv_w_shape_);
  Variable conv_sin(deconv_w_shape_);
  calculate_conv_weight(conv_cos, conv_sin);

  Variable x_cos(deconv_y_shape_);
  Variable x_sin(deconv_y_shape_);
  Variable sub2_out(deconv_y_shape_);

  if (center_) {
    sub2_out.reshape({deconv_y_shape_[0], deconv_y_shape_[2]}, false);
    slice_->backward({&sub2_out}, {x}, {true}, {false});

    sub2_out.reshape(deconv_y_shape_, false);
    sub2_->backward({&x_cos, &x_sin}, {&sub2_out}, {true, true},
                    {false, false});
  } else {
    const auto xs_org = x->shape();
    x->reshape(deconv_y_shape_, false);
    sub2_->backward({&x_cos, &x_sin}, {x}, {true, true}, {false, false});

    // restore output shape
    x->reshape(xs_org, false);
  }

  deconv_->backward({y_r, &conv_cos}, {&x_cos}, {propagate_down[0], false},
                    {accum[0], false});
  deconv_->backward({y_i, &conv_sin}, {&x_sin}, {propagate_down[1], false},
                    {accum[1], false});
}
}
