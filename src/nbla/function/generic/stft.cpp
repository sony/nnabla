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
#include <nbla/function/stft.hpp>
#include <nbla/variable.hpp>

#include <nbla/function/convolution.hpp>
#include <nbla/function/mul2.hpp>
#include <nbla/function/pad.hpp>

#include <nbla/imperative.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(STFT, int, int, int, const string &, bool,
                              const string &);

template <typename T>
void STFT<T>::setup_impl(const Variables &inputs, const Variables &outputs) {
  const bool is_valid_window_type = window_type_ == "hanning" ||
                                    window_type_ == "hamming" ||
                                    window_type_ == "rectangular";
  NBLA_CHECK(is_valid_window_type, error_code::value, "Unknown window type %s.",
             window_type_.c_str());
  NBLA_CHECK(fft_size_ >= window_size_, error_code::value,
             "FFT size has to be at least as large as window size.");

  // setup functions
  // pad
  const vector<int> pad_width = {fft_size_ / 2, fft_size_ / 2};
  pad_ = create_Pad(ctx_, pad_width, pad_mode_, 0. /* constant_value */);
  Variable pad_out;
  pad_->setup({inputs[0]}, {&pad_out});
  pad_out_shape_ = pad_out.shape();
  // mul2
  mul2_ = create_Mul2(ctx_, true);
  Variable mat({fft_size_ / 2 + 1, 1, fft_size_});
  Variable window_func({1, 1, fft_size_});
  Variable conv_weight;
  mul2_->setup({&mat, &window_func}, {&conv_weight});
  conv_weight_shape_ = conv_weight.shape();
  // conv
  const int base_axis = 1;
  const vector<int> pad = {0};
  const vector<int> stride = {stride_};
  const vector<int> dilation = {1};
  const int group = 1;
  const bool channel_last = false;
  conv_ = create_Convolution(ctx_, base_axis, pad, stride, dilation, group,
                             channel_last);
  Variable conv_out;
  if (center_) {
    pad_out.reshape({pad_out_shape_[0], 1, pad_out_shape_[1]}, false);
    conv_->setup({&pad_out, &conv_weight}, {&conv_out});
  } else {
    const auto xs = inputs[0]->shape();
    Variable conv_in({xs[0], 1, xs[1]});
    conv_->setup({&conv_in, &conv_weight}, {&conv_out});
  }
  // reshape output shape
  outputs[0]->reshape(conv_out.shape(), true);
  outputs[1]->reshape(conv_out.shape(), true);
}

template <typename T>
void STFT<T>::calculate_conv_weight(Variable &conv_r, Variable &conv_i) {
  // create window_func
  Variable window_func({1, 1, fft_size_});
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

  // calculate STFT filter coefficients
  const auto mat_shape = Shape_t{fft_size_ / 2 + 1, 1, fft_size_};
  Variable mat_r(mat_shape), mat_i(mat_shape);

  auto mat_r_data = mat_r.cast_data_and_get_pointer<T>(ctx_);
  auto mat_i_data = mat_i.cast_data_and_get_pointer<T>(ctx_);

  for (int w = 0; w < fft_size_ / 2 + 1; w++) {
    for (int t = 0; t < fft_size_; t++) {
      mat_r_data[w * fft_size_ + t] = std::cos(2.0 * pi * w * t / fft_size_);
      mat_i_data[w * fft_size_ + t] = -std::sin(2.0 * pi * w * t / fft_size_);
    }
  }

  mul2_->forward({&mat_r, &window_func}, {&conv_r});
  mul2_->forward({&mat_i, &window_func}, {&conv_i});
}

template <typename T>
void STFT<T>::forward_impl(const Variables &inputs, const Variables &outputs) {
  auto x = inputs[0];
  auto y_r = outputs[0];
  auto y_i = outputs[1];

  // calculate weight matrix
  Variable conv_r(conv_weight_shape_), conv_i(conv_weight_shape_);
  calculate_conv_weight(conv_r, conv_i);

  if (center_) {
    // pad at begin/end (per default this is a reflection padding)
    Variable pad_out(pad_out_shape_);
    pad_->forward({x}, {&pad_out});

    // add channel dimension
    pad_out.reshape(Shape_t{pad_out_shape_[0], 1, pad_out_shape_[1]}, false);

    // compute STFT
    conv_->forward({&pad_out, &conv_r}, {y_r});
    conv_->forward({&pad_out, &conv_i}, {y_i});
  } else {
    const auto x_shape_org = x->shape();

    // add channel dimension
    x->reshape(Shape_t{x_shape_org[0], 1, x_shape_org[1]}, false);

    // compute STFT
    conv_->forward({x, &conv_r}, {y_r});
    conv_->forward({x, &conv_i}, {y_i});

    // restore x shape
    x->reshape(x_shape_org, false);
  }
}

template <typename T>
void STFT<T>::backward_impl(const Variables &inputs, const Variables &outputs,
                            const vector<bool> &propagate_down,
                            const vector<bool> &accum) {
  if (!(propagate_down[0])) {
    return;
  }

  auto x = inputs[0];
  auto y_r = outputs[0];
  auto y_i = outputs[1];

  // calculate weight matrix
  Variable conv_r(conv_weight_shape_), conv_i(conv_weight_shape_);
  calculate_conv_weight(conv_r, conv_i);

  if (center_) {
    Variable pad_out({pad_out_shape_[0], 1, pad_out_shape_[1]});

    // compute STFT backward
    conv_->backward({&pad_out, &conv_r}, {y_r}, {true, false}, {false, false});
    conv_->backward({&pad_out, &conv_i}, {y_i}, {true, false}, {true, false});

    // remove channel dimension
    pad_out.reshape(pad_out_shape_, false);

    pad_->backward({x}, {&pad_out}, {true}, {accum[0]});
  } else {
    const auto x_shape_org = x->shape();

    // add channel dimension temporally
    x->reshape(Shape_t{x_shape_org[0], 1, x_shape_org[1]}, false);

    // compute STFT backward
    conv_->backward({x, &conv_r}, {y_r}, {true, false}, {accum[0], false});
    conv_->backward({x, &conv_i}, {y_i}, {true, false}, {true, false});

    // restore x shape
    x->reshape(x_shape_org, false);
  }
}
}
