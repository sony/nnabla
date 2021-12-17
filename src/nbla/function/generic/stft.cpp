// Copyright 2021 Sony Corporation.
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
#include <nbla/function/utils/stft_istft.hpp>
#include <nbla/variable.hpp>

#include <nbla/function/convolution.hpp>
#include <nbla/function/mul2.hpp>
#include <nbla/function/pad.hpp>

#include <nbla/function/istft.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(STFT, int, int, int, const string &, bool,
                              const string &, bool);

template <typename T>
void STFT<T>::setup_impl(const Variables &inputs, const Variables &outputs) {
  const bool is_valid_window_type = window_type_ == "hanning" ||
                                    window_type_ == "hamming" ||
                                    window_type_ == "rectangular";
  NBLA_CHECK(is_valid_window_type, error_code::value, "Unknown window type %s.",
             window_type_.c_str());
  NBLA_CHECK(fft_size_ >= window_size_, error_code::value,
             "FFT size has to be at least as large as window size.");

  // Setup functions
  const auto xs = inputs[0]->shape();
  // pad
  const vector<int> pad_width = {fft_size_ / 2, fft_size_ / 2};
  pad_ = create_Pad(ctx_, pad_width, pad_mode_, 0. /* constant_value */);
  pad_->setup({inputs[0]}, {&pad_out_});
  // mul2
  mul2_ = create_Mul2(ctx_, true);
  const auto mat_shape = Shape_t{fft_size_ / 2 + 1, 1, fft_size_};
  dft_w_r_.reshape(mat_shape, true);
  dft_w_i_.reshape(mat_shape, true);
  window_.reshape({1, 1, fft_size_}, true);
  mul2_->setup({&dft_w_r_, &window_}, {&conv_r_});
  mul2_->setup({&dft_w_i_, &window_}, {&conv_i_});
  // conv
  const int base_axis = 1;
  const vector<int> pad = {0};
  const vector<int> stride = {stride_};
  const vector<int> dilation = {1};
  const int group = 1;
  const bool channel_last = false;
  conv_ = create_Convolution(ctx_, base_axis, pad, stride, dilation, group,
                             channel_last);
  Variable conv_out; // dummy
  if (center_) {
    const auto pad_out_s = pad_out_.shape();
    pad_out_.reshape({pad_out_s[0], 1, pad_out_s[1]}, false);
    conv_->setup({&pad_out_, &conv_r_}, {&conv_out});
    // conv_->setup({&pad_out, &conv_i_}, {&conv_out});
    pad_out_.reshape(pad_out_s, false);
  } else {
    Variable conv_in({xs[0], 1, xs[1]});
    conv_->setup({&conv_in, &conv_r_}, {&conv_out});
    // conv_->setup({&conv_in, &conv_i_}, {&conv_out});
  }
  // Reshape output shape
  outputs[0]->reshape(conv_out.shape(), true);
  outputs[1]->reshape(conv_out.shape(), true);

  if (as_istft_backward_) {
    NBLA_CHECK(this->pad_mode_ == "constant", error_code::value,
               "`pad_mode` must be \"constant\" when `as_istft_backward == "
               "True`. Normal ISTFT never use `pad_mode` and just slice the "
               "output. Thus, STFT as a backward of normal ISTFT, STFT must be "
               "`pad_mode == \"constant\"`");

    // For the use of some ISTFT methods.
    istft_cpu_ = make_shared<ISTFT<T>>(ctx_, window_size_, stride_, fft_size_,
                                       window_type_, center_, pad_mode_,
                                       false /* as_stft_backward */);
    Variable dummy_y_r(outputs[0]->shape()), dummy_y_i(outputs[1]->shape()),
        dummy_x;
    istft_cpu_->setup({&dummy_y_r, &dummy_y_i}, {&dummy_x});

    // Setup internal buffers
    x_inv_window_.reshape({xs[0], 1, xs[1]}, true);
    conv_grad_.reshape({xs[0], 1, xs[1]}, true);
  }
}

template <typename T>
void STFT<T>::calculate_conv_weight(Variable &conv_r, Variable &conv_i) {
  if (as_istft_backward_) {
    // ISTFT backward needs IDFT coefficients not DFT's.
    // DFT and IDFT in STFT and ISTFT are not symmetrical in terms of
    // coefficients.
    istft_cpu_->calculate_conv_weight(conv_r, conv_i);
    return;
  }

  // Calculate DFT (descrete Fourier transform) coefficients.
  auto dft_w_r_data = dft_w_r_.cast_data_and_get_pointer<T>(ctx_);
  auto dft_w_i_data = dft_w_i_.cast_data_and_get_pointer<T>(ctx_);

  const double pi = std::acos(-1);
  for (int w = 0; w < fft_size_ / 2 + 1; w++) {
    for (int t = 0; t < fft_size_; t++) {
      dft_w_r_data[w * fft_size_ + t] = std::cos(2.0 * pi * w * t / fft_size_);
      dft_w_i_data[w * fft_size_ + t] = -std::sin(2.0 * pi * w * t / fft_size_);
    }
  }

  // Create window
  create_window<T>(&window_, window_type_, window_size_, fft_size_, ctx_);

  // Merge DFT coefficients and window func.
  // DFT and window application are performed by convolution.
  mul2_->forward({&dft_w_r_, &window_}, {&conv_r});
  mul2_->forward({&dft_w_i_, &window_}, {&conv_i});

  // Clear internal buffer
  window_.data()->array()->clear();
  dft_w_r_.data()->array()->clear();
  dft_w_i_.data()->array()->clear();
}

template <typename T>
void STFT<T>::apply_inv_window_forward(Variable *x, Variable *y) {
  istft_cpu_->apply_inv_window_forward(x, y);
}

template <typename T>
void STFT<T>::apply_inv_window_backward(Variable *x, Variable *y,
                                        const bool accum) {
  istft_cpu_->apply_inv_window_backward(x, y, accum);
}

template <typename T>
void STFT<T>::forward_impl(const Variables &inputs, const Variables &outputs) {
  auto x = inputs[0];
  auto y_r = outputs[0];
  auto y_i = outputs[1];

  if (center_) {
    pad_->forward({x}, {&pad_out_});

    // Add channel axis temporally to be used as input of convolution.
    const auto pad_out_s = pad_out_.shape();
    pad_out_.reshape(Shape_t{pad_out_s[0], 1, pad_out_s[1]}, false);

    if (as_istft_backward_) {
      apply_inv_window_forward(&pad_out_, &pad_out_);
    }

    // Compute STFT
    calculate_conv_weight(conv_r_, conv_i_);
    conv_->forward({&pad_out_, &conv_r_}, {y_r});
    conv_->forward({&pad_out_, &conv_i_}, {y_i});

    // Restore the shape and clear buffer
    pad_out_.reshape(pad_out_s, false);
    pad_out_.data()->array()->clear();
  } else {
    // Add channel axis temporally to be used as input of convolution.
    const auto x_shape_org = x->shape();
    x->reshape(Shape_t{x_shape_org[0], 1, x_shape_org[1]}, false);

    if (as_istft_backward_) {
      // Compute ISTFT backward
      apply_inv_window_forward(x, &x_inv_window_);

      calculate_conv_weight(conv_r_, conv_i_);
      conv_->forward({&x_inv_window_, &conv_r_}, {y_r});
      conv_->forward({&x_inv_window_, &conv_i_}, {y_i});

      // Clear buffer
      x_inv_window_.data()->array()->clear();
    } else {
      // Compute STFT
      calculate_conv_weight(conv_r_, conv_i_);
      conv_->forward({x, &conv_r_}, {y_r});
      conv_->forward({x, &conv_i_}, {y_i});
    }

    // Restore x shape
    x->reshape(x_shape_org, false);
  }

  // Clear internal buffers
  conv_r_.data()->array()->clear();
  conv_i_.data()->array()->clear();
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

  // Execute in reverse order of forward_impl.

  if (center_) {
    // Add channel axis
    const auto pad_out_s = pad_out_.shape();
    pad_out_.reshape({pad_out_s[0], 1, pad_out_s[1]}, false);

    // Compute STFT backward
    calculate_conv_weight(conv_r_, conv_i_);
    conv_->backward({&pad_out_, &conv_r_}, {y_r}, {true, false},
                    {false, false});
    conv_->backward({&pad_out_, &conv_i_}, {y_i}, {true, false}, {true, false});

    if (as_istft_backward_) {
      apply_inv_window_backward(&pad_out_, &pad_out_, false);
    }

    // Remove channel axis
    pad_out_.reshape(pad_out_s, false);

    pad_->backward({x}, {&pad_out_}, {true}, {accum[0]});

    pad_out_.grad()->array()->clear();
  } else {
    // Add channel dimension temporally
    const auto x_shape_org = x->shape();
    x->reshape(Shape_t{x_shape_org[0], 1, x_shape_org[1]}, false);

    if (as_istft_backward_) {
      // Compute ISTFT double backward
      calculate_conv_weight(conv_r_, conv_i_);
      conv_->backward({&conv_grad_, &conv_r_}, {y_r}, {true, false},
                      {false, false});
      conv_->backward({&conv_grad_, &conv_i_}, {y_i}, {true, false},
                      {true, false});

      apply_inv_window_backward(x, &conv_grad_, accum[0]);

      conv_grad_.grad()->array()->clear();
    } else {
      // Compute STFT backward
      calculate_conv_weight(conv_r_, conv_i_);
      conv_->backward({x, &conv_r_}, {y_r}, {true, false}, {accum[0], false});
      conv_->backward({x, &conv_i_}, {y_i}, {true, false}, {true, false});
    }

    // Restore x shape
    x->reshape(x_shape_org, false);
  }

  // Clear internal buffers
  conv_r_.data()->array()->clear();
  conv_i_.data()->array()->clear();
}
}
