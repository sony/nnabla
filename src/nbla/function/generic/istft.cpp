// Copyright 2021 Sony Corporation.
// Copyright 2021 Sony Group Corporation.
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
#include <nbla/function/utils/stft_istft.hpp>
#include <nbla/variable.hpp>

#include <nbla/function/add2.hpp>
#include <nbla/function/deconvolution.hpp>
#include <nbla/function/div2.hpp>
#include <nbla/function/mul2.hpp>
#include <nbla/function/pad.hpp>
#include <nbla/function/slice.hpp>

#include <nbla/function/stft.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(ISTFT, int, int, int, const string &, bool,
                              const string &, bool);

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

  const int batch_size = inputs[0]->shape()[0];

  // Setup functions
  // mul2
  mul2_ = create_Mul2(ctx_, true);
  window_.reshape({1, 1, fft_size_}, true);
  const auto idft_w_shape = Shape_t{fft_size_ / 2 + 1, 1, fft_size_};
  idft_w_cos_.reshape(idft_w_shape, true);
  idft_w_sin_.reshape(idft_w_shape, true);
  mul2_->setup({&idft_w_cos_, &window_}, {&conv_cos_});
  mul2_->setup({&idft_w_sin_, &window_}, {&conv_sin_});
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
  deconv_->setup({inputs[0], &conv_cos_}, {&x_cos_});
  deconv_->setup({inputs[0], &conv_sin_}, {&x_sin_});
  deconv_out_.reshape(x_cos_.shape(), true);
  // add2
  add2_ = create_Add2(ctx_, false);
  add2_->setup({&x_cos_, &x_sin_}, {&add2_out_});
  const auto add2_out_s = add2_out_.shape();
  add2_out_.reshape({add2_out_s[0], add2_out_s[2]}, false);
  // inv_window
  inv_window_.reshape({add2_out_.size() / batch_size}, true);
  if (center_) {
    // slice
    slice_ = create_Slice(ctx_, {0, fft_size_ / 2},
                          {batch_size, -fft_size_ / 2}, {1, 1});
    Variable slice_out;
    slice_->setup({&add2_out_}, {&slice_out});
    outputs[0]->reshape(slice_out.shape(), true);
  } else {
    outputs[0]->reshape(add2_out_.shape(), true);
  }
  add2_out_.reshape(add2_out_s, false);

  if (!as_stft_backward_) {
    NBLA_CHECK(this->pad_mode_ == "constant", error_code::value,
               "`pad_mode` should be \"constant\" for the normal use of ISTFT "
               "(`as_stft_backward == false`) since `pad_mode` is ignored and "
               "makes no effects in that case.");
  }

  // Check NOLA(Nonzero Overlap Add) condition
  // This check is not needed when `as_stft_backward == true` because division
  // by `inv_window` is not performed.
  // TODO: This check is not needed every setup execution.
  if (!as_stft_backward_) {
    Context cpu_ctx{{"cpu:float"}, "CpuCachedArray", "0"};
    const auto size = add2_out_.size() / batch_size;
    Variable inv_window(Shape_t{size});
    ISTFT<T>::calculate_inv_window(cpu_ctx, &inv_window);
    const auto inv_window_data = inv_window.get_data_pointer<float>(cpu_ctx);
    const int start = center_ ? fft_size_ / 2 : 0;
    const int end = center_ ? size - fft_size_ / 2 : size;
    for (int i = start; i < end; i++) {
      NBLA_CHECK(inv_window_data[i] >= 1e-11, error_code::value,
                 "NOLA(Nonzero Overlap Add) condition is not met. "
                 "`inv_window[%d] == %f`",
                 i, inv_window_data[i]);
    }
  }

  if (as_stft_backward_) {
    // For the use of some STFT methods.
    stft_cpu_ = make_shared<STFT<T>>(ctx_, window_size_, stride_, fft_size_,
                                     window_type_, center_, pad_mode_,
                                     false /* as_istft_backward */);
    Variable dummy_x(outputs[0]->shape()), dummy_y_r(inputs[0]->shape()),
        dummy_y_i(inputs[1]->shape());
    stft_cpu_->setup({&dummy_x}, {&dummy_y_r, &dummy_y_i});

    // Pad function for correct gradient calculation
    const vector<int> pad_width = {fft_size_ / 2, fft_size_ / 2};
    pad_ = create_Pad(ctx_, pad_width, pad_mode_, 0. /* constant_value */);
    Variable pad_in(outputs[0]->shape());
    Variable pad_out;
    pad_->setup({&pad_in}, {&pad_out});
  }
}

template <typename T>
void ISTFT<T>::calculate_window(Context &ctx, Variable *window) const {
  create_window<T>(window, window_type_, window_size_, fft_size_, ctx);
}

template <typename T>
void ISTFT<T>::calculate_inv_window(Context &ctx, Variable *inv_window) {
  // Create window
  create_window<T>(&window_, window_type_, window_size_, fft_size_, ctx);

  // Calculate inv_window
  const int size = inv_window->size();
  inv_window->data()->zero();
  const auto window_data = window_.get_data_pointer<T>(ctx);
  auto inv_window_data = inv_window->cast_data_and_get_pointer<T>(ctx);
  for (int i = 0; i < size - fft_size_ + 1; i += stride_) {
    for (int j = 0; j < fft_size_; j++) {
      inv_window_data[i + j] += window_data[j] * window_data[j];
    }
  }

  // Clear internal buffer
  window_.data()->array()->clear();
}

template <typename T>
void ISTFT<T>::calculate_conv_weight(Variable &conv_cos, Variable &conv_sin) {
  if (as_stft_backward_) {
    // STFT backward needs DFT coefficients not IDFT's.
    // DFT and IDFT in STFT and ISTFT are not symmetrical in terms of
    // coefficients.
    stft_cpu_->calculate_conv_weight(conv_cos, conv_sin);
    return;
  }

  // Calculate IDFT (Inverse descrete Fourier transform) coefficients.
  auto idft_w_cos_data = idft_w_cos_.cast_data_and_get_pointer<T>(ctx_);
  auto idft_w_sin_data = idft_w_sin_.cast_data_and_get_pointer<T>(ctx_);

  const double pi = std::acos(-1);
  for (int w = 0; w < fft_size_ / 2 + 1; w++) {
    const auto alpha = (w == 0 || w == fft_size_ / 2 ? 1.0 : 2.0) / fft_size_;
    for (int t = 0; t < fft_size_; t++) {
      idft_w_cos_data[w * fft_size_ + t] =
          alpha * std::cos(2.0 * pi * w * t / fft_size_);
      idft_w_sin_data[w * fft_size_ + t] =
          alpha * -std::sin(2.0 * pi * w * t / fft_size_);
    }
  }

  // Create window
  calculate_window(ctx_, &window_);

  // Merge IDFT coefficients and window func.
  // IDFT, window application and the part of overlap-add are performed by
  // deconvolution.
  // The complete overlap-add is achieved after applying `inv_window` to the
  // deconvolution output.
  mul2_->forward({&idft_w_cos_, &window_}, {&conv_cos});
  mul2_->forward({&idft_w_sin_, &window_}, {&conv_sin});

  // Clear internal buffers
  idft_w_cos_.data()->array()->clear();
  idft_w_sin_.data()->array()->clear();
  window_.data()->array()->clear();
}

template <typename T>
void ISTFT<T>::apply_inv_window_forward(Variable *x, Variable *y) {
  // Create inv_window
  const auto batch_size = x->shape()[0];
  const auto size = x->size() / batch_size;
  calculate_inv_window(ctx_, &inv_window_);
  const auto inv_window_data = inv_window_.get_data_pointer<T>(ctx_);

  // Apply inv_window
  auto x_data = x->get_data_pointer<T>(ctx_);
  auto y_data = y->cast_data_and_get_pointer<T>(ctx_, true);
  for (int b = 0; b < batch_size; b++) {
    if (center_) {
      // Avoid division by zero for padding region.
      for (int i = fft_size_ / 2; i < size - fft_size_ / 2; i++) {
        y_data[b * size + i] = x_data[b * size + i] / inv_window_data[i];
      }
    } else {
      for (int i = 0; i < size; i++) {
        y_data[b * size + i] = x_data[b * size + i] / inv_window_data[i];
      }
    }
  }

  // Clear internal buffer
  inv_window_.data()->array()->clear();
}

template <typename T>
void ISTFT<T>::apply_inv_window_backward(Variable *x, Variable *y,
                                         const bool accum) {
  // Create inv_window
  const auto batch_size = x->shape()[0];
  const auto size = x->size() / batch_size;
  calculate_inv_window(ctx_, &inv_window_);
  const auto inv_window_data = inv_window_.get_data_pointer<T>(ctx_);

  // Apply inv_window
  auto x_grad = x->cast_grad_and_get_pointer<T>(ctx_, !accum);
  const auto y_grad = y->get_grad_pointer<T>(ctx_);

  for (int b = 0; b < batch_size; b++) {
    if (center_) {
      // Avoid division by zero for padding region.
      for (int i = 0; i < size; i++) {
        if (!(fft_size_ / 2 <= i && i < size - fft_size_ / 2)) {
          x_grad[b * size + i] = (T)0;
        } else {
          x_grad[b * size + i] = y_grad[b * size + i] / inv_window_data[i] +
                                 (accum ? x_grad[b * size + i] : (T)0);
        }
      }
    } else {
      for (int i = 0; i < size; i++) {
        x_grad[b * size + i] = y_grad[b * size + i] / inv_window_data[i] +
                               (accum ? x_grad[b * size + i] : (T)0);
      }
    }
  }

  // Clear internal buffer
  inv_window_.data()->array()->clear();
}

template <typename T>
void ISTFT<T>::forward_impl(const Variables &inputs, const Variables &outputs) {
  auto y_r = inputs[0];
  auto y_i = inputs[1];
  auto x = outputs[0];

  // Performe IDFT and part of overlap-add.
  calculate_conv_weight(conv_cos_, conv_sin_);
  deconv_->forward({y_r, &conv_cos_}, {&x_cos_});
  deconv_->forward({y_i, &conv_sin_}, {&x_sin_});

  // Compute rest of overlap-add and slicing to original signal.
  if (center_) {
    add2_->forward({&x_cos_, &x_sin_}, {&add2_out_});

    // Remove channel axis temporally
    const auto add2_out_s = add2_out_.shape();
    add2_out_.reshape({add2_out_s[0], add2_out_s[2]}, false);

    if (as_stft_backward_) {
      // Use pad backward instead of slice forward because we need to accumulate
      // gradients on padding region especially when `pad_mode == "reflect"`.
      Variable pad_in(x->shape());
      Variable pad_out(add2_out_.shape());
      // Swap data and grad
      pad_in.set_grad(x->data());
      pad_out.set_grad(add2_out_.data());
      pad_->backward({&pad_in}, {&pad_out}, {true}, {false});
    } else {
      apply_inv_window_forward(&add2_out_, &add2_out_);
      slice_->forward({&add2_out_}, {x});
    }

    // Restore shape
    add2_out_.reshape(add2_out_s, false);
  } else {
    // Add channel axis temporally
    const auto xs = x->shape();
    x->reshape({xs[0], 1, xs[1]}, false);

    if (as_stft_backward_) {
      add2_->forward({&x_cos_, &x_sin_}, {x});
    } else {
      add2_->forward({&x_cos_, &x_sin_}, {&add2_out_});
      apply_inv_window_forward(&add2_out_, x);
    }

    // Restore shape
    x->reshape(xs, false);
  }

  // Clear internal buffers
  conv_cos_.data()->array()->clear();
  conv_sin_.data()->array()->clear();
  x_cos_.data()->array()->clear();
  x_sin_.data()->array()->clear();
  add2_out_.data()->array()->clear();
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

  calculate_conv_weight(conv_cos_, conv_sin_);

  // Execute in reverse order of forward_impl.

  if (center_) {
    // Remove channel axis temporally
    const auto deconv_out_s = deconv_out_.shape();
    deconv_out_.reshape({deconv_out_s[0], deconv_out_s[2]}, false);
    if (as_stft_backward_) {
      // Use pad forward instead of slice backward because we need to distribute
      // gradients to padding region especially when `pad_mode == "reflect"`.
      Variable pad_in(x->shape());
      Variable pad_out(deconv_out_.shape());
      // Swap data and grad
      pad_in.set_data(x->grad());
      pad_out.set_data(deconv_out_.grad());
      pad_->forward({&pad_in}, {&pad_out});
    } else {
      slice_->backward({&deconv_out_}, {x}, {true}, {false});
      apply_inv_window_backward(&deconv_out_, &deconv_out_, false);
    }
    // Restore shape
    deconv_out_.reshape(deconv_out_s, false);
  } else {
    if (!as_stft_backward_) {
      // Add channel axis temporally
      const auto xs = x->shape();
      x->reshape({xs[0], 1, xs[1]}, false);

      apply_inv_window_backward(&deconv_out_, x, false);

      // Restore shape
      x->reshape(xs, false);
    }
  }

  if (as_stft_backward_ && !center_) {
    // Add channel axis temporally
    const auto xs = x->shape();
    x->reshape({xs[0], 1, xs[1]}, false);

    deconv_->backward({y_r, &conv_cos_}, {x}, {propagate_down[0], false},
                      {accum[0], false});
    deconv_->backward({y_i, &conv_sin_}, {x}, {propagate_down[1], false},
                      {accum[1], false});

    // Restore shape
    x->reshape(xs, false);
  } else {
    deconv_->backward({y_r, &conv_cos_}, {&deconv_out_},
                      {propagate_down[0], false}, {accum[0], false});
    deconv_->backward({y_i, &conv_sin_}, {&deconv_out_},
                      {propagate_down[1], false}, {accum[1], false});
  }

  // Clear internal buffers
  conv_cos_.data()->array()->clear();
  conv_sin_.data()->array()->clear();
  deconv_out_.grad()->array()->clear();
}
}
