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

#ifndef NBLA_FUNCTION_STFT_HPP
#define NBLA_FUNCTION_STFT_HPP

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(STFT, int, int, int, const string &, bool,
                              const string &, bool);

// Prototype declaration for cross referencing between STFT and ISTFT.
template <typename T> class ISTFT;

/**
    @todo Write doc.

Inputs:

Outputs:

\ingroup FunctionImplGrp
 */
template <typename T>
class STFT : public BaseFunction<int, int, int, const string &, bool,
                                 const string &, bool> {
  friend ISTFT<T>;

protected:
  int window_size_;
  int stride_;
  int fft_size_;
  const string window_type_;
  bool center_;
  const string pad_mode_;
  bool as_istft_backward_;

  FunctionPtr pad_;
  FunctionPtr mul2_;
  FunctionPtr conv_;

  // Intermediate buffers
  Variable window_, dft_w_r_, dft_w_i_;
  Variable pad_out_;
  Variable conv_r_, conv_i_;

  // Only for `as_istft_backward == true`.
  shared_ptr<ISTFT<T>> istft_cpu_;
  Variable x_inv_window_;
  Variable conv_grad_;

public:
  STFT(const Context &ctx, int window_size, int stride, int fft_size,
       const string &window_type, bool center, const string &pad_mode,
       bool as_istft_backward)
      : BaseFunction(ctx, window_size, stride, fft_size, window_type, center,
                     pad_mode, as_istft_backward),
        window_size_(window_size), stride_(stride), fft_size_(fft_size),
        window_type_(window_type), center_(center), pad_mode_(pad_mode),
        as_istft_backward_(as_istft_backward) {}
  virtual ~STFT() {}
  virtual shared_ptr<Function> copy() const {
    return create_STFT(ctx_, window_size_, stride_, fft_size_, window_type_,
                       center_, pad_mode_, as_istft_backward_);
  }
  virtual int min_inputs() { return 1; }
  virtual int min_outputs() { return 2; }
  virtual vector<dtypes> in_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual vector<dtypes> out_types() {
    return vector<dtypes>{get_dtype<T>(), get_dtype<T>()};
  }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cpu>()->array_classes();
  }
  virtual string name() { return "STFT"; }
  virtual bool grad_depends_output_data(int i, int o) const { return false; }

protected:
  NBLA_API virtual void setup_impl(const Variables &inputs,
                                   const Variables &outputs);
  NBLA_API virtual void calculate_conv_weight(Variable &conv_r,
                                              Variable &conv_i);
  NBLA_API virtual void forward_impl(const Variables &inputs,
                                     const Variables &outputs);
  NBLA_API virtual void backward_impl(const Variables &inputs,
                                      const Variables &outputs,
                                      const vector<bool> &propagate_down,
                                      const vector<bool> &accum);

  // Only for `as_istft_backward == true`.
  NBLA_API virtual void apply_inv_window_forward(Variable *x, Variable *y);
  NBLA_API virtual void apply_inv_window_backward(Variable *x, Variable *y,
                                                  const bool accum);

  virtual bool grad_depends_input_data_impl(int i, int j) const {
    return false;
  }
};
}
#endif
