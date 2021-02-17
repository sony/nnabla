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

#ifndef NBLA_FUNCTION_STFT_HPP
#define NBLA_FUNCTION_STFT_HPP

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(STFT, int, int, int, const string &, bool,
                              const string &);

/**
    @todo Write doc.

Inputs:

Outputs:

\ingroup FunctionImplGrp
 */
template <typename T>
class STFT
    : public BaseFunction<int, int, int, const string &, bool, const string &> {
protected:
  int window_size_;
  int stride_;
  int fft_size_;
  const string window_type_;
  bool center_;
  const string pad_mode_;

  shared_ptr<Function> pad_;
  shared_ptr<Function> mul2_;
  shared_ptr<Function> conv_;
  Shape_t pad_out_shape_;
  Shape_t conv_weight_shape_;

public:
  STFT(const Context &ctx, int window_size, int stride, int fft_size,
       const string &window_type, bool center, const string &pad_mode)
      : BaseFunction(ctx, window_size, stride, fft_size, window_type, center,
                     pad_mode),
        window_size_(window_size), stride_(stride), fft_size_(fft_size),
        window_type_(window_type), center_(center), pad_mode_(pad_mode) {}
  virtual ~STFT() {}
  virtual shared_ptr<Function> copy() const {
    return create_STFT(ctx_, window_size_, stride_, fft_size_, window_type_,
                       center_, pad_mode_);
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
};
}
#endif
