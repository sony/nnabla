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

#ifndef NBLA_FUNCTION_ISTFT_HPP
#define NBLA_FUNCTION_ISTFT_HPP

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(ISTFT, int, int, int, const string &, bool);

/**
    @todo Write doc.

Inputs:

Outputs:

\ingroup FunctionImplGrp
 */
template <typename T>
class ISTFT : public BaseFunction<int, int, int, const string &, bool> {
protected:
  int window_size_;
  int stride_;
  int fft_size_;
  const string window_type_;
  bool center_;

  shared_ptr<Function> mul2_;
  shared_ptr<Function> div2_;
  shared_ptr<Function> sub2_;
  shared_ptr<Function> slice_;
  shared_ptr<Function> deconv_;
  Shape_t deconv_w_shape_;
  Shape_t deconv_y_shape_;

public:
  ISTFT(const Context &ctx, int window_size, int stride, int fft_size,
        const string &window_type, bool center)
      : BaseFunction(ctx, window_size, stride, fft_size, window_type, center),
        window_size_(window_size), stride_(stride), fft_size_(fft_size),
        window_type_(window_type), center_(center) {}
  virtual ~ISTFT() {}
  virtual shared_ptr<Function> copy() const {
    return create_ISTFT(ctx_, window_size_, stride_, fft_size_, window_type_,
                        center_);
  }
  virtual int min_inputs() { return 2; }
  virtual int min_outputs() { return 1; }
  virtual vector<dtypes> in_types() {
    return vector<dtypes>{get_dtype<T>(), get_dtype<T>()};
  }
  virtual vector<dtypes> out_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cpu>()->array_classes();
  }
  virtual string name() { return "ISTFT"; }

protected:
  NBLA_API virtual void setup_impl(const Variables &inputs,
                                   const Variables &outputs);
  NBLA_API virtual void calculate_conv_weight(Variable &conv_cos,
                                              Variable &conv_sin);
  NBLA_API virtual void forward_impl(const Variables &inputs,
                                     const Variables &outputs);
  NBLA_API virtual void backward_impl(const Variables &inputs,
                                      const Variables &outputs,
                                      const vector<bool> &propagate_down,
                                      const vector<bool> &accum);
};
}
#endif
