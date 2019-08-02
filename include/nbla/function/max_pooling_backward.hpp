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

#ifndef NBLA_FUNCTION_MAX_POOLING_BACKWARD_HPP
#define NBLA_FUNCTION_MAX_POOLING_BACKWARD_HPP

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(MaxPoolingBackward, const vector<int> &,
                              const vector<int> &, bool, const vector<int> &,
                              bool);

/**
    @todo Write doc.

Inputs:

Outputs:

\ingroup FunctionImplGrp
 */
template <typename T>
class MaxPoolingBackward
    : public BaseFunction<const vector<int> &, const vector<int> &, bool,
                          const vector<int> &, bool> {
protected:
  const vector<int> kernel_;
  const vector<int> stride_;
  bool ignore_border_;
  const vector<int> pad_;
  bool channel_last_;

public:
  MaxPoolingBackward(const Context &ctx, const vector<int> &kernel,
                     const vector<int> &stride, bool ignore_border,
                     const vector<int> &pad, bool channel_last)
      : BaseFunction(ctx, kernel, stride, ignore_border, pad, channel_last),
        kernel_(kernel), stride_(stride), ignore_border_(ignore_border),
        pad_(pad), channel_last_(channel_last) {}
  virtual ~MaxPoolingBackward() {}
  virtual shared_ptr<Function> copy() const {
    return create_MaxPoolingBackward(ctx_, kernel_, stride_, ignore_border_,
                                     pad_, channel_last_);
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
  virtual string name() { return "MaxPoolingBackward"; }
  // TODO: This must be overridden if any of input grad depends on a output
  // data. See doc in function.hpp.
  // virtual bool grad_depends_output_data(int i, int o) const {
  // }
  // TODO: If any of data/grad storage is inplaced with any of output, you must
  // override some of these. See doc in function.hpp.
  // virtual int inplace_data(int i) const {
  // }
  // virtual int inplace_data_with(int i) const {
  // }
  // virtual int inplace_grad(int i) const {
  // }
  // virtual int inplace_grad_with(int i) const {
  // }
  // TODO: If you want to avoid clearing input buffers in any case, define this
  // function returning true.
  // virtual bool prohibit_clear_input_buffers() const {
  //   return true;
  // }
  // TODO: If you want to avoid zero-ing gradient of inputs even when accum is
  // true, uncomment the following function definition.
  // virtual bool prohibit_zero_input_grad() const {
  //   return true;
  // }

protected:
  NBLA_API virtual void setup_impl(const Variables &inputs,
                                   const Variables &outputs);
  NBLA_API virtual void forward_impl(const Variables &inputs,
                                     const Variables &outputs);
  NBLA_API virtual void backward_impl(const Variables &inputs,
                                      const Variables &outputs,
                                      const vector<bool> &propagate_down,
                                      const vector<bool> &accum);
};
}
#endif
