// Copyright 2019,2020,2021 Sony Corporation.
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
  virtual bool grad_depends_output_data(int i, int o) const { return false; }

protected:
  NBLA_API virtual void setup_impl(const Variables &inputs,
                                   const Variables &outputs);
  NBLA_API virtual void forward_impl(const Variables &inputs,
                                     const Variables &outputs);
  NBLA_API virtual void backward_impl(const Variables &inputs,
                                      const Variables &outputs,
                                      const vector<bool> &propagate_down,
                                      const vector<bool> &accum);
  virtual bool grad_depends_input_data_impl(int i, int j) const { return true; }
};
}
#endif
