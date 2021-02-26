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

/** ReLU
 */
#ifndef __NBLA_FUNCTION_RELU_HPP__
#define __NBLA_FUNCTION_RELU_HPP__

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

#include <memory>
#include <string>

namespace nbla {

using std::string;

NBLA_REGISTER_FUNCTION_HEADER(ReLU, bool);

/** Rectified Linear Unit (ReLU) defined as
@f[
y_i = \max (0, x_i).
@f]

Inputs:
- N-D array.

Outputs:
- N-D array.

@tparam T Data type for computation.
@param inplace The output array is will be shared with the input array if true.
\ingroup FunctionImplGrp
 */
template <typename T> class ReLU : public BaseFunction<bool> {
protected:
  bool inplace_;

public:
  ReLU(const Context &ctx, bool inplace)
      : BaseFunction(ctx, inplace), inplace_(inplace) {}
  virtual ~ReLU() {}
  virtual shared_ptr<Function> copy() const {
    return create_ReLU(ctx_, inplace_);
  }
  virtual int min_inputs() { return 1; }
  virtual int min_outputs() { return 1; }
  virtual vector<dtypes> in_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual vector<dtypes> out_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual string name() { return "ReLU"; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cpu>()->array_classes();
  }
  virtual bool grad_depends_output_data(int i, int o) const { return inplace_; }
  virtual int inplace_data(int i) const {
    return inplace_ ? Function::INPLACE : Function::NOT_INPLACE;
  }
  virtual int inplace_data_with(int i) const { return 0; }

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
