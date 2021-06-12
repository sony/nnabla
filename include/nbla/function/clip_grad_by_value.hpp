// Copyright 2018,2019,2020,2021 Sony Corporation.
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

/** ClipGradByValue
 */
#ifndef __NBLA_FUNCTION_CLIPBYVALUE_HPP__
#define __NBLA_FUNCTION_CLIPBYVALUE_HPP__

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(ClipGradByValue);

/**

In forward pass, the function behaves the identity.

In backward pass,

@f[
\frac{\partial C}{\partial x} = \begin{cases}
        max & (\frac{\partial C}{\partial y} > max) \\
        \frac{\partial C}{\partial y} & (otherwise) \\
        min & (\frac{\partial C}{\partial y} < min)
    \end{cases},
@f]

Inputs:
- N-D array.

Outputs:
- N-D array.

@tparam T Data type for computation.
@param min Value of the scalar to clip to this `min` if the input value is less
than `min` in the forward pass.
@param max Value of the scalar to clip to this `max` if the input value is
greater than `max` in the forward pass.

\ingroup FunctionImplGrp

 */
template <typename T> class ClipGradByValue : public BaseFunction<> {

public:
  ClipGradByValue(const Context &ctx) : BaseFunction<>(ctx) {}
  virtual ~ClipGradByValue() {}
  virtual shared_ptr<Function> copy() const {
    return create_ClipGradByValue(ctx_);
  }
  virtual vector<dtypes> in_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual vector<dtypes> out_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual int min_inputs() { return 3; }
  virtual int min_outputs() { return 1; }
  virtual string name() { return "ClipGradByValue"; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cpu>()->array_classes();
  }
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
  virtual bool grad_depends_input_data_impl(int i, int j) const {
    if (i == 0) {
      if (j == 1 || j == 2)
        return true;
    }
    return false;
  }
};
}
#endif
