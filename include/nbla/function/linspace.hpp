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

#ifndef NBLA_FUNCTION_LINSPACE_HPP
#define NBLA_FUNCTION_LINSPACE_HPP

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(Linspace, float, float, int);

/** Generate a one-dimensional vector/tensor of size `num`
whose values are evenly spaced from `start` to `end`, inclusive.

Inputs:
- none

Outputs:
- 1-D array.

@tparam T Data type for computation.

@param start Start value.

@param stop End value.

@param num Size of the constructed vector/tensor.

\ingroup FunctionImplGrp
 */
template <typename T> class Linspace : public BaseFunction<float, float, int> {
protected:
  float start_;
  float stop_;
  int num_;
  double step_;

public:
  Linspace(const Context &ctx, float start, float stop, int num)
      : BaseFunction(ctx, start, stop, num), start_(start), stop_(stop),
        num_(num) {}
  virtual ~Linspace() {}
  virtual shared_ptr<Function> copy() const {
    return create_Linspace(ctx_, start_, stop_, num_);
  }
  virtual int min_inputs() { return 0; }
  virtual int min_outputs() { return 1; }
  virtual vector<dtypes> in_types() { return vector<dtypes>{}; }
  virtual vector<dtypes> out_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cpu>()->array_classes();
  }
  virtual string name() { return "Linspace"; }
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
    return false;
  }
};
}
#endif
