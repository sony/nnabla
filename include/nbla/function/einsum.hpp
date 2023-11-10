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

#ifndef NBLA_FUNCTION_EINSUM_HPP
#define NBLA_FUNCTION_EINSUM_HPP

#include <nbla/computation_graph/variable.hpp>
#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(Einsum, const string &);

/**
Evaluates the Einstein summation convention on the inputs.

Inputs:
- x: List of N-D array.

Outputs:
- y: A N-D array.

@param equation A string that folllows Einstein summation convention.

\ingroup FunctionImplGrp
 */
template <typename T> class Einsum : public BaseFunction<const string &> {
protected:
  const string equation_;
  vector<CgVariablePtr> input_cg_variables_;
  CgVariablePtr last_output_cg_variable_;

public:
  Einsum(const Context &ctx, const string &equation)
      : BaseFunction(ctx, equation), equation_(equation) {}
  virtual ~Einsum() {}
  virtual shared_ptr<Function> copy() const {
    return create_Einsum(ctx_, equation_);
  }
  virtual int min_inputs() { return 1; }
  virtual int min_outputs() { return 1; }
  virtual vector<dtypes> in_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual vector<dtypes> out_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cpu>()->array_classes();
  }
  virtual string name() { return "Einsum"; }
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
};
} // namespace nbla
#endif
