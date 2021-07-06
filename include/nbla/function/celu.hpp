// Copyright 2017,2018,2019,2020,2021 Sony Corporation.
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

/** CELU
 */
#ifndef __NBLA_FUNCTION_CELU_HPP__
#define __NBLA_FUNCTION_CELU_HPP__

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(CELU, double, int);

/** Concatenated Exponential Linear Unit (CELU) concatenates ELU outputs of
positive and negative inputs together at specified axis.

Inputs:
- N-D array.

Outputs:
- N-D array where axis dimension is doubled by concatenating.

@tparam T Data type for computation.
@param axis The ELU activations of positive inputs and negative inputs are
concatenated at axis.

\ingroup FunctionImplGrp
 */
template <typename T> class CELU : public BaseFunction<double, int> {
protected:
  double alpha_;
  int axis_;
  int size0_, size1_;

public:
  CELU(const Context &ctx, double alpha, int axis)
      : BaseFunction(ctx, alpha, axis), alpha_(alpha), axis_(axis) {}
  virtual ~CELU() {}
  virtual shared_ptr<Function> copy() const {
    return create_CELU(ctx_, alpha_, axis_);
  }
  virtual vector<dtypes> in_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual vector<dtypes> out_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual int min_inputs() { return 1; }
  virtual int min_outputs() { return 1; }
  virtual string name() { return "CELU"; }
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
  virtual bool grad_depends_input_data_impl(int i, int j) const { return true; }
};
}
#endif
