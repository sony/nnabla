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

/** Binary cross entropy
 */
#ifndef __NBLA_FUNCTION_BINARY_CROSS_ENTROPY_HPP__
#define __NBLA_FUNCTION_BINARY_CROSS_ENTROPY_HPP__

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(BinaryCrossEntropy);

/** BinaryCrossEntropy calculate the element-wise cross entropy between the
variable and label variables.
@f[
y_i = - \left(x^{(1)}_i * \ln \left(x^{(0)}_i\right) + \left(1 -
x^{(1)}_i\right) * \ln \left(1 - x^{(0)}_i\right)\right).
@f]

Inputs:
- Probabilities N-D array. \f$-\infty\f$ to \f$\infty\f$.
- Labels N-D array. Usually set as 0 or 1, but, unlike SigmoidCrossEntropy, it
  allows probability (0 to 1) as inputs and backpropagation can be done.

Outputs:
- Element-wise losses N-D array.

@tparam T Data type for computation.
\ingroup FunctionImplGrp
 */
template <typename T> class BinaryCrossEntropy : public BaseFunction<> {
public:
  BinaryCrossEntropy(const Context &ctx) : BaseFunction<>(ctx) {}
  virtual ~BinaryCrossEntropy() {}
  virtual shared_ptr<Function> copy() const {
    return create_BinaryCrossEntropy(ctx_);
  }
  virtual vector<dtypes> in_types() {
    return vector<dtypes>{get_dtype<T>(), get_dtype<T>()};
  }
  virtual vector<dtypes> out_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual int min_inputs() { return 2; }
  virtual int min_outputs() { return 1; }
  virtual string name() { return "BinaryCrossEntropy"; }
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
      if (j == 0 || j == 1)
        return true;
    }
    if (i == 1) {
      if (j == 0)
        return true;
    }
    return false;
  }
};
}
#endif
