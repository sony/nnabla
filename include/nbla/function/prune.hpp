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

#ifndef NBLA_FUNCTION_PRUNE_HPP
#define NBLA_FUNCTION_PRUNE_HPP

#include <nbla/array/cpu_array.hpp>
#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(Prune, float);

/** Prune function defined as
@f[
q_i = \left \{
  \begin{array}{ll}
  0   & abs(x_i) < threshold \\
  x_i & otherwise
  \end{array}
  \right.
@f]

\f$ threshold \f$ is determined by `threshold = np.sort(np.abs(x))[int((x.size -
1) * rate)]`.

Inputs:
- N-D array.
Outputs:
- N-D array.
@tparam T Data type for computation.
@param rate Rate for pruning. The rage is [0.0, 1.0] and 0.9 rate is default.
\ingroup FunctionImplGrp
 */
template <typename T> class Prune : public BaseFunction<float> {

protected:
  float rate_;
  int thresh_idx_;

public:
  Prune(const Context &ctx, float rate)
      : BaseFunction(ctx, rate), rate_(rate) {}
  virtual ~Prune() {}
  virtual shared_ptr<Function> copy() const {
    return create_Prune(ctx_, rate_);
  }
  virtual int min_inputs() { return 1; }
  virtual int min_outputs() { return 1; }
  virtual vector<dtypes> in_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual vector<dtypes> out_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cpu>()->array_classes();
  }
  virtual string name() { return "Prune"; }
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
    // Because of STE (Straight Through Estimator)
    return false;
  }
};
}
#endif
