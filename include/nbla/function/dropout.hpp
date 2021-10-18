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

/** Dropout
 */
#ifndef __NBLA_FUNCTION_DROPOUT_HPP__
#define __NBLA_FUNCTION_DROPOUT_HPP__

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

#include <memory>
#include <random>
#include <string>

namespace nbla {

using std::string;

NBLA_REGISTER_FUNCTION_HEADER(Dropout, double, int);

/** Dropout defined as
@f[
  y = \left\{
    \begin{array}{ll}
      \frac{x}{1 - p} & \mbox{if } u > p \\
      0 & \mbox{otherwise}
    \end{array} \right.
@f]
where u is generated from uniform dist in [0, 1].

Inputs:
- N-D array.

Outputs:
- N-D array with the same shape as input.

@tparam T Data type for computation.
@param p @f$p@f$ in definition.
@param seed Random seed

\ingroup FunctionImplGrp

*/
template <typename T> class Dropout : public BaseFunction<double, int> {
protected:
  float p_;
  int seed_;
  float scale_; // = 1./(1.-p_)
  VariablePtr mask_;
  bool save_rng_ = false;
  std::mt19937 rgen_, rgen_for_recompute_;
  std::bernoulli_distribution rdist_;

public:
  Dropout(const Context &ctx, double p, int seed = -1)
      : BaseFunction(ctx, p, seed), p_(p), seed_(seed) {}
  virtual ~Dropout() {}
  virtual shared_ptr<Function> copy() const {
    return create_Dropout(ctx_, p_, seed_);
  }
  virtual int min_inputs() { return 1; }
  virtual int min_outputs() { return 1; }
  virtual vector<dtypes> in_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual vector<dtypes> out_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual string name() { return "Dropout"; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cpu>()->array_classes();
  }
  virtual bool need_setup_recompute(int o) const { return true; }
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
  NBLA_API virtual void setup_recompute_impl(const Variables &inputs,
                                             const Variables &outputs);
  NBLA_API virtual void recompute_impl(const Variables &inputs,
                                       const Variables &outputs);
  void dropout(const Variables &inputs, const Variables &outputs,
               std::mt19937 &rgen);

  /** Clear the member variable, in particular "mask_". */
  NBLA_API void clear_buffer();

  virtual bool grad_depends_input_data_impl(int i, int j) const {
    return false;
  }
};
}
#endif
