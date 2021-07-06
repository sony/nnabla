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

/** Mean Subtraction
 */
#ifndef __NBLA_FUNCTION_RUNNINGMEAN_HPP__
#define __NBLA_FUNCTION_RUNNINGMEAN_HPP__

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

#include <vector>

using std::vector;

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(MeanSubtraction, int, bool);

/** MeanSubtraction normalizes input to mean 0. Using this as a preprocess has
the effect of improving accuracy in image classification and the like.

At training time defined as
@f[
\begin{array}{lcl}
\mu &=& \frac{1}{M} \sum x_i \\
rm &=& ({\rm decay\_rate}) rm + (1 - {\rm decay\_rate}) \mu \\
y_i &=& x_i - rm
\end{array}
@f]

At validation time defined as
@f[
y_i = x_i - rm
@f]

Inputs:
- N-D array of input.
- N-D array of running mean (modified during forward execution).
- Scalar of num of iteration of running mean (modified during forward
  execution).

Outputs:
- N-D array.

@tparam T Data type for computation.
@param base_axis Base axis of mean subtraction operation. Dimensions up to
base_axis is treated as sample dimension.
@param update_running_mean Update running mean during forward execution.

@note The backward performs an approximated differentiation that takes into
      account only the latest mini-batch.

\ingroup FunctionImplGrp
 */
template <typename T> class MeanSubtraction : public BaseFunction<int, bool> {
protected:
  int base_axis_;
  bool update_running_mean_;
  Variable mean_;
  Size_t size0_, size1_;

public:
  MeanSubtraction(const Context &ctx, int base_axis, bool update_running_mean)
      : BaseFunction(ctx, base_axis, update_running_mean),
        base_axis_(base_axis), update_running_mean_(update_running_mean) {}
  virtual ~MeanSubtraction() {}
  virtual shared_ptr<Function> copy() const {
    return create_MeanSubtraction(ctx_, base_axis_, update_running_mean_);
  }
  virtual vector<dtypes> in_types() {
    return vector<dtypes>{get_dtype<T>(), get_dtype<T>(), get_dtype<int>()};
  }
  virtual vector<dtypes> out_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual int min_inputs() { return 2; }
  virtual int min_outputs() { return 1; }
  virtual string name() { return "MeanSubtraction"; }
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
  NBLA_API virtual void forward_impl_batch(const Variables &inputs,
                                           const Variables &outputs);
  NBLA_API virtual void forward_impl_global(const Variables &inputs,
                                            const Variables &outputs);
  NBLA_API virtual void backward_impl_batch(const Variables &inputs,
                                            const Variables &outputs,
                                            const vector<bool> &propagate_down,
                                            const vector<bool> &accum);
  NBLA_API virtual void backward_impl_global(const Variables &inputs,
                                             const Variables &outputs,
                                             const vector<bool> &propagate_down,
                                             const vector<bool> &accum);
  NBLA_API virtual void recompute_impl(const Variables &inputs,
                                       const Variables &outputs);
  virtual bool grad_depends_input_data_impl(int i, int j) const {
    if (update_running_mean_) { // Training mode.
      if (i == 0 && j == 2)
        return true;
    }
    return false;
  }
  virtual bool overwrite_input_data_in_forward_impl(int i) const {
    if (i == 1) {
      return true;
    }
    return false;
  }
};
}
#endif
