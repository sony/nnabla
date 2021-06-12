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

/** Batch Normalization
 */
#ifndef __NBLA_FUNCTION_BATCHNORM_HPP__
#define __NBLA_FUNCTION_BATCHNORM_HPP__

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

#include <nbla/function/add2.hpp>
#include <nbla/function/add_scalar.hpp>
#include <nbla/function/identity.hpp>
#include <nbla/function/mul2.hpp>
#include <nbla/function/pow_scalar.hpp>
#include <nbla/function/sub2.hpp>
#include <nbla/function/sum.hpp>

#include <nbla/imperative.hpp>
#include <vector>

using std::vector;

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(BatchNormalization, const vector<int> &, float,
                              float, bool, bool, bool);

/** Batch normalization at training time defined as
@f[
\begin{array}{lcl}
\mu &=& \frac{1}{M} \sum x_i\\
\sigma^2 &=& \frac{1}{M} \left(\sum x_i - \mu\right)^2\\
\hat{x}_i &=& \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} \\
y_i &=& \hat{x}_i \gamma + \beta.
\end{array}
@f]
In testing, mean and variance computed by moving average calculated during
training are used.

Inputs:
- N-D array of input.
- N-D array of beta which is learned.
- N-D array of gamma which is learned.
- N-D array of running mean (modified during forward execution).
- N-D array of running variance (modified during forward execution).

Outputs (1 or 3):
- N-D array.
- (Optional) N-D array of batch mean.
- (Optional) N-D array of batch variance.

@tparam T Data type for computation.

@param axes Axes mean and variance are taken.
@param decay_rate Decay rate of running mean and variance.
@param eps Tiny value to avoid zero division by std.
@param batch_stat Use mini-batch statistics rather than running ones.

@sa Ioffe and Szegedy, Batch Normalization: Accelerating Deep Network Training
by Reducing Internal Covariate Shift. https://arxiv.org/abs/1502.03167

\ingroup FunctionImplGrp
 */
template <typename T>
class BatchNormalization
    : public BaseFunction<const vector<int> &, float, float, bool, bool, bool> {
protected:
  vector<int> axes_;
  float decay_rate_;
  float eps_;
  bool batch_stat_;
  bool no_scale_;
  bool no_bias_;
  // beta, gamma, mean, variance index
  int b_idx_, g_idx_, m_idx_, v_idx_;
  Variable mean_;
  Variable var_;
  Size_t size0_, size1_, size2_, size02_, size12_;
  shared_ptr<Function> identity_;
  shared_ptr<Function> add2_;
  shared_ptr<Function> sub2_;
  shared_ptr<Function> mul2_;
  shared_ptr<Function> add_epsilon_;
  shared_ptr<Function> square_root_;
  shared_ptr<Function> sum_;

public:
  BatchNormalization(const Context &ctx, const vector<int> axes,
                     float decay_rate, float eps, bool batch_stat,
                     bool no_scale, bool no_bias)
      : BaseFunction(ctx, axes, decay_rate, eps, batch_stat, no_scale, no_bias),
        axes_(axes), decay_rate_(decay_rate), eps_(eps),
        batch_stat_(batch_stat), no_scale_(no_scale), no_bias_(no_bias) {}
  virtual ~BatchNormalization() {}
  virtual shared_ptr<Function> copy() const {
    return create_BatchNormalization(ctx_, axes_, decay_rate_, eps_,
                                     batch_stat_, no_scale_, no_bias_);
  }
  virtual vector<dtypes> in_types() {
    return vector<dtypes>{get_dtype<T>(), get_dtype<T>(), get_dtype<T>(),
                          get_dtype<T>(), get_dtype<T>()};
  }
  virtual vector<dtypes> out_types() {
    return vector<dtypes>{get_dtype<T>(), get_dtype<T>(), get_dtype<T>()};
  }
  virtual int min_inputs() { return 3; }
  virtual int min_outputs() { return 1; }
  virtual string name() { return "BatchNormalization"; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cpu>()->array_classes();
  }
  virtual bool grad_depends_output_data(int i, int o) const {
    // Gradient computation always requires output mean and var.
    return o > 0;
  }

protected:
  NBLA_API virtual void setup_impl(const Variables &inputs,
                                   const Variables &outputs);
  NBLA_API virtual void forward_impl(const Variables &inputs,
                                     const Variables &outputs);
  NBLA_API virtual void recompute_impl(const Variables &inputs,
                                       const Variables &outputs);
  NBLA_API virtual void backward_impl(const Variables &inputs,
                                      const Variables &outputs,
                                      const vector<bool> &propagate_down,
                                      const vector<bool> &accum);
  NBLA_API virtual void forward_impl_batch(const Variables &inputs,
                                           const Variables &outputs,
                                           const bool update_inputs);
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
  virtual bool grad_depends_input_data_impl(int i, int j) const {
    if (batch_stat_) { // Training mode.
      if (i == 0) {
        if (j == 0 || j == g_idx_)
          return true;
      }
      if (i == g_idx_) {
        if (j == 0)
          return true;
      }
      return false;

    } else { // Testing mode.
      if (i == 0) {
        if (j == g_idx_ || j == v_idx_)
          return true;
      }
      if (i == g_idx_) {
        if (j == 0 || j == m_idx_ || j == v_idx_)
          return true;
      }
    }
    return false;
  }
  virtual bool overwrite_input_data_in_forward_impl(int i) const {
    if (i == m_idx_ || i == v_idx_) {
      return true;
    }
    return false;
  }
};
}
#endif
