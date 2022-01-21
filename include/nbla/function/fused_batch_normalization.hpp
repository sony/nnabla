// Copyright 2019,2020,2021 Sony Corporation.
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

#pragma once
#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

#include <nbla/function/add2.hpp>
#include <nbla/function/batch_normalization.hpp>
#include <nbla/function/relu.hpp>

#include <vector>

namespace nbla {

using std::vector;

NBLA_REGISTER_FUNCTION_HEADER(FusedBatchNormalization, const vector<int> &,
                              float, float, bool, const string &);

/** Batch normalization fused with add2 (adding a residual input) and
activation.

This is an equivalent operation to the following (in Python), but is more
computationally efficient:
@code{.py}
h = F.BatchNormalization(x, beta, gamma, mean, variance, *opts)
y = F.ReLU(h + z)
@endcode

Inputs:
- N-D array of input.
- N-D array of beta which is learned.
- N-D array of gamma which is learned.
- N-D array of running mean (modified during forward execution).
- N-D array of running variance (modified during forward execution).
- N-D array of N-D array of a residual input (optional). If not passed, the
  activation function will follow immediately after BN operation.

Outputs (1 or 3):
- N-D array.
- (Optional) N-D array of batch mean.
- (Optional) N-D array of batch variance.

@tparam T Data type for computation.

@param axes Axes mean and variance are taken.
@param decay_rate Decay rate of running mean and variance.
@param eps Tiny value to avoid zero division by std.
@param batch_stat Use mini-batch statistics rather than running ones.
@param nonlinearity Activation chosen from ("relu").

@sa Ioffe and Szegedy, Batch Normalization: Accelerating Deep Network Training
by Reducing Internal Covariate Shift. https://arxiv.org/abs/1502.03167

\ingroup FunctionImplGrp
 */
template <typename T>
class FusedBatchNormalization
    : public BaseFunction<const vector<int> &, float, float, bool,
                          const string &> {
protected:
  vector<int> axes_;
  float decay_rate_;
  float eps_;
  bool batch_stat_;
  string nonlinearity_;

  // Not ideal, but BN backward relies on forward results internally stored.
  shared_ptr<Function> bn_;

public:
  FusedBatchNormalization(const Context &ctx, const vector<int> axes,
                          float decay_rate, float eps, bool batch_stat,
                          const string &nonlinearity)

      : BaseFunction<const vector<int> &, float, float, bool, const string &>(
            ctx, axes, decay_rate, eps, batch_stat, nonlinearity),
        axes_(axes), decay_rate_(decay_rate), eps_(eps),
        batch_stat_(batch_stat), nonlinearity_(nonlinearity) {}
  virtual ~FusedBatchNormalization() {}
  virtual shared_ptr<Function> copy() const {
    return create_FusedBatchNormalization(
        this->ctx_, this->axes_, this->decay_rate_, this->eps_,
        this->batch_stat_, this->nonlinearity_);
  }
  virtual int min_inputs() { return 5; }
  virtual int min_outputs() { return 1; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cpu>()->array_classes();
  }
  virtual vector<dtypes> in_types() {
    return vector<dtypes>{get_dtype<T>(), get_dtype<T>(), get_dtype<T>(),
                          get_dtype<T>(), get_dtype<T>(), get_dtype<T>()};
  }
  virtual vector<dtypes> out_types() {
    return vector<dtypes>{get_dtype<T>(), get_dtype<T>(), get_dtype<T>()};
  }
  virtual string name() { return "FusedBatchNormalization"; }
  virtual bool grad_depends_output_data(int i, int o) const {
    // Gradient computation always requires output mean and var.
    // If nonlinearity is relu, then y is also needed.
    if (nonlinearity_ == "relu") {
      return o >= 0;
    } else {
      return o > 0;
    }
  }

protected:
  NBLA_API virtual void setup_impl(const Variables &inputs,
                                   const Variables &outputs);
  NBLA_API virtual void forward_impl(const Variables &inputs,
                                     const Variables &outputs);
  NBLA_API virtual void backward_impl(const Variables &inputs,
                                      const Variables &outputs,
                                      const vector<bool> &propagate_down,
                                      const vector<bool> &accum);
  NBLA_API virtual void recompute_impl(const Variables &inputs,
                                       const Variables &outputs);
  virtual bool grad_depends_input_data_impl(int i, int j) const {
    if (batch_stat_) { // Training mode.
      if (i == 0) {
        if (j == 0 || j == 2)
          return true;
      }
      if (i == 2) {
        if (j == 0)
          return true;
      }
      return false;

    } else { // Testing mode.
      if (i == 0) {
        if (j == 2 || j == 4)
          return true;
      }
      if (i == 2) {
        if (j == 0 || j == 3)
          return true;
      }
    }
    return false;
  }
  virtual bool overwrite_input_data_in_forward_impl(int i) const {
    if (i == 3 || i == 4) {
      return true;
    }
    return false;
  }

  NBLA_API virtual void relu_add2_backward(const Variables &inputs,
                                           const Variables &outputs,
                                           const vector<bool> &propagate_down,
                                           const vector<bool> &accum,
                                           Variable &relu_buf);
};

} // namespace nbla
