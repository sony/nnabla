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

/** Batch Normalization
 */
#ifndef __NBLA_FUNCTION_SYNC_BATCHNORM_HPP__
#define __NBLA_FUNCTION_SYNC_BATCHNORM_HPP__

#include <nbla/communicator.hpp>
#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function/batch_normalization.hpp>
#include <nbla/function_registry.hpp>

#include <vector>

using std::vector;

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(SyncBatchNormalization,
                              const std::shared_ptr<Communicator> &,
                              const std::string &, const vector<int> &, float,
                              float, bool);

/** Batch normalization with sync between other processes at training time
defined as
@f[
\begin{array}{lcl}
\mu &=& \frac{1}{M} \sum x_i\\
\sigma^2 &=& \frac{1}{M} \left(\sum x_i - \mu\right)^2\\
\hat{x}_i &=& \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} \\
y_i &=& \hat{x}_i \gamma + \beta.
\end{array}
@f]

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

@param comm The communicator
@param group The name of the communicator group
@param axes Axes mean and variance are taken.
@param decay_rate Decay rate of running mean and variance.
@param eps Tiny value to avoid zero division by std.

@sa Implementing Synchronized Multi-GPU Batch Normalization
https://hangzhang.org/PyTorch-Encoding/notes/syncbn.html

\ingroup FunctionImplGrp
 */
template <typename T>
class SyncBatchNormalization : public BatchNormalization<T> {
protected:
  std::shared_ptr<Communicator> comm_;
  std::string group_;
  size_t num_processes_;

public:
  SyncBatchNormalization(const Context &ctx,
                         const std::shared_ptr<Communicator> &comm,
                         const std::string &group, const vector<int> axes,
                         float decay_rate, float eps, bool batch_stat)
      : BatchNormalization<T>(ctx, axes, decay_rate, eps, batch_stat,
                              false /* no_scale */, false /* no_bias */),
        comm_(comm), group_(group) {}
  virtual ~SyncBatchNormalization() {}
  virtual shared_ptr<Function> copy() const override {
    return create_SyncBatchNormalization(this->ctx_, this->comm_, this->group_,
                                         this->axes_, this->decay_rate_,
                                         this->eps_, this->batch_stat_);
  }
  virtual string name() override { return "SyncBatchNormalization"; }
  virtual bool grad_depends_output_data(int i, int o) const { return o > 0; }

protected:
  NBLA_API virtual void setup_impl(const Variables &inputs,
                                   const Variables &outputs) override;
  NBLA_API virtual void forward_impl_batch(const Variables &inputs,
                                           const Variables &outputs,
                                           const bool update_inputs) override;
  NBLA_API virtual void backward_impl_batch(const Variables &inputs,
                                            const Variables &outputs,
                                            const vector<bool> &propagate_down,
                                            const vector<bool> &accum) override;
  virtual bool grad_depends_input_data_impl(int i, int j) const {
    if (i == 0) {
      if (j == 0)
        return true;
    }
    if (i == 1) {
      if (j == 0)
        return true;
    }
    if (i == 2) {
      if (j == 0 || j == 2)
        return true;
    }
    return false;
  }
  virtual bool overwrite_input_data_in_forward_impl(int i) const {
    if (i == 3 || i == 4) {
      return true;
    }
    return false;
  }
};
}
#endif
