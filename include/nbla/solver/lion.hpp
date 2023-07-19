// Copyright 2023 Sony Group Corporation.
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

#ifndef __NBLA_SOLVER_LION_HPP__
#define __NBLA_SOLVER_LION_HPP__
#include <nbla/solver.hpp>
#include <nbla/solver_registry.hpp>

namespace nbla {

NBLA_REGISTER_SOLVER_HEADER(Lion, float /*lr*/, float /*beta1*/,
                            float /* beta2 */);

/** LION solver defined as
\f[
u = \beta_1 m_{t} + (1 - \beta_1) g_t\\
u = {\rm sign}(u)\\
m_{t+1} &\leftarrow \beta_2 m_{t} + (1 - \beta_2) g_t\\
w_{t+1} &\leftarrow w_t - \alpha \left( u + \lambda w_t \right)
\f]

@param lr\f$\alpha\f$ Learning rate.
@param beta1 \f$\beta_1\f$ Decay rate.
@param beta2 \f$\beta_2\f$ Decay rate.

@sa See the paper linked below for more details.
Xiangning Chen et al., Symbolic Discovery of Optimization Algorithms.
https://arxiv.org/abs/2302.06675

\ingroup SolverImplGrp
*/
template <typename T> class NBLA_API Lion : public Solver {
public:
  Lion(const Context &ctx, float lr, float beta1, float beta2);
  virtual ~Lion();
  virtual string name() { return "Lion"; }

  virtual float learning_rate() { return lr_; }
  virtual void set_learning_rate(float lr) { lr_ = lr; }

protected:
  float lr_;
  float beta1_;
  float beta2_;

  virtual void set_state_impl(const string &key, VariablePtr param) override;
  virtual void remove_state_impl(const string &key) override;
  virtual void update_impl(const string &key, VariablePtr param) override;
  NBLA_DECL_WEIGHT_DECAY();
  NBLA_DECL_CLIP_GRAD_BY_NORM();
  NBLA_DECL_CHECK_INF_GRAD();
  NBLA_DECL_CHECK_NAN_GRAD();
  NBLA_DECL_CHECK_INF_OR_NAN_GRAD();
  NBLA_DECL_SCALE_GRAD();
};
} // namespace nbla
#endif
