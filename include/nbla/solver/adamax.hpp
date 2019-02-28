// Copyright (c) 2017 Sony Corporation. All Rights Reserved.
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

#ifndef __NBLA_SOLVER_ADAMAX_HPP__
#define __NBLA_SOLVER_ADAMAX_HPP__
#include <nbla/solver.hpp>
#include <nbla/solver_registry.hpp>

namespace nbla {

NBLA_REGISTER_SOLVER_HEADER(Adamax, float, float, float, float);

/** Adamax solver defined as
\f[
\theta_{t+1} \leftarrow \theta_t - \alpha
\frac{\sqrt{1 - \beta_2^t}}{1 - \beta_1^t}
\frac{m_t}{u_t + \epsilon}
\f]
where \f$\theta_t\f$ is a gradient of a parameter, \f$m_t\f$ and \f$u_t\f$ are
moving average and exponentially weighted infinity norm of a sequence of
gradients \f$t=0,...,t\f$.

@param alpha \f$\alpha\f$ Learning rate.
@param beta1 \f$\beta_1\f$ Decay rate of moving mean.
@param beta2 \f$\beta_2\f$ Decay rate of exponentially weighted infinity norm.
@param eps \f$\epsilon\f$ Tiny factor for avoiding 0-division.

@sa See the paper linked below for more details.
Kingma and Ba, Adam: A Method for Stochastic Optimization.
https://arxiv.org/abs/1412.6980

\ingroup SolverImplGrp
*/
template <typename T> class NBLA_API Adamax : public Solver {
public:
  Adamax(const Context &ctx, float alpha, float beta1, float beta2, float eps);
  virtual ~Adamax();
  virtual string name() { return "Adamax"; }

  virtual float learning_rate() { return alpha_; }
  virtual void set_learning_rate(float lr) { alpha_ = lr; }

protected:
  // Settings.
  float alpha_; ///< \f$\alpha\f$
  float beta1_; ///< \f$\beta_1\f$
  float beta2_; ///< \f$\beta_2\f$
  float eps_;   ///< \f$\epsilon\f$

  // Functions.
  virtual void set_state_impl(const string &key, VariablePtr param);
  virtual void remove_state_impl(const string &key);
  virtual void update_impl(const string &key, VariablePtr param);
  NBLA_DECL_WEIGHT_DECAY();
  NBLA_DECL_CHECK_INF_GRAD();
  NBLA_DECL_CHECK_NAN_GRAD();
  NBLA_DECL_CHECK_INF_OR_NAN_GRAD();
  NBLA_DECL_SCALE_GRAD();
};
}
#endif
