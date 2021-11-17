// Copyright 2018,2019,2020,2021 Sony Corporation.
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

#ifndef __NBLA_SOLVER_AMSGRAD_HPP__
#define __NBLA_SOLVER_AMSGRAD_HPP__
#include <nbla/solver.hpp>
#include <nbla/solver_registry.hpp>

namespace nbla {

NBLA_REGISTER_SOLVER_HEADER(AMSGRAD, float, float, float, float, bool);

/** AMSGRAD solver defined as
\f[
m_t \leftarrow \beta_1 m_{t-1} + (1 - \beta_1) g_t\\
v_t \leftarrow \beta_2 v_{t-1} + (1 - \beta_2) g_t^2\\
\hat{v_t} = \max(\hat{v_{t-1}}, v_t)\\
\theta_{t+1} \leftarrow \theta_t - \alpha
\frac{m_t}{\sqrt{\hat{v_t}} + \epsilon}
\f]
where \f$\theta_t\f$ is a gradient of a parameter, \f$m_t\f$ and \f$v_t\f$ are
moving average and 0-mean variance of a sequence of gradients \f$t=0,...,t\f$.

@param alpha \f$\alpha\f$ Learning rate.
@param beta1 \f$\beta_1\f$ Decay rate of moving mean.
@param beta2 \f$\beta_2\f$ Decay rate of moving 0-mean variance.
@param eps \f$\epsilon\f$ Small value for avoiding zero
division(:math:`\epsilon`). Note this does not appear in the paper.
@param bias_correction Apply bias correction to moving averages defined in ADAM.
Note this does not appear in the paper.

@sa See the paper linked below for more details.
Reddi et al. On the convergence of ADAM and beyond.
https://openreview.net/pdf?id=ryQu7f-RZ


\ingroup SolverImplGrp
*/
template <typename T> class NBLA_API AMSGRAD : public Solver {
public:
  AMSGRAD(const Context &ctx, float alpha, float beta1, float beta2, float eps,
          bool bias_correction);
  virtual ~AMSGRAD();
  virtual string name() { return "AMSGRAD"; }

  virtual float learning_rate() { return alpha_; }
  virtual void set_learning_rate(float lr) { alpha_ = lr; }

protected:
  // Settings.
  float alpha_;          ///< \f$\alpha\f$
  float beta1_;          ///< \f$\beta_1\f$
  float beta2_;          ///< \f$\beta_2\f$
  float eps_;            ///< \f$\epsilon\f$
  bool bias_correction_; ///< \bias_correction

  // Functions.
  virtual void set_state_impl(const string &key, VariablePtr param);
  virtual void remove_state_impl(const string &key);
  virtual void update_impl(const string &key, VariablePtr param);
  NBLA_DECL_WEIGHT_DECAY();
  NBLA_DECL_CLIP_GRAD_BY_NORM();
  NBLA_DECL_CHECK_INF_GRAD();
  NBLA_DECL_CHECK_NAN_GRAD();
  NBLA_DECL_CHECK_INF_OR_NAN_GRAD();
  NBLA_DECL_SCALE_GRAD();
};
}
#endif
