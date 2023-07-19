// Copyright 2022 Sony Group Corporation.
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

#ifndef __NBLA_SOLVER_LAMB_HPP__
#define __NBLA_SOLVER_LAMB_HPP__
#include <nbla/solver.hpp>
#include <nbla/solver_registry.hpp>

namespace nbla {

NBLA_REGISTER_SOLVER_HEADER(Lamb, float /* eta */, float /* beta1 */,
                            float /* beta2 */, float /* gamma_l */,
                            float /* gamma_u */, float /* eps */,
                            bool /* bias_correction */
);

/** LAMB.


\f[
m_t &\leftarrow \beta_1 m_{t-1} + (1 - \beta_1) g_t\\
v_t &\leftarrow \beta_2 v_{t-1} + (1 - \beta_2) g_t^2\\
\hat{m} &= m_t / (1-\beta_1^t)\\
\hat{v} &= v_t / (1-\beta_2^t)\\
r &= \frac{\hat{m}}{\sqrt{\hat{v}}+\epsilon}\\
w_t &\leftarrow w_{t-1} - \eta_t \frac{\phi (\|w_{t-1}\|)}{\|r + \lambda w_{t-1}
\|} \left(r + \lambda w_{t-1} \right)
\f]

where \f$g_t\f$ denotes a gradient,
\f$m_t\f$ and \f$v_t\f$ are 1st and 2nd order momentum of the gradient
initialized with 0 at \f$t=0\f$,
\f$\lambda\f$ is the decoupled weight decay rate set by weight_decay method
(lazy evaluation),
\f$\phi\f$ is a scaling function defined as \f$\phi(z)=\min\{\max\{z,
\gamma_l\}, \gamma_u\}\f$,
and the rest is described in the arguments.

@param eta Learning rate (\f$\eta_t\f$).
@param beta1 Decay rate of first-order momentum (\f$\beta_1\f$).
@param beta2 Decay rate of second-order momentum (\f$\beta_2\f$).
@param gamma_l Lower bound of the clamp scaling function \f$\phi\f$
(\f$\gamma_l\f$).
@param gamma_u Upper bound of the clamp scaling function \f$\phi\f$
(\f$\gamma_u\f$).
@param eps Small value for avoiding zero division (\f$\epsilon\f$).
@param bias_correction Whether to apply bias correction in momentum computation
\f$\hat{m}\f$ and \f$\hat{v}\f$.

@sa See the paper linked below for more details.
Yang You, Jing Li, Sashank Reddi. Large Batch Optimization for Deep Learning:
Training BERT in 76 minutes.
https://arxiv.org/abs/1904.00962

\ingroup SolverImplGrp
*/
template <typename T> class NBLA_API Lamb : public Solver {
public:
  Lamb(const Context &ctx, float eta, float beta1, float beta2, float gamma_l,
       float gamma_u, float eps, bool bias_correction);
  virtual ~Lamb();
  virtual string name() { return "Lamb"; }

  virtual float learning_rate() { return eta_; }
  virtual void set_learning_rate(float lr) { eta_ = lr; }

protected:
  // Settings.
  float eta_;   ///< \f$\eta\f$
  float beta1_; ///< \f$\beta_1\f$
  float beta2_; ///< \f$\beta_2\f$
  float gamma_l_;
  float gamma_u_;
  float eps_; ///< \f$\epsilon\f$
  bool bias_correction_;

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
} // namespace nbla
#endif
