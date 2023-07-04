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

#ifndef __NBLA_SOLVER_SGDW_HPP__
#define __NBLA_SOLVER_SGDW_HPP__
#include <nbla/solver.hpp>
#include <nbla/solver_registry.hpp>

namespace nbla {

NBLA_REGISTER_SOLVER_HEADER(SgdW, float /*lr*/, float /*momentum*/,
                            float /*wd*/);

/** SGDW.

\f[
m_{t} &\leftarrow \gamma m_{t-1} + \eta_t \alpha g_t\\
w_{t} &\leftarrow w_{t-1} - m_{t} - \eta_t \lambda w_{t-1}
\f]
where \f$g_t\f$ denotes a gradient,
\f$m_t\f$ is momentum of the gradient initialized with 0 at \f$t=0\f$,
\f$\eta _t\f$ is the scheduled learning rate,
\f$\lambda\f$ is the decoupled weight decay rate set by weight_decay method
(lazy evaluation),
and the rest is described in the argument documentation.

@param lr Initial learning rate (\f$\alpha\f$). Note that you have to manage the
scheduled
learning rate \f$\eta_t\f$ yourelf. By denoting learning rate updated at the
set_learning_rate  by \f$\alpha_t\f$,
we define \f$\eta_t = \frac{\alpha_t}{\alpha}\f$.
@param momentum Decay rate of momentum (\f$\gamma\f$).
@param wd The default weight decay rate (\f$\lambda\f$). The weight decay
operation is fused into the
update operation in SgdW. It uses this default decay rate unless you overwrite a
decay rate
via weight_decay for the next call of update.

@sa See the paper linked below for more details.
Loshchilov and Hutter, Decoupled Weight Decay Regularization.
https://arxiv.org/abs/1711.05101

\ingroup SolverImplGrp
*/
template <typename T> class NBLA_API SgdW : public Solver {
public:
  SgdW(const Context &ctx, float lr, float momentum, float wd);
  virtual ~SgdW();
  virtual string name() { return "SgdW"; }

  virtual float learning_rate() { return lr_; }
  virtual void set_learning_rate(float lr) { lr_ = lr; }

protected:
  float lr_; ///< learning rate
  float momentum_;
  float init_lr_;

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
