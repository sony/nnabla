// Copyright 2020,2021 Sony Corporation.
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

#ifndef __NBLA_SOLVER_RMSPROPGRAVES_HPP__
#define __NBLA_SOLVER_RMSPROPGRAVES_HPP__
#include <nbla/solver.hpp>
#include <nbla/solver_registry.hpp>

namespace nbla {

NBLA_REGISTER_SOLVER_HEADER(RMSpropGraves, float /*lr*/, float /*decay*/,
                            float /*momentum*/, float /*eps*/);

/** RMSpropGraves. This is defined as

\f[
n_t \leftarrow \rho n_{t-1} + \left(1 - \rho \right) e^2\\
g_t \leftarrow \rho g_{t-1} + \left(1 - \rho \right) e\\
d_t \leftarrow \beta d_{t-1} - \eta \frac{e}{\sqrt{n_t - {g_t}^2 + \epsilon}}\\
w_{t+1} \leftarrow w_t + d_t
\f]

@param lr \f$\eta\f$ Learning rate.
@param decay \f$\rho\f$ Decay rate.
@param momentum \f$\beta\f$ Momentum.
@param eps \f$\epsilon\f$ Tiny factor for avoiding 0-division.

@sa See the paper linked below for more details.
A. Graves
Generating Sequences With Recurrent Neural Networks
http://arxiv.org/pdf/1308.0850v5.pdf


\ingroup SolverImplGrp
*/
template <typename T> class NBLA_API RMSpropGraves : public Solver {
public:
  RMSpropGraves(const Context &ctx, float lr, float decay, float momentum,
                float eps);
  virtual ~RMSpropGraves();
  virtual string name() { return "RMSpropGraves"; }

  virtual float learning_rate() { return lr_; }
  virtual void set_learning_rate(float lr) { lr_ = lr; }

protected:
  float lr_;       ///< learning rate
  float decay_;    ///< decay factor
  float momentum_; ///< momentum factor
  float eps_;      ///< small value

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
