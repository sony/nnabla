// Copyright 2017,2018,2019,2020,2021 Sony Corporation.
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

#ifndef __NBLA_SOLVER_NESTOREV_HPP__
#define __NBLA_SOLVER_NESTOREV_HPP__
#include <nbla/solver.hpp>
#include <nbla/solver_registry.hpp>

namespace nbla {

NBLA_REGISTER_SOLVER_HEADER(Nesterov, float /*lr*/, float /*momentum*/);

/** Nesterov accelerated gradient. This is defined as

\f[
v_t \leftarrow \gamma v_{t-1} - \eta \Delta w_t \\
w_{t+1} \leftarrow w_t - \gamma v_{t-1} + \left(1 + \gamma \right) v_t
\f]

@param lr \f$\eta\f$ Learning rate.
@param momentum \f$\gamma\f$ Momentum

@sa See the paper linked below for more details.
Yurii Nesterov
A method for unconstrained convex minimization problem with the rate of
convergence o(1/k2)

\ingroup SolverImplGrp
*/
template <typename T> class NBLA_API Nesterov : public Solver {
public:
  Nesterov(const Context &ctx, float lr, float momentum);
  virtual ~Nesterov();
  virtual string name() { return "Nesterov"; }

  virtual float learning_rate() { return lr_; }
  virtual void set_learning_rate(float lr) { lr_ = lr; }

protected:
  float lr_; ///< learning rate
  float momentum_;

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
