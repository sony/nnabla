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

#ifndef __NBLA_SOLVER_ADAGRAD_HPP__
#define __NBLA_SOLVER_ADAGRAD_HPP__
#include <nbla/solver.hpp>
#include <nbla/solver_registry.hpp>

namespace nbla {

NBLA_REGISTER_SOLVER_HEADER(Adagrad, float /*lr*/, float /*eps*/);

/** Adagrad. This is defined as

\f[
g_t \leftarrow \Delta w_t\\
G_t \leftarrow G_{t-1} + g_t^2\\
w_{t+1} \leftarrow w_t - \frac{\eta}{\sqrt{G_t} + \epsilon} g_t\\
\f]

@param lr \f$\eta\f$ Learning rate.
@param eps \f$\epsilon\f$ Tiny factor for avoiding 0-division.

@sa See the paper linked below for more details.
John Duchi, Elad Hazan and Yoram Singer (2011)
Adaptive Subgradient Methods for Online Learning and Stochastic Optimization
http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf


\ingroup SolverImplGrp
*/
template <typename T> class NBLA_API Adagrad : public Solver {
public:
  Adagrad(const Context &ctx, float lr, float eps);
  virtual ~Adagrad();
  virtual string name() { return "Adagrad"; }

  virtual float learning_rate() { return lr_; }
  virtual void set_learning_rate(float lr) { lr_ = lr; }

protected:
  float lr_;  ///< learning rate
  float eps_; ///< small value

  unordered_map<string, VariablePtr> state_;

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
