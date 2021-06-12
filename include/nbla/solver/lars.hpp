// Copyright 2019,2020,2021 Sony Corporation.
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

#ifndef __NBLA_SOLVER_LARS_HPP__
#define __NBLA_SOLVER_LARS_HPP__
#include <nbla/solver.hpp>
#include <nbla/solver_registry.hpp>

namespace nbla {

NBLA_REGISTER_SOLVER_HEADER(Lars, float /*lr*/, float /*momentum*/,
                            float /* coefficient */, float /* eps */);

/** LARS solver defined as
\f[
\lambda \leftarrow \eta
\frac{\| w_t \|}{\| \Delta w_t + \beta w_t \|} \\
v_{t+1} \leftarrow m v_t + \gamma \lambda
(\Delta w_t + \beta w_t) \\
w_{t+1} \leftarrow w_t - v_{t+1}
\f]

@param lr \f$\gamma\f$ Learning rate.
@param momentum \f$m\f$ Momentum.
@param coefficient \f$\eta\f$ Coefficient of the local learning rate.
@param eps \f$\epsilon\f$ Tiny factor for avoiding 0-division.

@sa See the paper linked below for more details.
Yang Youity, Igor Gitmann, and Boris Ginsburg:
Large Batch Training of Convolutional Networks
https://arxiv.org/pdf/1708.03888

\ingroup SolverImplGrp
*/
template <typename T> class NBLA_API Lars : public Solver {
public:
  Lars(const Context &ctx, float lr, float momentum, float coefficient,
       float eps);
  virtual ~Lars();
  virtual string name() { return "Lars"; }

  virtual float learning_rate() { return lr_; }
  virtual void set_learning_rate(float lr) { lr_ = lr; }

protected:
  float lr_; ///< learning rate
  float momentum_;
  float coefficient_;
  float eps_;
  float decay_rate_;

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
}
#endif
