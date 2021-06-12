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

#ifndef __NBLA_SOLVER_ADADELTA_HPP__
#define __NBLA_SOLVER_ADADELTA_HPP__
#include <nbla/solver.hpp>
#include <nbla/solver_registry.hpp>

namespace nbla {

NBLA_REGISTER_SOLVER_HEADER(Adadelta, float /*lr*/, float /*decay*/,
                            float /*eps*/);

/** Adadelta. This is defined as

\f[
g_t \leftarrow \Delta w_t\\
v_t \leftarrow - \frac{RMS \left[ v_t \right]_{t-1}}{RMS \left[ g
\right]_t}g_t\\
w_{t+1} \leftarrow w_t + \eta v_t
\f]

@param lr \f$\eta\f$ Learning rate.
@param decay \f$\gamma\f$ Decay rate.
@param eps \f$\epsilon\f$ Tiny factor for avoiding 0-division.

@sa See the paper linked below for more details.
Matthew D. Zeiler
ADADELTA: An Adaptive Learning Rate Method
https://arxiv.org/abs/1212.5701


\ingroup SolverImplGrp
*/
template <typename T> class NBLA_API Adadelta : public Solver {
public:
  Adadelta(const Context &ctx, float lr, float decay, float eps);
  virtual ~Adadelta();
  virtual string name() { return "Adadelta"; }

  virtual float learning_rate() { return lr_; }
  virtual void set_learning_rate(float lr) { lr_ = lr; }

protected:
  float lr_;    ///< learning rate
  float decay_; ///< decay factor
  float eps_;   ///< small value

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
