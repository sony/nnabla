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

#ifndef __NBLA_SOLVER_AMSBOUND_HPP__
#define __NBLA_SOLVER_AMSBOUND_HPP__
#include <nbla/solver.hpp>
#include <nbla/solver_registry.hpp>

namespace nbla {

NBLA_REGISTER_SOLVER_HEADER(AMSBound, float, float, float, float, float, float,
                            bool);

/** AMSBound solver
@sa See the paper linked below for more details.
L. Luo, Y. Xiong, Y. Liu and X. Sun. Adaptive Gradient Methods with Dynamic
Bound of Learning Rate.
https://arxiv.org/abs/1902.09843

\ingroup SolverImplGrp
*/
template <typename T> class NBLA_API AMSBound : public Solver {
public:
  AMSBound(const Context &ctx, float alpha, float beta1, float beta2, float eps,
           float final_lr, float gamma, bool bias_correction);
  virtual ~AMSBound();
  virtual string name() { return "AMSBound"; }

  virtual float learning_rate() { return alpha_; }
  virtual void set_learning_rate(float lr) { alpha_ = lr; }

protected:
  // Settings.
  float alpha_; ///< \f$\alpha\f$
  float beta1_; ///< \f$\beta_1\f$
  float beta2_; ///< \f$\beta_2\f$
  float eps_;   ///< \f$\epsilon\f$
  float final_lr_;
  float gamma_;
  float init_alpha_;
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
