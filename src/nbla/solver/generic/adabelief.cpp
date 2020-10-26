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

#include <algorithm>
#include <cmath>
#include <limits>
#include <nbla/solver/adabelief.hpp>
#include <nbla/solver/clip_grad.hpp>
#include <nbla/solver/mixed_precision_training.hpp>
#include <nbla/solver/weight_decay.hpp>

namespace nbla {
using std::make_shared;
using std::shared_ptr;

NBLA_REGISTER_SOLVER_SOURCE(AdaBelief, float, float, float, float, float, bool,
                            bool, bool, bool);

template <typename T>
AdaBelief<T>::AdaBelief(const Context &ctx, float alpha, float beta1,
                        float beta2, float eps, float wd, bool amsgrad,
                        bool weight_decouple, bool fixed_decay, bool rectify)
    : Solver(ctx), alpha_(alpha), beta1_(beta1), beta2_(beta2), eps_(eps),
      wd_(wd), amsgrad_(amsgrad), weight_decouple_(weight_decouple),
      fixed_decay_(fixed_decay), rectify_(rectify) {}

template <typename T> AdaBelief<T>::~AdaBelief() {}

template <typename T>
void AdaBelief<T>::set_state_impl(const string &key, VariablePtr param) {
  auto shape = param->shape();
  auto m = make_shared<Variable>(shape);
  auto s = make_shared<Variable>(shape);
  auto max_s = make_shared<Variable>(shape);
  m->data()->zero();
  s->data()->zero();
  max_s->data()->zero();
  unordered_map<string, VariablePtr> pstate{
      {"mean", m}, {"sqr_var", s}, {"max_sqr_var", max_s}};
  SolverState state{pstate, 0};
  states_.insert({key, state});
}
template <typename T> void AdaBelief<T>::remove_state_impl(const string &key) {
  states_.erase(key);
}

template <typename T>
void AdaBelief<T>::update_impl(const string &key, VariablePtr param) {
  Size_t size = param->size();
  auto &state = states_.at(key);
  auto &t = state.t;
  const T *g = param->get_grad_pointer<T>(this->ctx_);
  VariablePtr s1 = state.pstate["mean"];
  VariablePtr s2 = state.pstate["sqr_var"];
  VariablePtr s3 = state.pstate["max_sqr_var"];
  T *m = s1->cast_data_and_get_pointer<T>(this->ctx_);
  T *s = s2->cast_data_and_get_pointer<T>(this->ctx_);
  T *max_s = s3->cast_data_and_get_pointer<T>(this->ctx_);
  T *theta = param->cast_data_and_get_pointer<T>(this->ctx_);
  t = std::min(t + 1, std::numeric_limits<uint32_t>::max() - 1);

  const T beta1_t = std::pow(beta1_, t);
  const T beta2_t = std::pow(beta2_, t);
  const T bias_correction = std::sqrt(1 - beta2_t) / (1 - beta1_t);
  T alpha_t = alpha_ * bias_correction;
  bool update_in_sgd_style = false;

  if (rectify_) {
    const T rho_inf = 2 / (1 - beta2_) - 1;
    const T rho_t =
        rho_inf - 2 * t * std::pow(beta2_, t) / (1 - std::pow(beta2_, t));

    if (rho_t > 4.0) {
      const T rt = std::sqrt((rho_t - 4) * (rho_t - 2) * rho_inf /
                             (rho_inf - 4) / (rho_inf - 2) / rho_t);
      alpha_t = rt * alpha_t;
    } else {
      alpha_t = alpha_;
      update_in_sgd_style = true;
    }
  }

  for (int i = 0; i < size; ++i) {
    m[i] = beta1_ * m[i] + (1 - beta1_) * g[i];
    s[i] = beta2_ * s[i] + (1 - beta2_) * std::pow(g[i] - m[i], 2);

    T denom = std::sqrt(s[i] + eps_);
    if (amsgrad_) {
      max_s[i] = std::max(s[i], max_s[i]);
      denom = std::sqrt(max_s[i] + eps_);
    }

    if (weight_decouple_) {
      if (fixed_decay_) {
        theta[i] = theta[i] - theta[i] * wd_;
      } else {
        theta[i] = theta[i] - theta[i] * wd_ * alpha_;
      }
    }

    if (update_in_sgd_style) {
      theta[i] = theta[i] - alpha_t * m[i];
    } else {
      theta[i] = theta[i] - alpha_t * m[i] / denom;
    }
  }
}

NBLA_DEF_WEIGHT_DECAY(AdaBelief, weight_decay_cpu);
NBLA_DEF_CLIP_GRAD_BY_NORM(AdaBelief, clip_grad_by_norm_cpu);
NBLA_DEF_CHECK_INF_GRAD(AdaBelief, check_inf_grad_cpu);
NBLA_DEF_CHECK_NAN_GRAD(AdaBelief, check_nan_grad_cpu);
NBLA_DEF_CHECK_INF_OR_NAN_GRAD(AdaBelief, check_inf_or_nan_grad_cpu);
NBLA_DEF_SCALE_GRAD(AdaBelief, scale_grad_impl_cpu);
} // namespace nbla
