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
  m->data()->zero();
  s->data()->zero();
  unordered_map<string, VariablePtr> pstate{{"mean", m}, {"var", s}};
  if (amsgrad_) {
    auto s_max = make_shared<Variable>(shape);
    s_max->data()->zero();
    pstate["s_max"] = s_max;
  }
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
  VariablePtr s2 = state.pstate["var"];

  t = std::min(t + 1, std::numeric_limits<uint32_t>::max() - 1);
  const T beta1_t = std::pow(beta1_, t);
  const T beta2_t = std::pow(beta2_, t);
  const T bias_correction1 = (1.0 - beta1_t);
  const T bias_correction2 = std::sqrt(1.0 - beta2_t);
  float r_t = 1.0;
  float rho_t = 0.0;

  if (rectify_) {
    auto rho_inf = 2.0 / (1.0 - beta2_) - 1.0;
    rho_t = rho_inf - 2.0 * t * beta2_t / (1.0 - beta2_t);
    auto r_t_numerator = (rho_t - 4.0) * (rho_t - 2.0) * rho_inf;
    auto r_t_denominator = (rho_inf - 4.0) * (rho_inf - 2.0) * rho_t;
    r_t = std::sqrt(r_t_numerator / r_t_denominator);
  }

  const bool sgd_update = (rectify_ && rho_t <= 4.0);
  const float alpha_t = sgd_update ? alpha_ : alpha_ * r_t / bias_correction1;
  const float decay_ratio = fixed_decay_ ? wd_ : wd_ * alpha_;

  T *m = s1->cast_data_and_get_pointer<T>(this->ctx_);
  T *s = s2->cast_data_and_get_pointer<T>(this->ctx_);
  T *s_max = nullptr;
  if (amsgrad_) {
    VariablePtr s3 = state.pstate["s_max"];
    s_max = s3->cast_data_and_get_pointer<T>(this->ctx_);
  }
  T *theta = param->cast_data_and_get_pointer<T>(this->ctx_);

  for (int i = 0; i < size; ++i) {
    // Updating running mean and var.
    m[i] = beta1_ * m[i] + (1 - beta1_) * g[i];
    s[i] = beta2_ * s[i] + (1 - beta2_) * std::pow(g[i] - m[i], 2);

    if (weight_decouple_) {
      theta[i] = theta[i] - theta[i] * decay_ratio;
    }

    if (amsgrad_) {
      s_max[i] = std::max(s_max[i], s[i]);
      s_max[i] += eps_;
    } else {
      s[i] += eps_;
    }
    // Update parameters.
    if (sgd_update) {
      theta[i] = theta[i] - alpha_t * m[i];
    } else {
      float s_t = amsgrad_ ? s_max[i] : s[i];
      float denominator = std::sqrt(s_t) / bias_correction2;
      theta[i] = theta[i] - alpha_t * m[i] / (denominator + eps_);
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
