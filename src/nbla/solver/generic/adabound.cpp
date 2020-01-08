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
#include <nbla/solver/adabound.hpp>
#include <nbla/solver/clip_grad.hpp>
#include <nbla/solver/mixed_precision_training.hpp>
#include <nbla/solver/weight_decay.hpp>

namespace nbla {
using std::shared_ptr;
using std::make_shared;

NBLA_REGISTER_SOLVER_SOURCE(AdaBound, float, float, float, float, float, float);

template <typename T>
AdaBound<T>::AdaBound(const Context &ctx, float alpha, float beta1, float beta2,
                      float eps, float final_lr, float gamma)
    : Solver(ctx), alpha_(alpha), beta1_(beta1), beta2_(beta2), eps_(eps),
      final_lr_(final_lr), gamma_(gamma), init_alpha_(alpha_) {}

template <typename T> AdaBound<T>::~AdaBound() {}

template <typename T>
void AdaBound<T>::set_state_impl(const string &key, VariablePtr param) {
  auto shape = param->shape();
  auto m = make_shared<Variable>(shape);
  auto v = make_shared<Variable>(shape);
  m->data()->zero();
  v->data()->zero();
  unordered_map<string, VariablePtr> pstate{{"mean", m}, {"var", v}};
  SolverState state{pstate, 0};
  states_.insert({key, state});
}
template <typename T> void AdaBound<T>::remove_state_impl(const string &key) {
  states_.erase(key);
}

template <typename T>
void AdaBound<T>::update_impl(const string &key, VariablePtr param) {
  Size_t size = param->size();
  auto &state = states_.at(key);
  auto &t = state.t;
  const T *g = param->get_grad_pointer<T>(this->ctx_);
  VariablePtr s1 = state.pstate["mean"];
  VariablePtr s2 = state.pstate["var"];
  T *m = s1->cast_data_and_get_pointer<T>(this->ctx_);
  T *v = s2->cast_data_and_get_pointer<T>(this->ctx_);
  T *theta = param->cast_data_and_get_pointer<T>(this->ctx_);
  t = std::min(t + 1, std::numeric_limits<uint32_t>::max() - 1);
  const T bias_correction =
      std::sqrt(1 - std::pow(beta2_, t)) / (1 - std::pow(beta1_, t));
  T alpha_t = alpha_ * bias_correction;
  T final_lr = final_lr_ * (alpha_ / init_alpha_);
  for (int s = 0; s < size; ++s) {
    // Updating running mean and var.
    m[s] = beta1_ * m[s] + (1 - beta1_) * g[s];
    v[s] = beta2_ * v[s] + (1 - beta2_) * g[s] * g[s];
    T lower_bound = final_lr * (1 - 1 / (gamma_ * t + 1));
    T upper_bound = final_lr * (1 + 1 / (gamma_ * t));
    T denom = std::sqrt(v[s]) + eps_;
    T eta = std::min(upper_bound, std::max(alpha_t / denom, lower_bound));
    // Update parameters.
    theta[s] = theta[s] - eta * m[s];
  }
}

NBLA_DEF_WEIGHT_DECAY(AdaBound, weight_decay_cpu);
NBLA_DEF_CLIP_GRAD_BY_NORM(AdaBound, clip_grad_by_norm_cpu);
NBLA_DEF_CHECK_INF_GRAD(AdaBound, check_inf_grad_cpu);
NBLA_DEF_CHECK_NAN_GRAD(AdaBound, check_nan_grad_cpu);
NBLA_DEF_CHECK_INF_OR_NAN_GRAD(AdaBound, check_inf_or_nan_grad_cpu);
NBLA_DEF_SCALE_GRAD(AdaBound, scale_grad_impl_cpu);
}
