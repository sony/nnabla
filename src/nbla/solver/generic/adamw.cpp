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
#include <nbla/solver/adamw.hpp>
#include <nbla/solver/clip_grad.hpp>
#include <nbla/solver/mixed_precision_training.hpp>
#include <nbla/solver/weight_decay.hpp>

namespace nbla {
using std::shared_ptr;
using std::make_shared;

NBLA_REGISTER_SOLVER_SOURCE(AdamW, float, float, float, float, float);

template <typename T>
AdamW<T>::AdamW(const Context &ctx, float alpha, float beta1, float beta2,
                float eps, float wd)
    : Solver(ctx), alpha_(alpha), beta1_(beta1), beta2_(beta2), eps_(eps),
      wd_(wd), init_alpha_(alpha) {}

template <typename T> AdamW<T>::~AdamW() {}

template <typename T>
void AdamW<T>::set_state_impl(const string &key, VariablePtr param) {
  auto shape = param->shape();
  auto m = make_shared<Variable>(shape);
  auto v = make_shared<Variable>(shape);
  m->data()->zero();
  v->data()->zero();
  unordered_map<string, VariablePtr> pstate{{"mean", m}, {"var", v}};
  SolverState state{pstate, 0};
  states_.insert({key, state});
}
template <typename T> void AdamW<T>::remove_state_impl(const string &key) {
  states_.erase(key);
}

template <typename T>
void AdamW<T>::update_impl(const string &key, VariablePtr param) {
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
  T eta_t = alpha_ / init_alpha_;
  const T bias_correction =
      std::sqrt(1 - std::pow(beta2_, t)) / (1 - std::pow(beta1_, t));
  const T alpha_t = alpha_ * bias_correction;
  for (int s = 0; s < size; ++s) {
    // Updating running mean and var.
    m[s] = beta1_ * m[s] + (1 - beta1_) * g[s];
    v[s] = beta2_ * v[s] + (1 - beta2_) * g[s] * g[s];
    // Update parameters.
    theta[s] = theta[s] - alpha_t * m[s] / (std::sqrt(v[s]) + eps_) -
               eta_t * wd_ * theta[s];
  }
}

template <typename T>
void AdamW<T>::weight_decay_impl(const string &key, VariablePtr param,
                                 float decay_rate) {
  NBLA_CHECK(decay_rate == wd_, error_code::value,
             "decay rate should remain the same");
  weight_decay_cpu<T>(this->ctx_, param, decay_rate);
}

NBLA_DEF_CLIP_GRAD_BY_NORM(AdamW, clip_grad_by_norm_cpu);
NBLA_DEF_CHECK_INF_GRAD(AdamW, check_inf_grad_cpu);
NBLA_DEF_CHECK_NAN_GRAD(AdamW, check_nan_grad_cpu);
NBLA_DEF_CHECK_INF_OR_NAN_GRAD(AdamW, check_inf_or_nan_grad_cpu);
NBLA_DEF_SCALE_GRAD(AdamW, scale_grad_impl_cpu);
}
