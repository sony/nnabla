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
#include <nbla/solver/amsgrad.hpp>
#include <nbla/solver/weight_decay.hpp>

namespace nbla {
using std::shared_ptr;
using std::make_shared;

NBLA_REGISTER_SOLVER_SOURCE(AMSGRAD, float, float, float, float, bool);

template <typename T>
AMSGRAD<T>::AMSGRAD(const Context &ctx, float alpha, float beta1, float beta2,
                    float eps, bool bias_correction)
    : Solver(ctx), alpha_(alpha), beta1_(beta1), beta2_(beta2), eps_(eps),
      bias_correction_(bias_correction) {}

template <typename T> AMSGRAD<T>::~AMSGRAD() {}

template <typename T>
void AMSGRAD<T>::set_state_impl(const string &key, VariablePtr param) {
  auto shape = param->shape();
  auto m = make_shared<Variable>(shape);
  auto v = make_shared<Variable>(shape);
  auto v_hat = make_shared<Variable>(shape);
  m->data()->zero();
  v->data()->zero();
  v_hat->data()->zero();
  state_.insert({key, State{m, v, v_hat, 0}});
}
template <typename T> void AMSGRAD<T>::remove_state_impl(const string &key) {
  state_.erase(key);
}

template <typename T>
void AMSGRAD<T>::update_impl(const string &key, VariablePtr param) {
  Size_t size = param->size();
  auto &state = state_.at(key);
  auto &t = state.t;
  const T *g = param->get_grad_pointer<T>(this->ctx_);
  VariablePtr s1 = state.mean;
  VariablePtr s2 = state.var;
  VariablePtr s3 = state.var_hat;
  T *m = s1->cast_data_and_get_pointer<T>(this->ctx_);
  T *v = s2->cast_data_and_get_pointer<T>(this->ctx_);
  T *v_hat = s3->cast_data_and_get_pointer<T>(this->ctx_);
  T *theta = param->cast_data_and_get_pointer<T>(this->ctx_);
  t = std::min(t + 1, std::numeric_limits<int>::max());
  const T bias_correction =
      std::sqrt(1 - std::pow(beta2_, t)) / (1 - std::pow(beta1_, t));
  const T alpha_t = alpha_ * (bias_correction_ ? bias_correction : 1);
  for (int s = 0; s < size; ++s) {
    // Updating running mean and var.
    m[s] = beta1_ * m[s] + (1 - beta1_) * g[s];
    v[s] = beta2_ * v[s] + (1 - beta2_) * g[s] * g[s];
    v_hat[s] = std::max(v_hat[s], v[s]);
    // Update parameters.
    theta[s] = theta[s] - alpha_t * m[s] / (std::sqrt(v_hat[s]) + eps_);
  }
}

NBLA_DEF_WEIGHT_DECAY(AMSGRAD, weight_decay_cpu);
}
