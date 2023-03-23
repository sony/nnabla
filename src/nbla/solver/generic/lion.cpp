// Copyright 2023 Sony Group Corporation.
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
#include <cassert>
#include <nbla/solver/clip_grad.hpp>
#include <nbla/solver/lion.hpp>
#include <nbla/solver/mixed_precision_training.hpp>
#include <nbla/solver/weight_decay.hpp>
#include <numeric>

namespace nbla {

NBLA_REGISTER_SOLVER_SOURCE(Lion, float, float, float);

template <typename T>
Lion<T>::Lion(const Context &ctx, float lr, float beta1, float beta2)
    : Solver(ctx, true), lr_(lr), beta1_(beta1), beta2_(beta2) {}

template <typename T> Lion<T>::~Lion() {}

template <typename T>
void Lion<T>::set_state_impl(const string &key, VariablePtr param) {
  auto shape = param->shape();
  auto m = make_shared<Variable>(shape);
  m->data()->zero();
  unordered_map<string, VariablePtr> pstate{{"m", m}};
  SolverState state{pstate, 0};
  this->states_.insert({key, state});
}

template <typename T> void Lion<T>::remove_state_impl(const string &key) {
  this->states_.erase(key);
}

template <typename T>
void Lion<T>::update_impl(const string &key, VariablePtr param) {
  Size_t size = param->size();
  VariablePtr m_var = this->states_.at(key).pstate["m"];
  const T *grad = param->get_grad_pointer<T>(this->ctx_);
  T *data = param->cast_data_and_get_pointer<T>(this->ctx_);
  T *m = m_var->cast_data_and_get_pointer<T>(this->ctx_);
  auto sign = [](const T &v) { return (v > T(0)) - (v < T(0)); };
  auto lerp = [](const T &x, const T &y, const float &t) {
    return x + t * (y - x);
  };
  for (Size_t i = 0; i < size; i++) {
    auto u = sign(lerp(grad[i], m[i], beta1_));
    m[i] = lerp(grad[i], m[i], beta2_);
    data[i] -= lr_ * (u + weight_decay_rate_ * data[i]);
  }
  auto &t = this->states_.at(key).t;
  t = std::min(t + 1, std::numeric_limits<uint32_t>::max() - 1);
}

NBLA_DEF_WEIGHT_DECAY(Lion, weight_decay_cpu);
NBLA_DEF_CLIP_GRAD_BY_NORM(Lion, clip_grad_by_norm_cpu);
NBLA_DEF_CHECK_INF_GRAD(Lion, check_inf_grad_cpu);
NBLA_DEF_CHECK_NAN_GRAD(Lion, check_nan_grad_cpu);
NBLA_DEF_CHECK_INF_OR_NAN_GRAD(Lion, check_inf_or_nan_grad_cpu);
NBLA_DEF_SCALE_GRAD(Lion, scale_grad_impl_cpu);
}
