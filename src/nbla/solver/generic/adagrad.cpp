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
#include <nbla/solver/adagrad.hpp>
#include <nbla/solver/clip_grad.hpp>
#include <nbla/solver/mixed_precision_training.hpp>
#include <nbla/solver/weight_decay.hpp>

namespace nbla {
using std::shared_ptr;
using std::make_shared;

NBLA_REGISTER_SOLVER_SOURCE(Adagrad, float, float);

template <typename T>
Adagrad<T>::Adagrad(const Context &ctx, float lr, float eps)
    : Solver(ctx), lr_(lr), eps_(eps) {}

template <typename T> Adagrad<T>::~Adagrad() {}

template <typename T>
void Adagrad<T>::set_state_impl(const string &key, VariablePtr param) {
  auto v = make_shared<Variable>(param->shape());
  v->data()->zero();
  unordered_map<string, VariablePtr> pstate{{"v", v}};
  SolverState state{pstate, 0};
  states_.insert({key, state});
}
template <typename T> void Adagrad<T>::remove_state_impl(const string &key) {
  states_.erase(key);
}

template <typename T>
void Adagrad<T>::update_impl(const string &key, VariablePtr param) {
  Size_t size = param->size();
  auto &state = states_.at(key);
  VariablePtr g_ = state.pstate["v"];
  auto &t = state.t;
  T *g = g_->cast_data_and_get_pointer<T>(this->ctx_);
  const T *grad = param->get_grad_pointer<T>(this->ctx_);
  T *data = param->cast_data_and_get_pointer<T>(this->ctx_);
  t = std::min(t + 1, std::numeric_limits<uint32_t>::max() - 1);
  for (int s = 0; s < size; ++s) {
    g[s] += grad[s] * grad[s];
    data[s] -= lr_ * grad[s] / (std::sqrt(g[s]) + eps_);
  }
}

NBLA_DEF_WEIGHT_DECAY(Adagrad, weight_decay_cpu);
NBLA_DEF_CLIP_GRAD_BY_NORM(Adagrad, clip_grad_by_norm_cpu);
NBLA_DEF_CHECK_INF_GRAD(Adagrad, check_inf_grad_cpu);
NBLA_DEF_CHECK_NAN_GRAD(Adagrad, check_nan_grad_cpu);
NBLA_DEF_CHECK_INF_OR_NAN_GRAD(Adagrad, check_inf_or_nan_grad_cpu);
NBLA_DEF_SCALE_GRAD(Adagrad, scale_grad_impl_cpu);
}
