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
#include <nbla/solver/adadelta.hpp>
#include <nbla/solver/clip_grad.hpp>
#include <nbla/solver/mixed_precision_training.hpp>
#include <nbla/solver/weight_decay.hpp>

namespace nbla {
using std::shared_ptr;
using std::make_shared;

NBLA_REGISTER_SOLVER_SOURCE(Adadelta, float, float, float);

template <typename T>
Adadelta<T>::Adadelta(const Context &ctx, float lr, float decay, float eps)
    : Solver(ctx), lr_(lr), decay_(decay), eps_(eps) {}

template <typename T> Adadelta<T>::~Adadelta() {}

template <typename T>
void Adadelta<T>::set_state_impl(const string &key, VariablePtr param) {
  auto shape = param->shape();
  auto e_sqr_grad = make_shared<Variable>(shape);
  auto e_sqr_delta = make_shared<Variable>(shape);
  e_sqr_grad->data()->zero();
  e_sqr_delta->data()->zero();
  unordered_map<string, VariablePtr> pstate{{"e_sqr_grad", e_sqr_grad},
                                            {"e_sqr_delta", e_sqr_delta}};
  SolverState state{pstate, 0};
  states_.insert({key, state});
}
template <typename T> void Adadelta<T>::remove_state_impl(const string &key) {
  states_.erase(key);
}

template <typename T>
void Adadelta<T>::update_impl(const string &key, VariablePtr param) {
  Size_t size = param->size();
  auto &state = states_.at(key);
  VariablePtr e1 = state.pstate["e_sqr_grad"];
  VariablePtr e2 = state.pstate["e_sqr_delta"];
  T *e_sqr_grad = e1->cast_data_and_get_pointer<T>(this->ctx_);
  T *e_sqr_delta = e2->cast_data_and_get_pointer<T>(this->ctx_);
  const T *grad = param->get_grad_pointer<T>(this->ctx_);
  T *data = param->cast_data_and_get_pointer<T>(this->ctx_);

  for (int s = 0; s < size; ++s) {
    e_sqr_grad[s] = e_sqr_grad[s] * decay_ + grad[s] * grad[s] * (1 - decay_);
    const T delta =
        std::sqrt((e_sqr_delta[s] + eps_) / (e_sqr_grad[s] + eps_)) * grad[s];
    e_sqr_delta[s] = e_sqr_delta[s] * decay_ + delta * delta * (1 - decay_);
    data[s] -= lr_ * delta;
  }
  auto &t = state.t;
  t = std::min(t + 1, std::numeric_limits<uint32_t>::max() - 1);
}

NBLA_DEF_WEIGHT_DECAY(Adadelta, weight_decay_cpu);
NBLA_DEF_CLIP_GRAD_BY_NORM(Adadelta, clip_grad_by_norm_cpu);
NBLA_DEF_CHECK_INF_GRAD(Adadelta, check_inf_grad_cpu);
NBLA_DEF_CHECK_NAN_GRAD(Adadelta, check_nan_grad_cpu);
NBLA_DEF_CHECK_INF_OR_NAN_GRAD(Adadelta, check_inf_or_nan_grad_cpu);
NBLA_DEF_SCALE_GRAD(Adadelta, scale_grad_impl_cpu);
}
