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
#include <nbla/solver/clip_grad.hpp>
#include <nbla/solver/mixed_precision_training.hpp>
#include <nbla/solver/rmsprop_graves.hpp>
#include <nbla/solver/weight_decay.hpp>

namespace nbla {
using std::make_shared;
using std::shared_ptr;

NBLA_REGISTER_SOLVER_SOURCE(RMSpropGraves, float, float, float, float);

template <typename T>
RMSpropGraves<T>::RMSpropGraves(const Context &ctx, float lr, float decay,
                                float momentum, float eps)
    : Solver(ctx), lr_(lr), decay_(decay), momentum_(momentum), eps_(eps) {}

template <typename T> RMSpropGraves<T>::~RMSpropGraves() {}

template <typename T>
void RMSpropGraves<T>::set_state_impl(const string &key, VariablePtr param) {
  auto shape = param->shape();
  auto n = make_shared<Variable>(shape);
  auto g = make_shared<Variable>(shape);
  auto d = make_shared<Variable>(shape);
  n->data()->zero();
  g->data()->zero();
  d->data()->zero();
  unordered_map<string, VariablePtr> pstate{{"n", n}, {"g", g}, {"d", d}};
  SolverState state{pstate, 0};
  states_.insert({key, state});
}

template <typename T>
void RMSpropGraves<T>::remove_state_impl(const string &key) {
  states_.erase(key);
}

template <typename T>
void RMSpropGraves<T>::update_impl(const string &key, VariablePtr param) {
  Size_t size = param->size();
  auto &state = states_.at(key);
  VariablePtr s1 = state.pstate["n"];
  VariablePtr s2 = state.pstate["g"];
  VariablePtr s3 = state.pstate["d"];
  T *n = s1->cast_data_and_get_pointer<T>(this->ctx_);
  T *g = s2->cast_data_and_get_pointer<T>(this->ctx_);
  T *d = s3->cast_data_and_get_pointer<T>(this->ctx_);
  const T *grad = param->get_grad_pointer<T>(this->ctx_);
  T *data = param->cast_data_and_get_pointer<T>(this->ctx_);
  for (int s = 0; s < size; ++s) {
    n[s] = decay_ * n[s] + (1 - decay_) * grad[s] * grad[s];
    g[s] = decay_ * g[s] + (1 - decay_) * grad[s];
    d[s] = (momentum_)*d[s] -
           lr_ * grad[s] / (std::sqrt(n[s] - g[s] * g[s] + eps_));
    data[s] += d[s];
  }
  auto &t = state.t;
  t = std::min(t + 1, std::numeric_limits<uint32_t>::max() - 1);
}

NBLA_DEF_WEIGHT_DECAY(RMSpropGraves, weight_decay_cpu);
NBLA_DEF_CLIP_GRAD_BY_NORM(RMSpropGraves, clip_grad_by_norm_cpu);
NBLA_DEF_CHECK_INF_GRAD(RMSpropGraves, check_inf_grad_cpu);
NBLA_DEF_CHECK_NAN_GRAD(RMSpropGraves, check_nan_grad_cpu);
NBLA_DEF_CHECK_INF_OR_NAN_GRAD(RMSpropGraves, check_inf_or_nan_grad_cpu);
NBLA_DEF_SCALE_GRAD(RMSpropGraves, scale_grad_impl_cpu);
}
