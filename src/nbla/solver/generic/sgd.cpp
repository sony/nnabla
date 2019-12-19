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
#include <nbla/solver/clip_grad.hpp>
#include <nbla/solver/mixed_precision_training.hpp>
#include <nbla/solver/sgd.hpp>
#include <nbla/solver/weight_decay.hpp>

namespace nbla {
using std::shared_ptr;
using std::make_shared;

NBLA_REGISTER_SOLVER_SOURCE(Sgd, float);

template <typename T>
Sgd<T>::Sgd(const Context &ctx, float lr) : Solver(ctx), lr_(lr) {}

template <typename T> Sgd<T>::~Sgd() {}

template <typename T>
void Sgd<T>::set_state_impl(const string &key, VariablePtr param) {
  unordered_map<string, VariablePtr> pstate;
  SolverState state{pstate, 0};
  states_.insert({key, state});
}
template <typename T> void Sgd<T>::remove_state_impl(const string &key) {}

template <typename T>
void Sgd<T>::update_impl(const string &key, VariablePtr param) {
  Size_t size = param->size();

  const T *grad = param->get_grad_pointer<T>(this->ctx_);
  T *data = param->cast_data_and_get_pointer<T>(this->ctx_);
  std::transform(grad, grad + size, data, data,
                 [this](T g, T x) { return x - lr_ * g; });
  auto &state = states_.at(key);
  auto &t = state.t;
  t = std::min(t + 1, std::numeric_limits<uint32_t>::max() - 1);
}

NBLA_DEF_WEIGHT_DECAY(Sgd, weight_decay_cpu);
NBLA_DEF_CLIP_GRAD_BY_NORM(Sgd, clip_grad_by_norm_cpu);
NBLA_DEF_CHECK_INF_GRAD(Sgd, check_inf_grad_cpu);
NBLA_DEF_CHECK_NAN_GRAD(Sgd, check_nan_grad_cpu);
NBLA_DEF_CHECK_INF_OR_NAN_GRAD(Sgd, check_inf_or_nan_grad_cpu);
NBLA_DEF_SCALE_GRAD(Sgd, scale_grad_impl_cpu);
}
