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
#include <nbla/solver/mixed_precision_training.hpp>
#include <nbla/solver/rmsprop.hpp>
#include <nbla/solver/weight_decay.hpp>

namespace nbla {
using std::shared_ptr;
using std::make_shared;

NBLA_REGISTER_SOLVER_SOURCE(RMSprop, float, float, float);

template <typename T>
RMSprop<T>::RMSprop(const Context &ctx, float lr, float decay, float eps)
    : Solver(ctx), lr_(lr), decay_(decay), eps_(eps) {}

template <typename T> RMSprop<T>::~RMSprop() {}

template <typename T>
void RMSprop<T>::set_state_impl(const string &key, VariablePtr param) {
  auto shape = param->shape();
  auto v = make_shared<Variable>(shape);
  v->data()->zero();
  state_.insert({key, v});
}
template <typename T> void RMSprop<T>::remove_state_impl(const string &key) {
  state_.erase(key);
}

template <typename T>
void RMSprop<T>::update_impl(const string &key, VariablePtr param) {
  Size_t size = param->size();
  VariablePtr state = state_.at(key);
  T *e_sqr_grad = state->cast_data_and_get_pointer<T>(this->ctx_);
  const T *grad = param->get_grad_pointer<T>(this->ctx_);
  T *data = param->cast_data_and_get_pointer<T>(this->ctx_);
  for (int s = 0; s < size; ++s) {
    e_sqr_grad[s] = e_sqr_grad[s] * decay_ + grad[s] * grad[s] * (1 - decay_);
    data[s] -= lr_ / (std::sqrt(e_sqr_grad[s]) + eps_) * grad[s];
  }
}

NBLA_DEF_WEIGHT_DECAY(RMSprop, weight_decay_cpu);
NBLA_DEF_CHECK_INF_GRAD(RMSprop, check_inf_grad_cpu);
NBLA_DEF_CHECK_NAN_GRAD(RMSprop, check_nan_grad_cpu);
NBLA_DEF_CHECK_INF_OR_NAN_GRAD(RMSprop, check_inf_or_nan_grad_cpu);
NBLA_DEF_SCALE_GRAD(RMSprop, scale_grad_impl_cpu);
}
