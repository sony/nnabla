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
  state_.insert({key, v});
}
template <typename T> void Adagrad<T>::remove_state_impl(const string &key) {
  state_.erase(key);
}

template <typename T>
void Adagrad<T>::update_impl(const string &key, VariablePtr param) {
  Size_t size = param->size();
  VariablePtr g_ = state_.at(key);
  T *g = g_->cast_data_and_get_pointer<T>(this->ctx_);
  const T *grad = param->get_grad_pointer<T>(this->ctx_);
  T *data = param->cast_data_and_get_pointer<T>(this->ctx_);
  for (int s = 0; s < size; ++s) {
    g[s] += grad[s] * grad[s];
    data[s] -= lr_ * grad[s] / (std::sqrt(g[s]) + eps_);
  }
}

NBLA_DEF_WEIGHT_DECAY(Adagrad, weight_decay_cpu);
}
