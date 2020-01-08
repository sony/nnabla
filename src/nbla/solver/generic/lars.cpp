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
#include <cassert>
#include <nbla/solver/clip_grad.hpp>
#include <nbla/solver/lars.hpp>
#include <nbla/solver/mixed_precision_training.hpp>
#include <nbla/solver/weight_decay.hpp>
#include <numeric>

namespace nbla {
using std::shared_ptr;
using std::make_shared;

NBLA_REGISTER_SOLVER_SOURCE(Lars, float, float, float, float);

template <typename T>
Lars<T>::Lars(const Context &ctx, float lr, float momentum, float coefficient,
              float eps)
    : Solver(ctx), lr_(lr), momentum_(momentum), coefficient_(coefficient),
      eps_(eps), decay_rate_(0) {}

template <typename T> Lars<T>::~Lars() {}

template <typename T>
void Lars<T>::set_state_impl(const string &key, VariablePtr param) {
  auto shape = param->shape();
  auto m = make_shared<Variable>(shape);
  m->data()->zero();
  unordered_map<string, VariablePtr> pstate{{"v", m}};
  SolverState state{pstate, 0};
  this->states_.insert({key, state});
}

template <typename T> void Lars<T>::remove_state_impl(const string &key) {
  this->states_.erase(key);
}

template <typename T>
void Lars<T>::update_impl(const string &key, VariablePtr param) {
  Size_t size = param->size();
  VariablePtr v_var = this->states_.at(key).pstate["v"];
  T *v = v_var->cast_data_and_get_pointer<T>(this->ctx_);
  T *grad = param->cast_grad_and_get_pointer<T>(this->ctx_);
  T *data = param->cast_data_and_get_pointer<T>(this->ctx_);

  auto g_norm = std::sqrt(
      std::accumulate(grad, grad + size, T(0),
                      [](const T &acc, const T &g) { return acc + g * g; }));
  auto d_norm = std::sqrt(
      std::accumulate(data, data + size, T(0),
                      [](const T &acc, const T &d) { return acc + d * d; }));
  auto x = g_norm + this->decay_rate_ * d_norm;
  NBLA_CHECK(x >= 0, error_code::value,
             "local learning rate should be positive or zero.");
  if (x < this->eps_) {
    x += this->eps_;
  }
  auto lr = (d_norm < this->eps_) ? this->lr_
                                  : this->lr_ * this->coefficient_ * d_norm / x;

  weight_decay_cpu<T>(this->ctx_, param, this->decay_rate_);
  std::transform(grad, grad + size, v, v,
                 [this, lr](T g, T v) { return this->momentum_ * v + lr * g; });
  std::transform(v, v + size, data, data, [](T v, T x) { return x - v; });

  auto &t = this->states_.at(key).t;
  t = std::min(t + 1, std::numeric_limits<uint32_t>::max() - 1);
}

template <typename T>
void Lars<T>::weight_decay_impl(const string &key, VariablePtr param,
                                float decay_rate) {
  if (this->decay_rate_ == 0) {
    this->decay_rate_ = decay_rate;
  } else {
    /* decay_rate should be same between all layers */
    NBLA_CHECK(this->decay_rate_ == decay_rate, error_code::value,
               "decay_rate should be same between all layers");
  }
}
NBLA_DEF_CLIP_GRAD_BY_NORM(Lars, clip_grad_by_norm_cpu);
NBLA_DEF_CHECK_INF_GRAD(Lars, check_inf_grad_cpu);
NBLA_DEF_CHECK_NAN_GRAD(Lars, check_nan_grad_cpu);
NBLA_DEF_CHECK_INF_OR_NAN_GRAD(Lars, check_inf_or_nan_grad_cpu);
NBLA_DEF_SCALE_GRAD(Lars, scale_grad_impl_cpu);
}
