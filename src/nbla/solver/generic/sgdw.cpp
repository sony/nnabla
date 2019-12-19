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
#include <nbla/solver/sgdw.hpp>
#include <nbla/solver/weight_decay.hpp>

namespace nbla {
using std::shared_ptr;
using std::make_shared;

NBLA_REGISTER_SOLVER_SOURCE(SgdW, float, float, float);

template <typename T>
SgdW<T>::SgdW(const Context &ctx, float lr, float momentum, float wd)
    : Solver(ctx), lr_(lr), momentum_(momentum), wd_(wd), init_lr_(lr) {}

template <typename T> SgdW<T>::~SgdW() {}

template <typename T>
void SgdW<T>::set_state_impl(const string &key, VariablePtr param) {
  auto shape = param->shape();
  auto m = make_shared<Variable>(shape);
  m->data()->zero();
  unordered_map<string, VariablePtr> pstate{{"m", m}};
  SolverState state{pstate, 0};
  states_.insert({key, state});
}
template <typename T> void SgdW<T>::remove_state_impl(const string &key) {
  states_.erase(key);
}

template <typename T>
void SgdW<T>::update_impl(const string &key, VariablePtr param) {
  Size_t size = param->size();
  auto &state = states_.at(key);
  VariablePtr v_ = state.pstate["m"];
  T *v = v_->cast_data_and_get_pointer<T>(this->ctx_);
  const T *grad = param->get_grad_pointer<T>(this->ctx_);
  T *data = param->cast_data_and_get_pointer<T>(this->ctx_);
  std::transform(grad, grad + size, v, v, [this](T g, T v) {
    return momentum_ * v + lr_ * g - (lr_ / init_lr_) * wd_ * v;
  });
  std::transform(v, v + size, data, data, [this](T v, T x) { return x - v; });
  auto &t = state.t;
  t = std::min(t + 1, std::numeric_limits<uint32_t>::max() - 1);
}

template <typename T>
void SgdW<T>::weight_decay_impl(const string &key, VariablePtr param,
                                float decay_rate) {
  NBLA_CHECK(decay_rate == wd_, error_code::value,
             "decay rate should remain the same");
  weight_decay_cpu<T>(this->ctx_, param, decay_rate);
}

NBLA_DEF_CLIP_GRAD_BY_NORM(SgdW, clip_grad_by_norm_cpu);
NBLA_DEF_CHECK_INF_GRAD(SgdW, check_inf_grad_cpu);
NBLA_DEF_CHECK_NAN_GRAD(SgdW, check_nan_grad_cpu);
NBLA_DEF_CHECK_INF_OR_NAN_GRAD(SgdW, check_inf_or_nan_grad_cpu);
NBLA_DEF_SCALE_GRAD(SgdW, scale_grad_impl_cpu);
}
