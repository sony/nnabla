// Copyright 2022 Sony Group Corporation.
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
#include <nbla/solver/clip_grad.hpp>
#include <nbla/solver/lamb.hpp>
#include <nbla/solver/mixed_precision_training.hpp>
#include <nbla/solver/weight_decay.hpp>

namespace nbla {
using std::shared_ptr;
using std::make_shared;
using std::pow;

NBLA_REGISTER_SOLVER_SOURCE(Lamb, float /* eta */, float /* beta1 */,
                            float /* beta2 */, float /* gamma_l */,
                            float /* gamma_u */, float /* eps */,
                            bool /* bias_correction */
                            );

template <typename T>
Lamb<T>::Lamb(const Context &ctx, float eta, float beta1, float beta2,
              float gamma_l, float gamma_u, float eps, bool bias_correction)
    : Solver(ctx, true), eta_(eta), beta1_(beta1), beta2_(beta2),
      gamma_l_(gamma_l), gamma_u_(gamma_u), eps_(eps),
      bias_correction_(bias_correction) {}

template <typename T> Lamb<T>::~Lamb() {}

template <typename T>
void Lamb<T>::set_state_impl(const string &key, VariablePtr param) {
  auto shape = param->shape();
  auto m = make_shared<Variable>(shape);
  auto v = make_shared<Variable>(shape);
  m->data()->zero();
  v->data()->zero();
  unordered_map<string, VariablePtr> pstate{{"mean", m}, {"var", v}};
  SolverState state{pstate, 0};
  states_.insert({key, state});
}
template <typename T> void Lamb<T>::remove_state_impl(const string &key) {
  states_.erase(key);
}

template <typename T>
void Lamb<T>::update_impl(const string &key, VariablePtr param) {
  auto dtype = get_dtype<T>();

  Size_t size = param->size();
  auto &state = states_.at(key);
  auto &t = state.t;
  t = std::min(t + 1, std::numeric_limits<uint32_t>::max() - 1);

  VariablePtr s1 = state.pstate["mean"];
  VariablePtr s2 = state.pstate["var"];
  auto r_arr = make_shared<NdArray>(param->shape());
  T *m = s1->cast_data_and_get_pointer<T>(this->ctx_);
  T *v = s2->cast_data_and_get_pointer<T>(this->ctx_);
  const T *g = param->get_grad_pointer<T>(this->ctx_);
  T *w = param->cast_data_and_get_pointer<T>(this->ctx_);
  T *r = r_arr->cast(dtype, this->ctx_)->template pointer<T>();

  const float correction_1 = bias_correction_ ? 1 - pow(beta1_, t) : 1.0f;
  const float correction_2 = bias_correction_ ? 1 - pow(beta2_, t) : 1.0f;

  for (int s = 0; s < size; ++s) {
    // Updating running mean and var.
    m[s] = this->beta1_ * m[s] + (1 - this->beta1_) * g[s];
    v[s] = this->beta2_ * v[s] + (1 - this->beta2_) * g[s] * g[s];
    r[s] =
        (m[s] / correction_1) / (std::sqrt(v[s] / correction_2) + this->eps_);
    r[s] = r[s] + this->weight_decay_rate_ * w[s];
  }

  // Calculate norm
  auto g_norm = std::sqrt(std::accumulate(
      r, r + size, T(0), [](const T &acc, const T &g) { return acc + g * g; }));
  auto v_norm = std::sqrt(std::accumulate(
      w, w + size, T(0), [](const T &acc, const T &d) { return acc + d * d; }));
  if (v_norm < this->gamma_l_) {
    v_norm = this->gamma_l_;
  }
  if (v_norm > this->gamma_u_) {
    v_norm = this->gamma_u_;
  }

  auto local_lr = 1.0;
  if (g_norm > this->eps_) {
    local_lr = v_norm / g_norm;
  }

  // Update weight
  for (int s = 0; s < size; ++s) {
    w[s] -= this->eta_ * local_lr * r[s];
  }
}

NBLA_DEF_WEIGHT_DECAY(Lamb, weight_decay_cpu);
NBLA_DEF_CLIP_GRAD_BY_NORM(Lamb, clip_grad_by_norm_cpu);
NBLA_DEF_CHECK_INF_GRAD(Lamb, check_inf_grad_cpu);
NBLA_DEF_CHECK_NAN_GRAD(Lamb, check_nan_grad_cpu);
NBLA_DEF_CHECK_INF_OR_NAN_GRAD(Lamb, check_inf_or_nan_grad_cpu);
NBLA_DEF_SCALE_GRAD(Lamb, scale_grad_impl_cpu);
}
