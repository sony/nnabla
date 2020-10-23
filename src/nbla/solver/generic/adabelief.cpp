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
#include <limits>
#include <nbla/solver/adabelief.hpp>
#include <nbla/solver/clip_grad.hpp>
#include <nbla/solver/mixed_precision_training.hpp>
#include <nbla/solver/weight_decay.hpp>

namespace nbla {
using std::make_shared;
using std::shared_ptr;

NBLA_REGISTER_SOLVER_SOURCE(AdaBelief, float, float, float, float, bool, bool,
                            bool, bool);

template <typename T>
AdaBelief<T>::AdaBelief(const Context &ctx, float alpha, float beta1,
                        float beta2, float eps, bool amsgrad,
                        bool weight_decouple, bool fixed_decay, bool rectify)
    : Solver(ctx), alpha_(alpha), beta1_(beta1), beta2_(beta2), eps_(eps),
      amsgrad_(amsgrad), weight_decouple_(weight_decouple),
      fixed_decay_(fixed_decay), rectify_(rectify) {}

template <typename T> AdaBelief<T>::~AdaBelief() {}

template <typename T>
void AdaBelief<T>::set_state_impl(const string &key, VariablePtr param) {
  // Implement here
}
template <typename T> void AdaBelief<T>::remove_state_impl(const string &key) {
  // Implement here
}

template <typename T>
void AdaBelief<T>::update_impl(const string &key, VariablePtr param) {
  // Implement here
}

NBLA_DEF_WEIGHT_DECAY(AdaBelief, weight_decay_cpu);
NBLA_DEF_CLIP_GRAD_BY_NORM(AdaBelief, clip_grad_by_norm_cpu);
NBLA_DEF_CHECK_INF_GRAD(AdaBelief, check_inf_grad_cpu);
NBLA_DEF_CHECK_NAN_GRAD(AdaBelief, check_nan_grad_cpu);
NBLA_DEF_CHECK_INF_OR_NAN_GRAD(AdaBelief, check_inf_or_nan_grad_cpu);
NBLA_DEF_SCALE_GRAD(AdaBelief, scale_grad_impl_cpu);
}
