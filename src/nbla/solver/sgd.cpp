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
#include <nbla/solver/sgd.hpp>

namespace nbla {
using std::shared_ptr;
using std::make_shared;

NBLA_REGISTER_SOLVER_SOURCE(Sgd, float);

template <typename T>
Sgd<T>::Sgd(const Context &ctx, float lr) : BaseSolver<T>(ctx), lr_(lr) {}

template <typename T> Sgd<T>::~Sgd() {}

template <typename T>
void Sgd<T>::set_state_impl(const string &key, VariablePtr param) {}
template <typename T> void Sgd<T>::remove_state_impl(const string &key) {}

template <typename T>
void Sgd<T>::update_impl(const string &key, VariablePtr param) {
  Size_t size = param->size();

  const T *grad = param->get_grad_pointer<T>(this->ctx_);
  T *data = param->cast_data_and_get_pointer<T>(this->ctx_);
  std::transform(grad, grad + size, data, data,
                 [this](T g, T x) { return x - lr_ * g; });
}

// Template instanciation
template class Sgd<float>;
}
