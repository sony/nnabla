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
#include <nbla/solver/base_solver.hpp>

namespace nbla {
using std::shared_ptr;
using std::make_shared;

template <typename T>
BaseSolver<T>::BaseSolver(const Context &ctx) : Solver(ctx) {}

template <typename T> BaseSolver<T>::~BaseSolver() {}

template <typename T>
void BaseSolver<T>::set_state_impl(const string &key, VariablePtr param) {}
template <typename T>
void BaseSolver<T>::remove_state_impl(const string &key) {}

template <typename T>
void BaseSolver<T>::weight_decay_impl(const string &key, VariablePtr param,
                                      float decay_rate) {
  Size_t size = param->size();
  const T *data = param->get_data_pointer<T>(ctx_);
  T *grad = param->cast_grad_and_get_pointer<T>(ctx_);
  std::transform(data, data + size, grad, grad,
                 [this, decay_rate](T x, T g) { return g + decay_rate * x; });
}
// Template instanciation
template class BaseSolver<float>;
}
