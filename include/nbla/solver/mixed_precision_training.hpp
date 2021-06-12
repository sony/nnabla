// Copyright 2018,2019,2020,2021 Sony Corporation.
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

#ifndef __NBLA_SOLVER_MIXED_PRECISION_TRAINING_HPP__
#define __NBLA_SOLVER_MIXED_PRECISION_TRAINING_HPP__
#include <nbla/array.hpp>
#include <nbla/context.hpp>
#include <nbla/variable.hpp>

namespace nbla {

template <typename T>
bool check_inf_grad_cpu(const Context &ctx, const shared_ptr<Variable> param) {
  Size_t size = param->size();
  const T *grad = param->get_grad_pointer<T>(ctx);
  for (int i = 0; i < size; i++) {
    if (std::isinf(grad[i]))
      return true;
  }
  return false;
}

template <typename T>
bool check_nan_grad_cpu(const Context &ctx, const shared_ptr<Variable> param) {
  Size_t size = param->size();
  const T *grad = param->get_grad_pointer<T>(ctx);
  for (int i = 0; i < size; i++) {
    if (std::isnan(grad[i]))
      return true;
  }
  return false;
}

template <typename T>
bool check_inf_or_nan_grad_cpu(const Context &ctx,
                               const shared_ptr<Variable> param) {
  Size_t size = param->size();
  const T *grad = param->get_grad_pointer<T>(ctx);
  for (int i = 0; i < size; i++) {
    if (std::isinf(grad[i]) || std::isnan(grad[i])) {
      return true;
    }
  }
  return false;
}

template <typename T>
void scale_grad_impl_cpu(const Context &ctx, const shared_ptr<Variable> param,
                         float scale) {
  Size_t size = param->size();
  const T *data = param->get_data_pointer<T>(ctx);
  T *grad = param->cast_grad_and_get_pointer<T>(ctx);
  std::transform(data, data + size, grad, grad,
                 [scale](T x, T g) { return g * scale; });
}
}
#endif
