// Copyright 2019,2020,2021 Sony Corporation.
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

#ifndef __NBLA_SOLVER_CLIP_GRAD_HPP__
#define __NBLA_SOLVER_CLIP_GRAD_HPP__

#include <nbla/context.hpp>
#include <nbla/variable.hpp>

#include <memory>

namespace nbla {

template <typename T>
void clip_grad_by_norm_cpu(const Context &ctx, const shared_ptr<Variable> param,
                           float clip_norm) {
  Size_t size = param->size();
  T *grad = param->cast_grad_and_get_pointer<T>(ctx);
  T sum = 0;
  for (int i = 0; i < size; ++i)
    sum += grad[i] * grad[i];
  // sum > 0.0 is to avoid zero sqrt
  if (sum > 0.0 && sum > clip_norm * clip_norm) {
    T norm = std::sqrt(sum);
    for (int i = 0; i < size; ++i)
      grad[i] = clip_norm * grad[i] / norm;
  }
}
}
#endif
