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

#ifndef __NBLA_SOLVER_WEIGHT_DECAY_HPP__
#define __NBLA_SOLVER_WEIGHT_DECAY_HPP__

namespace nbla {

template <typename T>
void weight_decay_cpu(const Context &ctx, const shared_ptr<Variable> param,
                      float decay_rate) {
  Size_t size = param->size();
  const T *data = param->get_data_pointer<T>(ctx);
  T *grad = param->cast_grad_and_get_pointer<T>(ctx);
  std::transform(data, data + size, grad, grad,
                 [decay_rate](T x, T g) { return g + decay_rate * x; });
}
}
#endif
