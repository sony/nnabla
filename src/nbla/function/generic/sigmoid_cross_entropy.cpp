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

// sigmoid_cross_entropy.cpp

#include <nbla/array.hpp>
#include <nbla/function/sigmoid.hpp>
#include <nbla/function/sigmoid_cross_entropy.hpp>
#include <nbla/variable.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(SigmoidCrossEntropy);

template <typename T, typename Tl>
void SigmoidCrossEntropy<T, Tl>::setup_impl(const Variables &inputs,
                                            const Variables &outputs) {

  NBLA_CHECK(inputs[0]->shape() == inputs[1]->shape(), error_code::value,
             "Dimensions of inputs must match. "
             "inputs[0]: %s != inputs[1]: %s.",
             string_join(inputs[0]->shape(), string(", ")).c_str(),
             string_join(inputs[1]->shape(), string(", ")).c_str());
  outputs[0]->reshape(inputs[0]->shape(), true);
}

template <typename T, typename Tl>
void SigmoidCrossEntropy<T, Tl>::forward_impl(const Variables &inputs,
                                              const Variables &outputs) {
  const T *x0 = inputs[0]->get_data_pointer<T>(this->ctx_);
  const Tl *x1 = inputs[1]->get_data_pointer<Tl>(this->ctx_);
  T *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_, true);
  const Size_t size = inputs[0]->size();
  for (int s = 0; s < size; s++) {
    y[s] = -(x0[s] * (x1[s] - (x0[s] >= 0)) -
             std::log(1 + std::exp(x0[s] - 2 * x0[s] * (x0[s] >= 0))));
  }
}

template <typename T, typename Tl>
void SigmoidCrossEntropy<T, Tl>::backward_impl(
    const Variables &inputs, const Variables &outputs,
    const vector<bool> &propagate_down, const vector<bool> &accum) {
  NBLA_CHECK(!propagate_down[1], error_code::value,
             "Label can not be propagated down.");
  if (!propagate_down[0]) {
    return;
  }
  const T *dy = outputs[0]->get_grad_pointer<T>(this->ctx_);
  const T *x0 = inputs[0]->get_data_pointer<T>(this->ctx_);
  const Tl *x1 = inputs[1]->get_data_pointer<Tl>(this->ctx_);
  const Size_t size = inputs[0]->size();
  if (propagate_down[0]) {
    T *dx0 = inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_, !accum[0]);
    for (int s = 0; s < size; ++s) {
      const T tmp = dy[s] * (1 / (1 + std::exp(-x0[s])) - x1[s]);
      if (accum[0])
        dx0[s] += tmp;
      else
        dx0[s] = tmp;
    }
  }
}
}
