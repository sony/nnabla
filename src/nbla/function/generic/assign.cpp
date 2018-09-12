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


#include <nbla/array.hpp>
#include <nbla/common.hpp>
#include <nbla/function/assign.hpp>
#include <nbla/variable.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(Assign);

template <typename T>
void Assign<T>::setup_impl(const Variables &inputs,
                               const Variables &outputs) {
  NBLA_CHECK(inputs[0]->shape() == inputs[1]->shape(), error_code::value,
             "Dimensions of inputs must match. "
             "inputs[0]: %s != inputs[1]: %s.",
             string_join(inputs[0]->shape(), string(", ")).c_str(),
             string_join(inputs[1]->shape(), string(", ")).c_str());
  outputs[0]->reshape(inputs[0]->shape(), true);
}

template <typename T>
void Assign<T>::forward_impl(const Variables &inputs,
                               const Variables &outputs) {
  // remove const from below to update destination variable
  T* dst = inputs[0]->cast_data_and_get_pointer<T>(this->ctx_, true);
  const T* src = inputs[1]->get_data_pointer<T>(this->ctx_);
  T* y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_, true);
  for (int s = 0; s < inputs[0]->size(); s++) {
    dst[s] = src[s];
    y[s] = src[s];
  }
}


template <typename T>
void Assign<T>::backward_impl(const Variables &inputs,
                                const Variables &outputs,
                                const vector<bool> &propagate_down,
                                const vector<bool> &accum) {
  // assign function does not anything.
  return;
}
}
