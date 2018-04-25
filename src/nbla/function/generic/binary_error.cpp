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

/** BinaryError
 */
#include <nbla/array.hpp>
#include <nbla/function/binary_error.hpp>
#include <nbla/variable.hpp>

#include <algorithm>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(BinaryError);

template <typename T>
void BinaryError<T>::setup_impl(const Variables &inputs,
                                const Variables &outputs) {
  NBLA_CHECK(inputs[0]->shape() == inputs[1]->shape(), error_code::value,
             "Dimensions of inputs must match. "
             "inputs[0]: %s != inputs[1]: %s.",
             string_join(inputs[0]->shape(), string(", ")).c_str(),
             string_join(inputs[1]->shape(), string(", ")).c_str());
  outputs[0]->reshape(inputs[0]->shape(), true);
}

template <typename T>
void BinaryError<T>::forward_impl(const Variables &inputs,
                                  const Variables &outputs) {
  const T *x0 = inputs[0]->get_data_pointer<T>(this->ctx_);
  const T *x1 = inputs[1]->get_data_pointer<T>(this->ctx_);
  T *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_, true);
  const Size_t size = inputs[0]->size();
  for (int s = 0; s < size; s++) {
    y[s] = (x0[s] >= 0.5) != (x1[s] >= 0.5);
  }
}

template <typename T>
void BinaryError<T>::backward_impl(const Variables &inputs,
                                   const Variables &outputs,
                                   const vector<bool> &propagate_down,
                                   const vector<bool> &accum) {
  // Not supported.
}

} // namespace nbla
