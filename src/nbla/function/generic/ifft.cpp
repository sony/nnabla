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
#include <nbla/function/ifft.hpp>
#include <nbla/variable.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(IFFT, int, bool);

template <typename T>
void IFFT<T>::setup_impl(const Variables &inputs, const Variables &outputs) {
  outputs[0]->reshape(inputs[0]->shape(), true);
}

template <typename T>
void IFFT<T>::forward_impl(const Variables &inputs, const Variables &outputs) {
  NBLA_ERROR(error_code::not_implemented,
             "IFFT forward not implemented in CPU yet.");
}

template <typename T>
void IFFT<T>::backward_impl(const Variables &inputs, const Variables &outputs,
                            const vector<bool> &propagate_down,
                            const vector<bool> &accum) {
  NBLA_ERROR(error_code::not_implemented,
             "IFFT backward not implemented in CPU yet.");
}
}
