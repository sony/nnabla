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

/** Constant
 */
#include <nbla/array.hpp>
#include <nbla/function/constant.hpp>
#include <nbla/variable.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(Constant, float, const vector<int> &);

template <typename T>
void Constant<T>::setup_impl(const Variables &inputs,
                             const Variables &outputs) {
  Shape_t out_shape(shape_.begin(), shape_.end());
  outputs[0]->reshape(out_shape, true);
}

template <typename T>
void Constant<T>::forward_impl(const Variables &inputs,
                               const Variables &outputs) {
  if (val_ == 0) {
    outputs[0]->data()->zero();
    return;
  }
  outputs[0]->data()->fill(val_);
}

template <typename T>
void Constant<T>::backward_impl(const Variables &inputs,
                                const Variables &outputs,
                                const vector<bool> &propagate_down,
                                const vector<bool> &accum) {
  // pass
}

} // namespace nbla
