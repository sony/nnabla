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
#include <nbla/function/gru.hpp>
#include <nbla/variable.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(GRU, int, float, bool, bool);

template <typename T>
void GRU<T>::setup_impl(const Variables &inputs, const Variables &outputs) {
  NBLA_ERROR(error_code::not_implemented, "GRU is not implemented for CPU.")
}

template <typename T>
void GRU<T>::forward_impl(const Variables &inputs, const Variables &outputs) {
  if (this->training_) {
    forward_impl_training(inputs, outputs);
  } else {
    forward_impl_inference(inputs, outputs);
  }
}

template <typename T>
void GRU<T>::forward_impl_training(const Variables &inputs,
                                   const Variables &outputs) {
  NBLA_ERROR(error_code::not_implemented, "GRU is not implemented for CPU.")
}

template <typename T>
void GRU<T>::forward_impl_inference(const Variables &inputs,
                                    const Variables &outputs) {
  NBLA_ERROR(error_code::not_implemented, "GRU is not implemented for CPU.")
}

template <typename T>
void GRU<T>::backward_impl(const Variables &inputs, const Variables &outputs,
                           const vector<bool> &propagate_down,
                           const vector<bool> &accum) {
  NBLA_ERROR(error_code::not_implemented, "GRU is not implemented for CPU.")
}
}
