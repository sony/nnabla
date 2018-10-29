// Copyright (c) 2018 Sony Corporation. All Rights Reserved.
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
#include <nbla/function/arange.hpp>
#include <nbla/variable.hpp>

#include <limits>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(Arange, float, float, float);

template <typename T>
void Arange<T>::setup_impl(const Variables &inputs, const Variables &outputs) {
  NBLA_CHECK(this->step_ != 0, error_code::value,
             "step argument must not be zero");

  Size_t count = 0;
  if (step_ < 0) {
    for (float value = start_; value > stop_; value += step_) {
      count++;
    }
  } else {
    for (float value = start_; value < stop_; value += step_) {
      count++;
    }
  }
  outputs[0]->reshape(Shape_t{count}, true);
}

template <typename T>
void Arange<T>::forward_impl(const Variables &inputs,
                             const Variables &outputs) {
  Variable &y = *outputs[0];

  auto y_data = y.cast_data_and_get_pointer<T>(this->ctx_, true);
  auto value = this->start_;

  for (Size_t i = 0; i < y.size(); i++) {
    y_data[i] = value;
    value += this->step_;
  }
}

template <typename T>
void Arange<T>::backward_impl(const Variables &inputs, const Variables &outputs,
                              const vector<bool> &propagate_down,
                              const vector<bool> &accum) {
  // pass
}

} // namespace nbla
