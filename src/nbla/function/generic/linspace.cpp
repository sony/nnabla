// Copyright 2021 Sony Group Corporation.
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
#include <nbla/function/linspace.hpp>
#include <nbla/variable.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(Linspace, float, float, int);

template <typename T>
void Linspace<T>::setup_impl(const Variables &inputs,
                             const Variables &outputs) {
  NBLA_CHECK(this->num_ >= 0, error_code::value,
             "num argument must not be negative");
  step_ =
      this->num_ > 1
          ? static_cast<double>(this->stop_ - this->start_) / (this->num_ - 1)
          : 0.0;
  outputs[0]->reshape(Shape_t{num_}, true);
}

template <typename T>
void Linspace<T>::forward_impl(const Variables &inputs,
                               const Variables &outputs) {
  auto y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_, true);
  for (int i = 0; i < this->num_; i++) {
    y[i] = this->start_ + this->step_ * i;
  }
}

template <typename T>
void Linspace<T>::backward_impl(const Variables &inputs,
                                const Variables &outputs,
                                const vector<bool> &propagate_down,
                                const vector<bool> &accum) {
  // pass
}
}
