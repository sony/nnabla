// Copyright 2023 Sony Group Corporation.
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
#include <nbla/function/eye_like.hpp>
#include <nbla/variable.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(EyeLike, int);

template <typename T>
void EyeLike<T>::setup_impl(const Variables &inputs, const Variables &outputs) {
  NBLA_CHECK(inputs[0]->ndim() == 2, error_code::value,
             "inputs[0]->ndim() must be 2.");
  outputs[0]->reshape(inputs[0]->shape(), true);
}

template <typename T>
void EyeLike<T>::forward_impl(const Variables &inputs,
                              const Variables &outputs) {
  const Shape_t shape = outputs[0]->shape();

  outputs[0]->data()->zero();
  T *y = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_, false);

  Size_t start_row, end_row;
  if (k_ >= 0) {
    start_row = 0;
    end_row = std::min(shape[0], shape[1] - k_);
  } else {
    start_row = std::abs(k_);
    end_row = start_row + std::min(shape[0] - std::abs(k_), shape[1]);
  }

  for (Size_t i = start_row; i < end_row; i++) {
    y[i * shape[1] + i + k_] = 1;
  }
}

template <typename T>
void EyeLike<T>::backward_impl(const Variables &inputs,
                               const Variables &outputs,
                               const vector<bool> &propagate_down,
                               const vector<bool> &accum) {
  // pass
}
}
