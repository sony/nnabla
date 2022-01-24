// Copyright 2020,2021 Sony Corporation.
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

#include <memory>
#include <nbla/array.hpp>
#include <nbla/common.hpp>
#include <nbla/function/mul_n.hpp>
#include <nbla/variable.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(MulN);

template <typename T>
void MulN<T>::setup_impl(const Variables &inputs, const Variables &outputs) {
  NBLA_CHECK(inputs.size() >= 1, error_code::value,
             "at least one input should be given");
  for (Variables::size_type i = 1; i < inputs.size(); i++) {
    NBLA_CHECK(inputs[0]->shape() == inputs[i]->shape(), error_code::value,
               "shape of all inputs must be shame");
  }
  //
  // Set all inputs initially active. This function allows configuration of
  // inputs as active/inactive via Python set_active_input_mask() function.
  //
  cg_input_mask.assign(inputs.size(), true);

  outputs.at(0)->reshape(inputs[0]->shape(), true);
}

template <typename T>
void MulN<T>::forward_impl(const Variables &inputs, const Variables &outputs) {
  auto y = outputs.at(0)->cast_data_and_get_pointer<T>(this->ctx_, true);
  Variables::size_type i = 0;

  // Copy from first active input.
  for (; i < inputs.size(); i++) {
    if (is_active_input(i)) {
      auto x = inputs[i]->get_data_pointer<T>(ctx_);
      for (Size_t k = 0; k < outputs[0]->size(); k++) {
        y[k] = x[k];
      }
      break;
    }
  }
  // Multiply remaining active inputs.
  for (i++; i < inputs.size(); i++) {
    if (is_active_input(i)) {
      auto x = inputs[i]->get_data_pointer<T>(ctx_);
      for (Size_t k = 0; k < outputs[0]->size(); k++) {
        y[k] *= x[k];
      }
    }
  }
}

template <typename T>
void MulN<T>::backward_impl(const Variables &inputs, const Variables &outputs,
                            const vector<bool> &propagate_down,
                            const vector<bool> &accum) {
  auto dy = outputs.at(0)->get_grad_pointer<T>(ctx_);
  auto y = outputs.at(0)->get_data_pointer<T>(this->ctx_);

  for (Variables::size_type i = 0; i < inputs.size(); i++) {
    if (is_active_input(i) && propagate_down.at(i)) {
      auto x = inputs[i]->get_data_pointer<T>(this->ctx_);
      auto dx = inputs[i]->cast_grad_and_get_pointer<T>(ctx_, !(accum.at(i)));
      if (accum.at(i)) {
        for (Size_t k = 0; k < outputs[0]->size(); k++) {
          dx[k] += dy[k] * y[k] / x[k];
        }
      } else {
        for (Size_t k = 0; k < outputs[0]->size(); k++) {
          dx[k] = dy[k] * y[k] / x[k];
        }
      }
    }
  }
}
}
