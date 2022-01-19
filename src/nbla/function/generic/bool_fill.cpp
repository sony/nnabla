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
#include <nbla/function/bool_fill.hpp>
#include <nbla/imperative.hpp>
#include <nbla/variable.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(BoolFill, float);

template <typename T>
void BoolFill<T>::setup_impl(const Variables &inputs,
                             const Variables &outputs) {
  auto data = inputs[0];
  auto mask = inputs[1];
  auto output = outputs[0];
  NBLA_CHECK(data->ndim() >= mask->ndim(), error_code::value,
             "Input dim (%s) >= mask ndim (%s) must hold.", data->ndim(),
             mask->ndim());

  // Broadcast case
  if (data->shape() != mask->shape()) {
    vector<int> tshape;
    for (auto dim : data->shape()) {
      tshape.push_back(dim);
    }
    broadcast_func_ = create_Broadcast(this->ctx_, tshape);
  }

  output->reshape(data->shape(), true);
}

template <typename T>
void kernel_bool_fill_data_forward(const int N, T *output, const T *data,
                                   const T *mask, const float value) {
  for (int i = 0; i < N; ++i) {
    output[i] = (mask[i] != 0) ? T(value) : data[i];
  }
}

template <typename T>
void BoolFill<T>::forward_impl(const Variables &inputs,
                               const Variables &outputs) {
  auto data = inputs[0]->get_data_pointer<T>(this->ctx_);
  auto mask = inputs[1]->get_data_pointer<T>(this->ctx_);
  auto output = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_);
  auto N = inputs[0]->size();

  if (broadcast_func_ != nullptr) {
    Variable bmask;
    nbla::execute(broadcast_func_, {inputs[1]}, {&bmask});
    mask = bmask.get_data_pointer<T>(this->ctx_);
    kernel_bool_fill_data_forward(N, output, data, mask, value_);
  } else {
    kernel_bool_fill_data_forward(N, output, data, mask, value_);
  }
}

template <typename T, bool accum = false>
void kernel_bool_fill_data_backward(const int N, T *g_data, const T *g_output,
                                    const T *mask) {
  for (int i = 0; i < N; ++i) {
    auto mask_i = T(mask[i] != T(0));
    g_data[i] = accum ? g_data[i] + g_output[i] * (T(1) - mask_i)
                      : g_output[i] * (T(1) - mask_i);
  }
}

template <typename T>
void BoolFill<T>::backward_impl(const Variables &inputs,
                                const Variables &outputs,
                                const vector<bool> &propagate_down,
                                const vector<bool> &accum) {
  if (!(propagate_down[0] || propagate_down[1])) {
    return;
  }

  auto mask = inputs[1]->get_data_pointer<T>(this->ctx_);

  auto g_data = inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_, !accum[0]);
  auto g_output = outputs[0]->get_grad_pointer<T>(this->ctx_);
  auto N = inputs[0]->size();

  if (propagate_down[0]) {
    if (broadcast_func_ != nullptr) {
      Variable bmask;
      nbla::execute(broadcast_func_, {inputs[1]}, {&bmask});
      mask = bmask.get_data_pointer<T>(this->ctx_);
      auto kernel = accum[0] ? kernel_bool_fill_data_backward<T, true>
                             : kernel_bool_fill_data_backward<T, false>;
      kernel(N, g_data, g_output, mask);
    } else {
      auto kernel = accum[0] ? kernel_bool_fill_data_backward<T, true>
                             : kernel_bool_fill_data_backward<T, false>;
      kernel(N, g_data, g_output, mask);
    }
  }
}
}
