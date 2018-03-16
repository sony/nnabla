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

#include <nbla/function/top_k_select.hpp>
#include <nbla/utils/top_k.hpp>
#include <nbla/variable.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_SOURCE(TopKSelect, int, int, int);

template <typename T>
void TopKSelect<T>::setup_impl(const Variables &inputs,
                               const Variables &outputs) {
  NBLA_CHECK(base_axis_ < inputs[0]->shape().size(), error_code::value,
             "axis must be less than ndim of inputs[0]. "
             "axis %d >= ndim of inputs[0]: %d.",
             base_axis_, inputs[0]->shape().size());

  outputs[0]->reshape(inputs[0]->shape(), true);
}

template <typename T>
void TopKSelect<T>::forward_impl(const Variables &inputs,
                                 const Variables &outputs) {
  const size_t size_from_axis = inputs[0]->size(this->base_axis_);
  const T *input_data_ptr = inputs[0]->get_data_pointer<T>(this->ctx_);
  const T *input_data_ptr_end = input_data_ptr + inputs[0]->size();

  const auto K = std::abs(this->k_data_);
  if ((K == 0) || (K >= size_from_axis)) {
    T *output_data_ptr = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_);
    memcpy(output_data_ptr, input_data_ptr, inputs[0]->size() * sizeof(T));
    return;
  }

  outputs[0]->data()->zero();
  T *output_data_ptr = outputs[0]->cast_data_and_get_pointer<T>(this->ctx_);

  std::vector<size_t> top_k_idx(K);
  while (input_data_ptr != input_data_ptr_end) {
    if (this->k_data_ < 0) {
      top_k_abs<T>(input_data_ptr, size_from_axis, K, top_k_idx.data());
    } else {
      top_k<T>(input_data_ptr, size_from_axis, K, top_k_idx.data());
    }
    for (int k = 0; k < K; k++) {
      const auto i = top_k_idx[k];
      output_data_ptr[i] += input_data_ptr[i];
    }
    input_data_ptr += size_from_axis;
    output_data_ptr += size_from_axis;
  }
}

template <typename T>
void TopKSelect<T>::backward_impl(const Variables &inputs,
                                  const Variables &outputs,
                                  const vector<bool> &propagate_down,
                                  const vector<bool> &accum) {
  if (!propagate_down[0]) {
    return;
  }

  const size_t size_from_axis = outputs[0]->size(this->base_axis_);
  const T *output_grad_ptr = outputs[0]->get_grad_pointer<T>(this->ctx_);
  const T *output_grad_ptr_end = output_grad_ptr + outputs[0]->size();

  const auto K = std::abs(this->k_grad_);
  if ((K == 0) || (K >= size_from_axis)) {
    T *input_grad_ptr = inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_);
    memcpy(input_grad_ptr, output_grad_ptr, inputs[0]->size() * sizeof(T));
    return;
  }

  if (!accum[0]) {
    inputs[0]->grad()->zero();
  }
  T *input_grad_ptr = inputs[0]->cast_grad_and_get_pointer<T>(this->ctx_);

  std::vector<size_t> top_k_idx(K);
  while (output_grad_ptr != output_grad_ptr_end) {
    if (this->k_grad_ < 0) {
      top_k_abs<T>(output_grad_ptr, size_from_axis, K, top_k_idx.data());
    } else {
      top_k<T>(output_grad_ptr, size_from_axis, K, top_k_idx.data());
    }
    for (int k = 0; k < K; k++) {
      const auto i = top_k_idx[k];
      input_grad_ptr[i] += output_grad_ptr[i];
    }
    output_grad_ptr += size_from_axis;
    input_grad_ptr += size_from_axis;
  }
}

// Template instantiation
template class TopKSelect<float>;

} // namespace nbla
